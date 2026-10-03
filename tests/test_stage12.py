"""Stage 1/2 unit checks (CPU). Run: DATA_ROOT=... uv run python tests/test_stage12.py"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402
from stage0.attention import AttentionRecorder  # noqa: E402
from stage0.attribution import box_for_center_crop, crop_maps  # noqa: E402
from stage0.datasets import Stage0Dataset, build_train_transform  # noqa: E402
from stage0.engine import make_loader, numpy_collate_box  # noqa: E402
from stage0.flops import criterion_gflops, teacher_gflops  # noqa: E402
from stage0.masking import StudentMaskSelector, num_keep, random_idx  # noqa: E402
from stage0.models import create_student, create_teacher, teacher_forward  # noqa: E402


def test_teacher_forward_keep_idx():
    torch.manual_seed(0)
    t = create_teacher(10, pretrained=False).eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        ref = t(x)
        assert torch.equal(teacher_forward(t, x), ref), "keep_idx=None must equal teacher(x)"
        full = torch.arange(196).expand(3, -1)
        assert torch.equal(teacher_forward(t, x, full), ref), "keep_idx=arange(196) must equal teacher(x)"
        perm = torch.stack([torch.randperm(196) for _ in range(3)])
        assert torch.allclose(teacher_forward(t, x, perm), ref, atol=1e-5, rtol=0), "token order must not matter"
        sub = torch.stack([torch.randperm(196)[:50] for _ in range(3)])
        out = teacher_forward(t, x, sub)
        assert out.shape == ref.shape and not torch.allclose(out, ref)
        # per-sample selection: row b only depends on its own indices
        out1 = teacher_forward(t, x[1:2], sub[1:2])
        assert torch.allclose(out[1:2], out1, atol=1e-5, rtol=0)


def _direct_attention(model, x):
    """Attention probabilities captured from the non-fused path (input of attn_drop)."""
    caps = []
    for blk in model.blocks:
        blk.attn.fused_attn = False
    hs = [blk.attn.attn_drop.register_forward_hook(lambda m, i, o: caps.append(i[0].detach()))
          for blk in model.blocks]
    with torch.no_grad():
        out = model(x)
    for h in hs:
        h.remove()
    for blk in model.blocks:
        blk.attn.fused_attn = True
    return out, caps


def test_hook_attention_matches_direct():
    torch.manual_seed(0)
    for model in (create_student(10, pretrained=False).eval(), create_teacher(10, pretrained=False).eval()):
        x = torch.randn(2, 3, 224, 224)
        out_ref, caps = _direct_attention(model, x)
        with AttentionRecorder(model, need=("cls_last", "rollout")) as rec, torch.no_grad():
            out = model(x)
        assert torch.equal(out, model(x).detach()) or torch.allclose(out, out_ref, atol=1e-5)
        last = caps[-1].mean(1)[:, 0, 1:]
        assert torch.allclose(rec.cls_last, last, atol=1e-6), (rec.cls_last - last).abs().max()
        R = None
        for a in caps:
            a_hat = 0.5 * a.mean(1) + 0.5 * torch.eye(a.shape[-1])
            R = a_hat if R is None else a_hat @ R
        assert torch.allclose(rec.rollout, R[:, 0, 1:], atol=1e-6), (rec.rollout - R[:, 0, 1:]).abs().max()
        with AttentionRecorder(model, need=("cls_last",)) as rec2, torch.no_grad():
            model(x)
        assert torch.allclose(rec2.cls_last, last, atol=1e-6)
    # hooks are removed on exit and do not change outputs
    s = create_student(10, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        a = s(x)
        with AttentionRecorder(s, need=("rollout",)):
            b = s(x)
        assert torch.equal(a, b)
    assert all(len(blk.attn._forward_hooks) == 0 for blk in s.blocks)


def test_box_transform_bit_identical(data_root):
    ds_ref = Stage0Dataset(data_root, "cub", "train", build_train_transform(), limit=24)
    ds_box = Stage0Dataset(data_root, "cub", "train", build_train_transform(return_box=True), limit=24)
    assert isinstance(build_train_transform(), transforms.Compose)
    for seed in (0, 7):
        ds_ref.set_aug_seed(seed)
        ds_box.set_aug_seed(seed)
        for i in range(len(ds_ref)):
            a, _, _ = ds_ref[i]
            (b, box), _, _ = ds_box[i]
            assert torch.equal(a, b), f"image {i} differs"
            ii, jj, h, w, flip, W, H = box.tolist()
            assert 0 <= ii and 0 <= jj and ii + h <= H and jj + w <= W and flip in (0.0, 1.0)
    # loader path (numpy collate with boxes)
    ds_box.set_aug_seed(3)
    x, y, g, box = next(iter(make_loader(ds_box, 8, 0, collate_fn=numpy_collate_box)))
    assert x.shape == (8, 3, 224, 224) and box.shape == (8, 7) and y.dtype == torch.int64


def test_crop_maps():
    torch.manual_seed(0)
    m = torch.rand(2, 14, 14)
    W, H = 300.0, 200.0
    full = torch.tensor([[0, 0, H, W, 0, W, H]] * 2)
    assert torch.allclose(crop_maps(m, full), m, atol=1e-6), "full-image box must be the identity"
    flipped = full.clone()
    flipped[:, 4] = 1
    assert torch.allclose(crop_maps(m, flipped), m.flip(-1), atol=1e-6)
    # top-left quadrant box -> upsampled top-left 7x7 block: cell centres fall between source cells
    q = torch.tensor([[0, 0, H / 2, W / 2, 0, W, H]] * 2)
    out = crop_maps(m, q)
    assert torch.allclose(out[:, ::2, ::2].mean(), m[:, :7, :7].mean(), atol=0.15)
    one_hot = torch.zeros(1, 14, 14)
    one_hot[0, 3, 10] = 1.0
    peak = crop_maps(one_hot, torch.tensor([[0, 0, H, W, 1, W, H]]))[0]
    assert peak.argmax().item() == 3 * 14 + (13 - 10), "flip mirrors columns"
    b = box_for_center_crop(400, 300)
    assert abs(b[2] - 300 * 224 / 256) < 1e-6


def test_random_idx_uses_only_its_generator():
    torch.manual_seed(123)
    np.random.seed(5)
    random.seed(5)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    g = torch.Generator().manual_seed(C.epoch_seed(0, 3, 4) + 7)
    idx = random_idx(4, num_keep(0.3), g, "cpu")
    assert idx.shape == (4, 59) and all(len(set(r.tolist())) == 59 for r in idx)
    after = (torch.get_rng_state(), np.random.get_state()[1], random.getstate())
    assert torch.equal(before[0], after[0]) and (before[1] == after[1]).all() and before[2] == after[2]
    g2 = torch.Generator().manual_seed(C.epoch_seed(0, 3, 4) + 7)
    assert torch.equal(idx, random_idx(4, 59, g2, "cpu")), "same seed -> same draw"
    assert [num_keep(k) for k in (1.0, 0.7, 0.5, 0.3, 0.15)] == [196, 137, 98, 59, 29]


def test_selector_maskedkd_matches_recorder():
    torch.manual_seed(0)
    s = create_student(10, pretrained=False).train()
    x = torch.randn(4, 3, 224, 224)
    with StudentMaskSelector("maskedkd", 0.5, s) as sel:
        torch.manual_seed(1)
        out = s(x)
        idx = sel.select(4, "cpu")
    assert idx.shape == (4, 98)
    torch.manual_seed(1)
    out2 = s(x)
    assert torch.equal(out, out2), "selector hooks must not change the training forward"
    assert all(len(blk.attn._forward_hooks) == 0 for blk in s.blocks)


def test_stage0_config_unchanged():
    """A kd / ce run's config has no Stage 2 keys (resuming Stage 0 runs must not see a config diff)."""
    from stage0 import train as T
    stage2_keys = {"mask_criterion", "keep", "keep_k", "ckpt_epochs"}
    for mode in ("kd", "ce", "teacher"):
        a = T.parse_args(["--mode", mode, "--dataset", "cub", "--seed", "0", "--lr", "1e-4"])
        cfg = T.build_config(a, Path("/o"), Path("/d"))
        assert not stage2_keys & set(cfg), (mode, stage2_keys & set(cfg))
        assert cfg["kd_alpha"] == (0.5 if mode == "kd" else None)
    a = T.parse_args(["--mode", "maskedkd", "--mask-criterion", "random", "--keep", "0.3", "--dataset", "cub",
                      "--seed", "0", "--lr", "1e-4"])
    cfg = T.build_config(a, Path("/o"), Path("/d"))
    assert cfg["keep_k"] == 59 and cfg["smoothing"] == 0.1 and cfg["mixup"] == 0.8 and cfg["kd_alpha"] == 0.5
    kd = T.build_config(T.parse_args(["--mode", "kd", "--dataset", "cub", "--seed", "0", "--lr", "1e-4"]),
                        Path("/o"), Path("/d"))
    diff = {k for k in set(cfg) | set(kd) if cfg.get(k) != kd.get(k)}
    assert diff == {"mode"} | stage2_keys, diff          # maskedkd = kd + token selection, nothing else
    assert C.mask_dirname("maskedkd", 0.3) == "maskedkd_k0.3" and C.mask_dirname("random", 0.15) == "random_k0.15"


def test_flops_scale_with_tokens():
    full, half = teacher_gflops(196), teacher_gflops(98)
    assert 15 < full < 20, full                       # DeiT-B ~17.6 GFLOPs
    assert 0.4 < half / full < 0.55, half / full
    assert teacher_gflops(29) < teacher_gflops(59) < half
    assert criterion_gflops("random") == 0.0
    assert 0 < criterion_gflops("maskedkd") < criterion_gflops("rollout") < 1.0
    assert criterion_gflops("teacher_oracle") > full


if __name__ == "__main__":
    test_teacher_forward_keep_idx()
    test_hook_attention_matches_direct()
    test_crop_maps()
    test_random_idx_uses_only_its_generator()
    test_selector_maskedkd_matches_recorder()
    test_flops_scale_with_tokens()
    test_stage0_config_unchanged()
    dr = os.environ.get("DATA_ROOT")
    if dr and (Path(dr) / "cub" / "stage0_split.json").exists():
        test_box_transform_bit_identical(dr)
    else:
        print("skip box-transform test (DATA_ROOT not prepared)")
    print("stage 1/2 unit tests passed")
