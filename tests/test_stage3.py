"""Stage 3 pilot (TAM) unit checks (CPU). Run: DATA_ROOT=... uv run python tests/test_stage3.py"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402
from stage0.datasets import Stage0Dataset, WithIndex, build_train_transform  # noqa: E402
from stage0.engine import make_loader, numpy_collate_box_index  # noqa: E402
from stage0.flops import teacher_gflops  # noqa: E402
from stage0.masking import (StudentMaskSelector, TamSelector, num_keep, tam_bucket_ks, tam_buckets,  # noqa: E402
                            tam_gap_ratio, tam_select)
from stage0.models import create_student, create_teacher, teacher_forward, teacher_forward_buckets  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _stage12_masking():
    """stage0/masking.py as it is on the stage12 branch (reference for bit-identical selections)."""
    for ref in ("stage12", "origin/stage12"):
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:stage0/masking.py"], capture_output=True, text=True)
        if r.returncode == 0:
            break
    else:
        return None
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(r.stdout)
    f.close()
    spec = importlib.util.spec_from_file_location("masking_stage12", f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_existing_criteria_bit_identical():
    old = _stage12_masking()
    if old is None:
        print("skip stage12 comparison (branch not available)")
        return
    torch.manual_seed(0)
    s = create_student(10, pretrained=False).train()
    x = torch.randn(6, 3, 224, 224)
    for keep in (0.5, 0.3, 0.15):
        torch.manual_seed(1)
        with StudentMaskSelector("maskedkd", keep, s) as new_sel:
            s(x)
            a = new_sel.select(6, "cpu")
        torch.manual_seed(1)
        with old.StudentMaskSelector("maskedkd", keep, s) as old_sel:
            s(x)
            b = old_sel.select(6, "cpu")
        assert torch.equal(a, b), f"maskedkd selection changed at keep {keep}"
        g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        a = StudentMaskSelector("random", keep, s).select(6, "cpu", g1)
        b = old.StudentMaskSelector("random", keep, s).select(6, "cpu", g2)
        assert torch.equal(a, b), f"random selection changed at keep {keep}"
    assert old.num_keep(0.3) == num_keep(0.3)


def test_tam_select_properties():
    torch.manual_seed(0)
    B = 16
    T, S = torch.rand(B, 196), torch.rand(B, 196)
    for k in (29, 59, 98, 137, 196):
        pool = torch.topk(T, min(196, 2 * k), dim=1).indices
        for g in (0.0, 0.1, 0.33, 0.5, 1.0):
            idx = tam_select(T, S, k, g)
            assert idx.shape == (B, k)
            for b in range(B):
                sel = set(idx[b].tolist())
                assert len(sel) == k, "duplicates"
                assert sel <= set(pool[b].tolist()), "index outside the teacher candidate pool"
            if g == 0.0:   # = top-k by S inside the pool
                want = torch.gather(pool, 1, torch.topk(torch.gather(S, 1, pool), k, dim=1).indices)
                assert all(set(idx[b].tolist()) == set(want[b].tolist()) for b in range(B))
            if g == 1.0:   # = top-k by T inside the pool (= global top-k by T)
                want = torch.topk(T, k, dim=1).indices
                assert all(set(idx[b].tolist()) == set(want[b].tolist()) for b in range(B))
    # gap tokens are teacher-important but less student-attended than the shared ones
    k, g = 59, 0.5
    idx = tam_select(T, S, k, g)
    k_sh = int(round((1 - g) * k))
    shared, gap = idx[:, :k_sh], idx[:, k_sh:]
    assert (torch.gather(S, 1, shared).min(1).values >= torch.gather(S, 1, gap).max(1).values).all()
    assert tam_gap_ratio(0, 100, 0.1, 0.5) == 0.1 and abs(tam_gap_ratio(99, 100, 0.1, 0.5) - 0.5) < 1e-12
    assert tam_gap_ratio(0, 100, 0.3, 0.3) == tam_gap_ratio(99, 100, 0.3, 0.3) == 0.3


def test_tam_buckets_and_forward():
    torch.manual_seed(0)
    for B in (2, 3, 7, 8, 16, 64):
        T = torch.rand(B, 196) ** 4          # varied concentration
        for keep in (0.5, 0.3, 0.15):
            k = num_keep(keep)
            b = tam_buckets(T, k, 0.33)
            rows = torch.cat([r for r, _ in b])
            assert sorted(rows.tolist()) == list(range(B)), "every image in exactly one bucket"
            mean = sum(r.numel() * kb for r, kb in b) / B
            assert abs(mean - k) <= 1.0, (B, k, mean)
            if B >= 3:
                assert len(b) == 3 and [kb for _, kb in b] == list(tam_bucket_ks(k, 0.33))
                lo_rows, hi_rows = b[0][0], b[2][0]
                t = T / T.sum(1, keepdim=True)
                c = torch.topk(t, k, dim=1).values.sum(1)
                assert c[lo_rows].min() >= c[hi_rows].max(), "most concentrated images get the fewest tokens"
    teacher = create_teacher(10, pretrained=False).eval()
    x = torch.randn(7, 3, 224, 224)
    T, S = torch.rand(7, 196), torch.rand(7, 196)
    buckets = [(rows, tam_select(T[rows], S[rows], kb, 0.3)) for rows, kb in tam_buckets(T, 59, 0.33)]
    with torch.no_grad():
        out = teacher_forward_buckets(teacher, x, buckets)
        for rows, idx in buckets:
            for j, r in enumerate(rows.tolist()):
                single = teacher_forward(teacher, x[r:r + 1], idx[j:j + 1])
                assert torch.allclose(out[r], single[0], atol=1e-5, rtol=0), "bucketed != standalone forward"


def test_tam_selector_uses_student_hook():
    torch.manual_seed(0)
    s = create_student(10, pretrained=False).train()
    x = torch.randn(6, 3, 224, 224)
    T = torch.rand(6, 196)
    with TamSelector("tam", 0.3, s, gap=(0.0, 0.0)) as sel:
        torch.manual_seed(1)
        out = s(x)
        idx = sel.select(6, T)
    with StudentMaskSelector("maskedkd", 0.3, s) as ref:
        torch.manual_seed(1)
        out2 = s(x)
        S = ref.recorder.cls_last.clone()
    assert torch.equal(out, out2), "selector hooks must not change the training forward"
    assert torch.equal(idx, tam_select(T, S, 59, 0.0))
    with TamSelector("tam_var", 0.3, s) as sel:
        s(x)
        b = sel.select(6, T)
    assert isinstance(b, list) and sum(r.numel() for r, _ in b) == 6
    assert all(len(blk.attn._forward_hooks) == 0 for blk in s.blocks)


def test_rollout_matches_stage1():
    """Training rollout selection == Stage 1 rollout criterion (same student, same input)."""
    from stage0.attention import AttentionRecorder
    from stage0.masking import topk_idx
    torch.manual_seed(0)
    s = create_student(10, pretrained=False).eval()
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        with AttentionRecorder(s, need=("cls_last", "rollout")) as rec:     # as stage0/fidelity.py does
            ref_out = s(x)
        for keep in (0.3, 0.15):
            with StudentMaskSelector("rollout", keep, s) as sel:
                out = s(x)
                idx = sel.select(5, "cpu")
            assert torch.equal(out, ref_out), "rollout hooks must not change the student forward"
            assert torch.equal(idx, topk_idx(rec.rollout, num_keep(keep))), keep
    assert all(len(blk.attn._forward_hooks) == 0 for blk in s.blocks)


def test_tam_oracle_map_matches_stage1():
    """tam_oracle T == Stage 1 teacher_oracle:attn_last on the same view."""
    from stage0.attribution import teacher_attribution, topk_overlap
    from stage0.make_attribution_cache import teacher_attributions
    torch.manual_seed(0)
    t = create_teacher(10, pretrained=False).eval()
    x = torch.randn(4, 3, 224, 224)
    _, maps = teacher_attributions(t, x, torch.device("cpu"))                  # Stage 1 / cache computation
    for chunk in (None, 3):
        T = teacher_attribution(t, x, "attn_last", chunk=chunk)
        assert torch.allclose(T, maps["attn_last"], atol=1e-6), (T - maps["attn_last"]).abs().max()
        assert topk_overlap(T, maps["attn_last"], 29) == 1.0
    assert torch.allclose(teacher_attribution(t, x, "rollout"), maps["rollout"], atol=1e-6)


def test_scheduler_default_criteria():
    from stage0 import scheduler
    src = Path(scheduler.__file__).read_text()
    assert 'default=list(STAGE2_CRITERIA)' in src
    from stage0.masking import STAGE2_CRITERIA
    assert STAGE2_CRITERIA == ("maskedkd", "random")


def test_pilot_verdict():
    import json
    import shutil
    from stage0 import summarize_stage2 as S
    root = Path(tempfile.mkdtemp())

    def scenario(acc):
        shutil.rmtree(root, ignore_errors=True)
        for d, vals in acc.items():
            for s_, v in enumerate(vals):
                p = root / "out" / "cub" / d / f"seed{s_}"
                p.mkdir(parents=True, exist_ok=True)
                (p / "eval.json").write_text(json.dumps({"clean_acc": v, "corruption_acc": 50.0, "pretrained": True}))
        S.main(["--keeps", "0.3", "0.15", "--output-root", str(root / "out"), "--results-dir", str(root / "res")])
        md = (root / "res" / "stage2_summary.md").read_text()
        return md[md.index("### 파일럿 판단"):]

    base = {"kd": [81.5, 81.5], "ce": [80, 80], "maskedkd_k0.3": [81.0, 81.0], "maskedkd_k0.15": [80.0, 80.0],
            "rollout_k0.3": [81.2, 81.2], "rollout_k0.15": [80.5, 80.5], "tam_k0.15": [80.1, 80.1],
            "tam_var_k0.3": [81.0, 81.1], "tam_var_k0.15": [80.0, 80.0], "tam_oracle_k0.15": [80.0, 80.0]}
    v = scenario({**base, "tam_k0.3": [81.4, 81.4]})
    assert "가능성 있음" in v and "rollout 대비 우위 없음" in v, v
    v = scenario({**base, "tam_k0.3": [81.4, 81.4], "rollout_k0.3": [80.9, 80.9]})
    assert "가능성 있음" in v and "rollout도 이김" in v, v
    v = scenario({**base, "tam_k0.3": [81.1, 81.1], "tam_oracle_k0.15": [80.5, 80.4]})
    assert "캐시가 병목" in v, v
    v = scenario({**base, "tam_k0.3": [81.1, 81.1]})
    assert "가능성 낮음" in v, v
    no_oracle = {k: val for k, val in base.items() if not k.startswith("tam_oracle")}
    v = scenario({**no_oracle, "tam_k0.3": [81.1, 81.1]})
    assert "미완료" in v, v
    v = scenario({**base, "tam_k0.3": [81.1]})            # one seed only
    assert "미완료" in v, v


def test_tam_var_flops_close_to_fixed():
    for keep in (0.5, 0.3, 0.15):
        k = num_keep(keep)
        lo, mid, hi = tam_bucket_ks(k, 0.33)
        mean = (teacher_gflops(lo) + teacher_gflops(mid) + teacher_gflops(hi)) / 3
        assert abs(mean / teacher_gflops(k) - 1) <= 0.03, (keep, mean, teacher_gflops(k))


def test_configs():
    from stage0 import train as T
    base = ["--dataset", "cub", "--seed", "0", "--lr", "1e-4"]
    tam_keys = {"tam_gap", "tam_kind", "tam_budget", "tam_delta"}
    for crit in ("maskedkd", "random"):
        cfg = T.build_config(T.parse_args(["--mode", "maskedkd", "--mask-criterion", crit, "--keep", "0.3"] + base),
                             Path("/o"), Path("/d"))
        assert not tam_keys & set(cfg), "Stage 2 configs must not change"
    cfg = T.build_config(T.parse_args(["--mode", "maskedkd", "--mask-criterion", "tam", "--keep", "0.3"] + base),
                         Path("/o"), Path("/d"))
    assert cfg["tam_gap"] == [0.1, 0.5] and cfg["tam_kind"] == "attn_last" and cfg["tam_budget"] == "fixed" \
        and cfg["tam_delta"] is None and cfg["tam_teacher_signal"] == "cache"
    cfg = T.build_config(T.parse_args(["--mode", "maskedkd", "--mask-criterion", "tam_var", "--keep", "0.15",
                                       "--tam-gap", "0.2", "0.2", "--tam-kind", "rollout"] + base),
                         Path("/o"), Path("/d"))
    assert cfg["tam_gap"] == [0.2, 0.2] and cfg["tam_kind"] == "rollout" and cfg["tam_budget"] == "bucket" \
        and cfg["tam_delta"] == 0.33 and cfg["keep_k"] == 29
    cfg = T.build_config(T.parse_args(["--mode", "maskedkd", "--mask-criterion", "tam_oracle", "--keep", "0.15"]
                                      + base), Path("/o"), Path("/d"))
    assert cfg["tam_teacher_signal"] == "oracle" and cfg["tam_budget"] == "fixed" and cfg["selection_gflops"] > 17
    cfg = T.build_config(T.parse_args(["--mode", "maskedkd", "--mask-criterion", "rollout", "--keep", "0.3"] + base),
                         Path("/o"), Path("/d"))
    assert not tam_keys & set(cfg) and 0.3 < cfg["selection_gflops"] < 0.6
    for crit in ("maskedkd", "random"):
        cfg = T.build_config(T.parse_args(["--mode", "maskedkd", "--mask-criterion", crit, "--keep", "0.3"] + base),
                             Path("/o"), Path("/d"))
        assert "selection_gflops" not in cfg and "tam_teacher_signal" not in cfg
    for bad in (["--mask-criterion", "maskedkd", "--tam-gap", "0.1", "0.5"],
                ["--mask-criterion", "rollout", "--tam-kind", "rollout"],
                ["--mask-criterion", "tam", "--tam-budget", "bucket"],
                ["--mask-criterion", "tam_oracle", "--tam-delta", "0.2"],
                ["--mask-criterion", "tam", "--tam-delta", "0.2"]):
        try:
            T.build_config(T.parse_args(["--mode", "maskedkd", "--keep", "0.3"] + bad + base), Path("/o"), Path("/d"))
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")


def test_indexed_loader_same_images(data_root):
    plain = Stage0Dataset(data_root, "cub", "train", build_train_transform(), limit=24)
    wrapped = WithIndex(Stage0Dataset(data_root, "cub", "train", build_train_transform(return_box=True), limit=24))
    for ds in (plain, wrapped):
        ds.set_aug_seed(11)
    bs = lambda: torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(  # noqa: E731
        plain, generator=torch.Generator().manual_seed(3)), 8, drop_last=True)
    a = list(make_loader(plain, None, 2, batch_sampler=bs()))
    b = list(make_loader(wrapped, None, 2, batch_sampler=bs(), collate_fn=numpy_collate_box_index))
    order = [i for batch in bs() for i in batch]
    for (xa, ya, _), (xb, yb, _, box, idx) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb) and box.shape == (8, 7)
    assert [i for *_, idx in b for i in idx.tolist()] == order, "index must be the sampled dataset index"


if __name__ == "__main__":
    test_existing_criteria_bit_identical()
    test_tam_select_properties()
    test_tam_buckets_and_forward()
    test_tam_selector_uses_student_hook()
    test_rollout_matches_stage1()
    test_tam_oracle_map_matches_stage1()
    test_scheduler_default_criteria()
    test_tam_var_flops_close_to_fixed()
    test_configs()
    test_pilot_verdict()
    dr = os.environ.get("DATA_ROOT")
    if dr and (Path(dr) / "cub" / "stage0_split.json").exists():
        test_indexed_loader_same_images(dr)
    else:
        print("skip indexed-loader test (DATA_ROOT not prepared)")
    print("stage 3 pilot unit tests passed")
