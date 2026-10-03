"""Build the teacher attribution cache once, before training (see attribution.py for the format).

    uv run python -m stage0.make_attribution_cache --dataset cub

Input: every train-split image, whole image resized to 224x224 (bicubic, no crop), ImageNet-normalized.
Teacher: $OUTPUT_ROOT/{dataset}/teacher/seed0/best.pt (read only).
Maps: attn_last (last block CLS->patch attention, head mean) and rollout (attention-rollout CLS row), 14x14.
"""
import argparse
import sys
import time

import numpy as np
import torch
from torchvision import transforms

from stage0 import common as C
from stage0.attention import AttentionRecorder
from stage0.attribution import GRID, KINDS, cache_dir, items_hash
from stage0.datasets import Stage0Dataset, normalize
from stage0.engine import amp_ctx, get_device, make_loader
from stage0.models import load_finetuned_teacher, teacher_forward

INPUT_DESC = "Resize((224,224), bicubic) of the whole image -> ToTensor -> Normalize(ImageNet)"


def whole_image_transform():
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        normalize()])


@torch.no_grad()
def teacher_attributions(teacher, x, device, amp=True):
    """(logits, {kind: (B, 196)}) of a full teacher forward on x."""
    with AttentionRecorder(teacher, need=("cls_last", "rollout")) as rec, amp_ctx(device, amp):
        logits = teacher_forward(teacher, x)
    return logits.float(), {"attn_last": rec.cls_last, "rollout": rec.rollout}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="[smoke] first-N sample of the train split")
    args = ap.parse_args(argv)

    out_root, data_root = C.output_root(args.output_root), C.data_root(args.data_root)
    tdir = C.teacher_dir(out_root, args.dataset)
    if not (tdir / "DONE").exists():
        sys.exit(f"[error] teacher run {tdir} is not finished (no DONE)")
    ds = Stage0Dataset(data_root, args.dataset, "train", whole_image_transform(), limit=args.limit)
    tcfg = C.load_json(tdir / "config.json")
    manifest = {"dataset": args.dataset, "split": "train", "n": len(ds), "limit": args.limit,
                "split_hash": items_hash(ds.items), "teacher_run_uid": tcfg["run_uid"],
                "teacher_ckpt": str(tdir / "best.pt"), "input": INPUT_DESC, "kinds": list(KINDS),
                "grid": GRID, "dtype": "float16"}
    d = cache_dir(out_root, args.dataset)
    if (d / "DONE").exists():
        old = C.load_json(d / "manifest.json")
        same = {k: v for k, v in old.items() if k != "created"} == manifest
        if same and not args.overwrite:
            print(f"{d} already built for this teacher/split; nothing to do.")
            return
        if not args.overwrite:
            sys.exit(f"[error] {d} holds a cache for a different teacher/split. Use --overwrite.")
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").unlink(missing_ok=True)

    C.set_cpu_threads()
    device = get_device()
    teacher, _ = load_finetuned_teacher(tdir / "best.pt", device)
    nw = args.num_workers if args.num_workers is not None else C.default_num_workers()
    out = {k: np.zeros((len(ds), GRID, GRID), dtype=np.float16) for k in KINDS}
    t0, r = time.time(), 0
    for b, (x, _, _) in enumerate(make_loader(ds, args.batch_size, nw, shuffle=False)):
        _, maps = teacher_attributions(teacher, x.to(device, non_blocking=True), device)
        n = x.shape[0]
        for k in KINDS:
            out[k][r:r + n] = maps[k].reshape(n, GRID, GRID).cpu().numpy().astype(np.float16)
        r += n
        if b % 20 == 0:
            print(f"  {r}/{len(ds)} images ({time.time() - t0:.0f}s)", flush=True)
    assert r == len(ds)
    for k in KINDS:
        assert np.isfinite(out[k]).all(), k
        np.save(d / f"{k}.npy", out[k])
    C.save_json(d / "items.json", [it[0] for it in ds.items])
    C.save_json(d / "manifest.json", {**manifest, "created": time.strftime("%F %T")})
    C.save_json(d / "DONE", {"n": len(ds), "seconds": round(time.time() - t0, 1)})
    print(f"wrote {d} ({len(ds)} images, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
