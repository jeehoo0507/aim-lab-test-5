"""Bird-segmentation cache for the oracle criteria (oracle_bird / oracle_bg), same format as attribution.py.

    uv run python -m stage0.seg_cache --dataset waterbirds          # ~1-2 min, CPU

$OUTPUT_ROOT/{dataset}/seg_cache/: bird.npy float16 [N, 14, 14] (row r = items.json[r], train split order),
items.json, manifest.json, DONE. Cell (r, c) = fraction of bird pixels in rows [r H/14, (r+1) H/14) and columns
[c W/14, (c+1) W/14) of the ORIGINAL image (whole image, no crop), i.e. the attribution-cache convention, so
attribution.crop_maps() maps it onto any RandomResizedCrop view and masking.mix_attribution() onto mixup / cutmix.
Masks: CUB-200-2011 segmentations (Waterbirds pastes the CUB birds with them; image size == mask size).
Not reflected (as for the attribution cache): RandAugment geometric ops and random erasing.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from stage0 import common as C
from stage0.attribution import GRID, items_hash
from stage0.datasets import load_split

SIDE = 224


def cache_dir(out_root, dataset):
    return Path(out_root) / dataset / "seg_cache"


def mask_grid(mask_path, img_size):
    """(14, 14) bird fraction of the whole image, from a CUB mask png (resized to the image if sizes differ)."""
    with Image.open(mask_path) as m:
        m = m.convert("L")
        if m.size != img_size:
            m = m.resize(img_size, Image.BILINEAR)
        m = m.resize((SIDE, SIDE), Image.BILINEAR)          # whole image -> 224 x 224, like the attribution cache
    t = (TF.to_tensor(m)[0] > 0.5).float()
    return F.avg_pool2d(t[None, None], SIDE // GRID)[0, 0]


class SegCache:
    """lookup(rel_paths) -> (B, 14, 14) float32 bird fraction (whole image)."""

    def __init__(self, out_root, dataset):
        d = cache_dir(out_root, dataset)
        if not (d / "DONE").exists():
            sys.exit(f"[error] segmentation cache {d} missing. Run: uv run python -m stage0.seg_cache --dataset {dataset}")
        self.rows = {rel: r for r, rel in enumerate(C.load_json(d / "items.json"))}
        self.maps = np.load(d / "bird.npy", mmap_mode="r")

    def lookup(self, rel_paths):
        try:
            rows = [self.rows[r] for r in rel_paths]
        except KeyError as e:
            raise KeyError(f"image {e} is not in the segmentation cache (train split only)") from None
        return torch.from_numpy(np.asarray(self.maps[rows], dtype=np.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="waterbirds", choices=C.DATASETS)
    ap.add_argument("--seg-root", default=None, help="default $DATA_ROOT/cub_segmentations (downloaded if missing)")
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    ap.add_argument("--limit", type=int, default=None, help="[smoke] same first-N subset as training --limit")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out_root, data_root = C.output_root(a.output_root), C.data_root(a.data_root)
    from stage0.datasets import Stage0Dataset
    from stage0.why_collapse import ensure_masks
    seg_dir = ensure_masks(a.seg_root or Path(data_root) / "cub_segmentations")
    img_root, _ = load_split(data_root, a.dataset, "train")
    items = Stage0Dataset(data_root, a.dataset, "train", None, limit=a.limit).items   # same subset as training
    d = cache_dir(out_root, a.dataset)
    manifest = {"dataset": a.dataset, "split": "train", "n": len(items), "limit": a.limit,
                "split_hash": items_hash(items), "grid": GRID, "source": "CUB-200-2011 segmentations"}
    if (d / "DONE").exists() and not a.overwrite:
        old = C.load_json(d / "manifest.json")
        if {k: v for k, v in old.items() if k not in ("created", "missing")} == manifest:
            print(f"{d} already built; nothing to do.")
            return
        sys.exit(f"[error] {d} holds a different cache. Use --overwrite.")
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").unlink(missing_ok=True)
    t0, out, missing = time.time(), np.zeros((len(items), GRID, GRID), dtype=np.float16), 0
    for r, (rel, _, _) in enumerate(items):
        p = Path(seg_dir) / Path(rel).with_suffix(".png")
        if not p.exists():
            missing += 1                       # stays 0 = "no bird": oracle_bird then has nothing to prefer
            continue
        with Image.open(img_root / rel) as im:
            out[r] = mask_grid(p, im.size).numpy().astype(np.float16)
    if missing > 0.01 * len(items):
        sys.exit(f"[error] {missing}/{len(items)} masks missing under {seg_dir}")
    np.save(d / "bird.npy", out)
    C.save_json(d / "items.json", [it[0] for it in items])
    C.save_json(d / "manifest.json", {**manifest, "missing": missing, "created": time.strftime("%F %T")})
    C.save_json(d / "DONE", {"n": len(items), "seconds": round(time.time() - t0, 1)})
    print(f"wrote {d}: {len(items)} images, mean bird share {float(out.astype(np.float32).mean()):.3f}, "
          f"{missing} masks missing ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
