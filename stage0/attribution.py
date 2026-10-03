"""Teacher attribution cache and crop-coordinate mapping (built in Stage 1, reused by Stage 3).

Cache ($OUTPUT_ROOT/{dataset}/attribution_cache/, written by make_attribution_cache.py):
    attn_last.npy, rollout.npy   float16 [N, 14, 14], row r = items.json[r] (train split order)
    items.json                   relative image paths, one per row
    manifest.json                teacher run_uid / ckpt, split hash, input transform, kinds, created
    DONE
Each map is the teacher's attribution for the WHOLE image resized to 224x224 (no crop): map cell (r, c)
covers rows [r H/14, (r+1) H/14) and columns [c W/14, (c+1) W/14) of the original image.

crop_maps() turns those maps into maps for a RandomResizedCrop view: the crop box (in original pixels) is
resampled to 14x14 with bilinear grid_sample, then mirrored if the view was flipped.
Known limits (measured in Stage 1 as teacher_cache vs teacher_oracle): RandAugment's geometric ops
(rotate / shear / translate) and random erasing are not reflected, mixup / cutmix are not reflected, and a
whole-image 14x14 map has coarse resolution when the crop is small.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage0 import common as C

KINDS = ("attn_last", "rollout")
GRID = 14


def cache_dir(out_root, dataset):
    return Path(out_root) / dataset / "attribution_cache"


def items_hash(items):
    return hashlib.sha1(json.dumps([list(x) for x in items]).encode()).hexdigest()


class AttributionCache:
    """Loads and validates the cache. lookup(rel_paths) -> (B, 14, 14) float32 tensor per kind."""

    def __init__(self, out_root, dataset, kinds=KINDS):
        d = cache_dir(out_root, dataset)
        if not (d / "DONE").exists():
            sys.exit(f"[error] attribution cache {d} missing or incomplete. "
                     f"Run: uv run python -m stage0.make_attribution_cache --dataset {dataset}")
        self.manifest = C.load_json(d / "manifest.json")
        tcfg = C.load_json(C.teacher_dir(out_root, dataset) / "config.json")
        if self.manifest["teacher_run_uid"] != tcfg["run_uid"]:
            sys.exit(f"[error] {d} was built from teacher run {self.manifest['teacher_run_uid']}, "
                     f"but the current teacher is {tcfg['run_uid']}. Rebuild it with --overwrite.")
        self.rows = {rel: r for r, rel in enumerate(C.load_json(d / "items.json"))}
        self.maps = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in kinds}

    def lookup(self, kind, rel_paths):
        try:
            rows = [self.rows[r] for r in rel_paths]
        except KeyError as e:
            raise KeyError(f"image {e} is not in the attribution cache (cache covers the train split)") from None
        return torch.from_numpy(np.asarray(self.maps[kind][rows], dtype=np.float32))


def crop_maps(maps, boxes):
    """maps (B, 14, 14) whole-image attribution; boxes (B, 7) = [i, j, h, w, flip, W, H] (original pixels).
    Returns (B, 14, 14): the attribution resampled onto the crop's 14x14 patch grid (bilinear), flipped
    left-right where the view was flipped."""
    maps = maps.float()
    boxes = boxes.to(maps.device, torch.float32)
    i, j, h, w, flip, W, H = boxes.unbind(1)
    # centres of the 14 output cells, in normalized original coords [0, 1]
    t = (torch.arange(GRID, device=maps.device, dtype=torch.float32) + 0.5) / GRID
    ys = (i[:, None] + t[None] * h[:, None]) / H[:, None]                     # (B, 14)
    xs = (j[:, None] + t[None] * w[:, None]) / W[:, None]
    gy = (2 * ys - 1)[:, :, None].expand(-1, GRID, GRID)                       # grid_sample: align_corners=False
    gx = (2 * xs - 1)[:, None, :].expand(-1, GRID, GRID)
    grid = torch.stack([gx, gy], dim=-1)                                       # (B, 14, 14, 2) as (x, y)
    out = F.grid_sample(maps[:, None], grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]
    return torch.where(flip[:, None, None] > 0.5, out.flip(-1), out)


def box_for_center_crop(W, H):
    """Box of Resize(256) -> CenterCrop(224) in original pixels (used to sanity-check crop_maps)."""
    s = C.RESIZE_SIZE / min(W, H)
    side = C.IMG_SIZE / s
    return [(H - side) / 2, (W - side) / 2, side, side, 0.0, W, H]
