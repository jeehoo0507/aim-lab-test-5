"""Build the corruption cache once (CPU only; safe to run next to GPU jobs).

    uv run python -m stage0.make_corruptions --datasets cub waterbirds --workers 16

Per dataset: a fixed 1,000-image test subset (seed 0; Waterbirds stratified 250 per group) ->
Resize(256, bicubic) -> CenterCrop(224) -> each of the 15 imagecorruptions x severity 1..5 ->
lossless PNG.  Layout:  $CORRUPTION_ROOT/{dataset}/manifest.json, clean/0000.png,
{corruption}/{severity}/0000.png, DONE.  (CORRUPTION_ROOT defaults to $DATA_ROOT/corruptions.)
Interrupted runs continue where they stopped (existing PNGs are kept).
Each (image, corruption, severity) uses its own numpy seed, so the cache is reproducible.
"""
import argparse
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from stage0 import common as C
from stage0.datasets import eval_crop, load_split


def select_subset(data_root, dataset, n):
    root, items = load_split(data_root, dataset, "test")
    rng = random.Random(C.CORRUPTION_SEED)
    if dataset == "waterbirds":
        by_group = defaultdict(list)
        for it in items:
            by_group[it[2]].append(it)
        assert sorted(by_group) == [0, 1, 2, 3]
        per = n // 4
        picked = [it for g in range(4) for it in sorted(rng.sample(by_group[g], per))]
    else:
        picked = sorted(rng.sample(items, n))
    return root, picked


def image_seed(i, c_idx, sev):
    return (C.CORRUPTION_SEED * 1_000_003 + i * 1_009 + c_idx * 17 + sev) % (2**31 - 1)


_STATE = {}


def _init(out_dir, root, corruptions, severities):
    from stage0.corruption_compat import CORRUPTIONS, corrupt  # per worker (patches imagecorruptions)
    _STATE.update(out=Path(out_dir), root=Path(root), corr=corruptions, sev=severities, corrupt=corrupt,
                  crop=eval_crop(), index={c: k for k, c in enumerate(CORRUPTIONS)})


def _work(job):
    i, rel = job
    s = _STATE
    name = f"{i:04d}.png"
    clean_path = s["out"] / "clean" / name
    todo = [(c, v) for c in s["corr"] for v in s["sev"] if not (s["out"] / c / str(v) / name).exists()]
    if clean_path.exists() and not todo:
        return 0
    with Image.open(s["root"] / rel) as im:
        img = s["crop"](im.convert("RGB"))
    arr = np.asarray(img, dtype=np.uint8)
    if not clean_path.exists():
        _save(img, clean_path)
    for c, v in todo:
        np.random.seed(image_seed(i, s["index"][c], v))
        out = s["corrupt"](arr, corruption_name=c, severity=v)
        _save(Image.fromarray(np.uint8(out)), s["out"] / c / str(v) / name)
    return len(todo)


def _save(img, path):
    tmp = path.with_name(path.stem + ".tmp.png")
    img.save(tmp, format="PNG", compress_level=3)
    os.replace(tmp, path)


def build(dataset, data_root, out_root, n, corruptions, severities, workers, overwrite):
    out = out_root / dataset
    root, subset = select_subset(data_root, dataset, n)
    manifest = {"dataset": dataset, "n": len(subset), "seed": C.CORRUPTION_SEED,
                "stratified_by_group": dataset == "waterbirds", "corruptions": list(corruptions),
                "severities": list(severities), "preprocess": "Resize(256,bicubic)->CenterCrop(224)->corrupt->PNG",
                "subset": [list(x) for x in subset]}
    if out.exists():
        old = C.load_json(out / "manifest.json") if (out / "manifest.json").exists() else None
        if old != manifest:
            if not overwrite:
                sys.exit(f"[error] {out} holds a different cache (subset/corruptions differ). Use --overwrite.")
            shutil.rmtree(out)
    if (out / "DONE").exists():
        print(f"{dataset}: cache complete at {out}")
        return
    for c in corruptions:
        for v in severities:
            (out / c / str(v)).mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    C.save_json(out / "manifest.json", manifest)

    jobs = [(i, it[0]) for i, it in enumerate(subset)]
    t0 = time.time()
    print(f"{dataset}: {len(jobs)} images x {len(corruptions)} x {len(severities)} -> {out}  ({workers} workers)")
    with mp.get_context("spawn").Pool(workers, initializer=_init,
                                      initargs=(str(out), str(root), list(corruptions), list(severities))) as pool:
        for k, _ in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            if k % 50 == 0 or k == len(jobs):
                print(f"  {dataset}: {k}/{len(jobs)} images  ({time.time() - t0:.0f}s)", flush=True)
    C.save_json(out / "DONE", {"n": len(jobs), "seconds": round(time.time() - t0, 1)})
    print(f"{dataset}: done in {time.time() - t0:.0f}s")


def main(argv=None):
    from stage0.corruption_compat import CORRUPTIONS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=C.DATASETS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--data-root")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n", type=int, default=C.CORRUPTION_SUBSET, help="[smoke] subset size")
    ap.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS), choices=CORRUPTIONS,
                    help="[smoke] subset of corruptions")
    args = ap.parse_args(argv)
    data_root = C.data_root(args.data_root)
    out_root = C.corruption_root(data_root)
    if args.n % 4:
        sys.exit("[error] --n must be divisible by 4 (Waterbirds stratification)")
    for d in args.datasets:
        build(d, data_root, out_root, args.n, args.corruptions, C.SEVERITIES, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
