"""Download / extract CUB-200-2011 and Waterbirds into $DATA_ROOT and write fixed splits.

    uv run python -m stage0.prepare_data --datasets cub waterbirds

Layout produced:
    $DATA_ROOT/cub/CUB_200_2011/...                       (official tgz)
    $DATA_ROOT/cub/stage0_split.json                      train / val / test
    $DATA_ROOT/waterbirds/waterbird_complete95_forest2water2/...   (official tar.gz)
    $DATA_ROOT/waterbirds/stage0_split.json               train / val / test (+ group)

If the server has no internet, download the archives elsewhere and put them in
$DATA_ROOT/downloads/ (file names below); they are extracted from there.
Errors out if the split sizes differ from the official numbers.
"""
import argparse
import csv
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from stage0 import common as C
from stage0.datasets import EXPECTED_SPLITS, SPLIT_FILE, dataset_dir

SOURCES = {
    "cub": {
        "archive": "CUB_200_2011.tgz",
        "urls": [
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        ],
        "folder": "CUB_200_2011",
    },
    "waterbirds": {
        "archive": "waterbird_complete95_forest2water2.tar.gz",
        "urls": [
            "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
            "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        ],
        "folder": "waterbird_complete95_forest2water2",
    },
}
VAL_FRACTION = 0.10
VAL_SEED = 0


def download(urls, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"downloading {url} -> {dest}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
            return
        except Exception as e:  # noqa: BLE001 - try next mirror
            print(f"  failed: {e}")
    sys.exit(f"[error] could not download {dest.name}. Download it manually from one of\n  "
             + "\n  ".join(urls) + f"\nand place it at {dest}")


def ensure_extracted(data_root, name, skip_download):
    src = SOURCES[name]
    ddir = dataset_dir(data_root, name)
    if (ddir / src["folder"]).is_dir():
        return ddir / src["folder"]
    archive = Path(data_root) / "downloads" / src["archive"]
    if not archive.exists():
        if skip_download:
            sys.exit(f"[error] {ddir / src['folder']} not found and {archive} missing")
        download(src["urls"], archive)
    print(f"extracting {archive} -> {ddir}")
    ddir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(ddir, filter="data")
    if not (ddir / src["folder"]).is_dir():
        sys.exit(f"[error] {archive} did not contain {src['folder']}/")
    return ddir / src["folder"]


def _read_table(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def prepare_cub(data_root, skip_download):
    root = ensure_extracted(data_root, "cub", skip_download)
    images = {int(i): p for i, p in _read_table(root / "images.txt")}
    labels = {int(i): int(c) - 1 for i, c in _read_table(root / "image_class_labels.txt")}
    is_train = {int(i): int(t) == 1 for i, t in _read_table(root / "train_test_split.txt")}
    train_full = [i for i in sorted(images) if is_train[i]]
    test = [i for i in sorted(images) if not is_train[i]]

    sizes = {"train_full": len(train_full), "test": len(test)}
    check_sizes("cub", sizes, EXPECTED_SPLITS["cub"])

    by_class = defaultdict(list)
    for i in train_full:
        by_class[labels[i]].append(i)
    rng = random.Random(VAL_SEED)
    val = []
    for c in sorted(by_class):
        ids = sorted(by_class[c])
        n_val = max(1, round(VAL_FRACTION * len(ids)))
        val += rng.sample(ids, n_val)
    val_set = set(val)
    train = [i for i in train_full if i not in val_set]
    val = sorted(val)
    assert len(train) + len(val) == len(train_full)
    assert len(set(labels[i] for i in val)) == 200

    item = lambda i: (f"images/{images[i]}", labels[i], -1)  # noqa: E731
    meta = {
        "dataset": "cub",
        "image_root": "CUB_200_2011",
        "val_rule": f"per-class {VAL_FRACTION:.0%} of official train, random.Random({VAL_SEED})",
        "splits": {"train": [item(i) for i in train], "val": [item(i) for i in val],
                   "test": [item(i) for i in test]},
    }
    C.save_json(dataset_dir(data_root, "cub") / SPLIT_FILE, meta)
    print(f"cub: official train {len(train_full)} -> train {len(train)} / val {len(val)}; test {len(test)}")


def prepare_waterbirds(data_root, skip_download):
    root = ensure_extracted(data_root, "waterbirds", skip_download)
    splits = {0: "train", 1: "val", 2: "test"}
    out = {s: [] for s in splits.values()}
    with open(root / "metadata.csv") as f:
        for row in csv.DictReader(f):
            y, place = int(row["y"]), int(row["place"])
            out[splits[int(row["split"])]].append((row["img_filename"], y, 2 * y + place))
    sizes = {k: len(v) for k, v in out.items()}
    check_sizes("waterbirds", sizes, EXPECTED_SPLITS["waterbirds"])
    meta = {
        "dataset": "waterbirds",
        "image_root": SOURCES["waterbirds"]["folder"],
        "group_rule": "group = 2*y + place (y: 0 landbird / 1 waterbird, place: 0 land / 1 water)",
        "splits": out,
    }
    C.save_json(dataset_dir(data_root, "waterbirds") / SPLIT_FILE, meta)
    for s, items in out.items():
        g = Counter(x[2] for x in items)
        print(f"waterbirds {s}: {len(items)}  groups {[g[k] for k in range(4)]}")


def check_sizes(name, got, want):
    print(f"{name} split sizes: {got}")
    if got != want:
        sys.exit(f"[error] {name} split sizes {got} != expected {want}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=C.DATASETS)
    ap.add_argument("--data-root")
    ap.add_argument("--skip-download", action="store_true", help="only use files already in DATA_ROOT")
    args = ap.parse_args()
    root = C.data_root(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    for d in args.datasets:
        {"cub": prepare_cub, "waterbirds": prepare_waterbirds}[d](root, args.skip_download)


if __name__ == "__main__":
    main()
