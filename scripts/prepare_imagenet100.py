"""ImageNet-100 for the from-scratch check: download (HF ilee0022/ImageNet100, parquet, 17 GB) -> extract JPEGs ->
stage0_split.json -> teacher run dir (public ImageNet-1k DeiT-B, head restricted to the 100 classes; no fine-tuning).

    DATA_ROOT=... OUTPUT_ROOT=... HF_HOME=... uv run --with pyarrow python scripts/prepare_imagenet100.py
    (steps can be run separately: --steps download extract teacher)

Splits: HF train (117k, from the ImageNet train set) -> train; HF validation (13k, also ImageNet train) -> val;
HF test (5k = the original ImageNet val images of these classes) -> test.
The teacher is the public DeiT-B; it has seen the train/val images during its own ImageNet training (that is fine
for a teacher). The student is trained from scratch (train.py --random-init), so it has not.
"""
import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402
from stage0.datasets import EXPECTED_SPLITS, SPLIT_FILE  # noqa: E402

DS = "imagenet100"
REPO = "ilee0022/ImageNet100"
SPLITS = {"train": "train", "validation": "val", "test": "test"}  # HF split -> our split
# HF label (0..99, order of label2text.json) -> ImageNet-1k class index (timm ImageNetInfo).
# Matched by name; "crane" -> 134 (the bird; 517 is the machine), "rooster" -> 7 ("cock").
IMAGENET_INDEX = [133, 112, 14, 93, 92, 81, 6, 22, 123, 110, 135, 115, 36, 61, 66, 68, 29, 76, 80, 4, 48, 97, 84,
                  54, 118, 89, 46, 138, 106, 1, 94, 130, 18, 119, 134, 3, 39, 129, 52, 96, 99, 16, 70, 21, 77, 146,
                  2, 72, 55, 124, 74, 100, 26, 125, 5, 24, 59, 137, 90, 38, 8, 88, 113, 20, 32, 34, 141, 35, 116,
                  50, 11, 140, 71, 0, 73, 111, 143, 56, 41, 42, 117, 65, 83, 75, 144, 60, 57, 108, 104, 78, 91, 67,
                  150, 33, 7, 64, 28, 127, 19, 107]
assert len(IMAGENET_INDEX) == 100 and len(set(IMAGENET_INDEX)) == 100


def download(dl_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", local_dir=str(dl_dir), allow_patterns=["data/*", "label2text.json"])


def extract_file(args):
    pq_path, split, out_root = args
    import pyarrow.parquet as pq
    items = []
    stem = Path(pq_path).stem
    pf = pq.ParquetFile(pq_path)
    row = 0
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        imgs, labels = batch.column("image").to_pylist(), batch.column("label").to_pylist()
        for img, y in zip(imgs, labels):
            rel = f"{split}/{int(y):03d}/{stem}_{row:05d}.jpg"
            dst = Path(out_root) / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmp")
                tmp.write_bytes(img["bytes"])
                tmp.rename(dst)
            items.append([rel, int(y), 0])
            row += 1
    return split, items


def extract(dl_dir, data_root, workers):
    out_root = C.data_root(data_root) / DS / "images"
    files = []
    for hf_split, split in SPLITS.items():
        fs = sorted((Path(dl_dir) / "data").glob(f"{hf_split}-*.parquet"))
        if not fs:
            sys.exit(f"[error] no {hf_split}-*.parquet under {dl_dir}/data")
        files += [(str(f), split, str(out_root)) for f in fs]
    splits = {s: [] for s in SPLITS.values()}
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, (split, items) in enumerate(ex.map(extract_file, files), 1):
            splits[split] += items
            print(f"  {k}/{len(files)} parquet files ({time.time() - t0:.0f}s)", flush=True)
    for s in splits:
        splits[s].sort()
    sizes = {s: len(v) for s, v in splits.items()}
    exp = EXPECTED_SPLITS[DS]
    print(f"sizes {sizes} (expected {exp})")
    if not os.environ.get("IMAGENET100_ALLOW_SIZE_MISMATCH") and sizes != exp:
        sys.exit("[error] split sizes differ from the expected ones (set IMAGENET100_ALLOW_SIZE_MISMATCH=1 for tests)")
    labels = sorted({y for v in splits.values() for _, y, _ in v})
    meta = {"dataset": DS, "image_root": "images", "source": f"hf://datasets/{REPO}",
            "split_map": SPLITS, "num_classes_seen": len(labels), "splits": splits}
    C.save_json(out_root.parent / SPLIT_FILE, meta)
    print(f"wrote {out_root.parent / SPLIT_FILE}")


def make_teacher(out_root, data_root, pretrained=True, check_n=2000):
    import torch
    from stage0.datasets import Stage0Dataset, build_eval_transform
    from stage0.engine import accuracy, get_device, make_loader, predict
    from stage0.models import create_teacher
    import timm
    full = timm.create_model(C.TEACHER_ARCH, pretrained=pretrained)          # 1000-way ImageNet head
    t = create_teacher(100, pretrained=False)
    sd = {k: v for k, v in full.state_dict().items() if not k.startswith("head.")}
    idx = torch.tensor(IMAGENET_INDEX)
    sd["head.weight"], sd["head.bias"] = full.head.weight.data[idx].clone(), full.head.bias.data[idx].clone()
    t.load_state_dict(sd)
    dev = get_device()
    t.to(dev).eval()
    accs = {}
    for split in ("val", "test"):
        ds = Stage0Dataset(C.data_root(data_root), DS, split, build_eval_transform(), limit=check_n)
        logits, labels, _ = predict(t, make_loader(ds, 128, C.default_num_workers()), dev)
        accs[split] = accuracy(logits, labels)
    print(f"teacher (ImageNet-1k DeiT-B, 100-class head) accuracy: val {accs['val']:.2f} (seen by the teacher), "
          f"test {accs['test']:.2f}  [first {check_n} images each]")
    if pretrained and accs["test"] < 70:
        sys.exit("[error] teacher test accuracy < 70%: the label -> ImageNet index mapping is probably wrong")
    tdir = C.teacher_dir(C.output_root(out_root), DS)
    tdir.mkdir(parents=True, exist_ok=True)
    config = {"mode": "teacher", "dataset": DS, "seed": 0, "arch": C.TEACHER_ARCH, "num_classes": 100,
              "pretrained": pretrained, "epochs": 0, "lr": 0.0, "run_uid": uuid.uuid4().hex,
              "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "note": "public ImageNet-1k DeiT-B, head rows restricted to the 100 classes, no fine-tuning",
              "imagenet_index": IMAGENET_INDEX, "check_acc": accs}
    torch.save({"model": t.cpu().state_dict(), "config": config, "epoch": 0, "val_acc": accs["val"]}, tdir / "best.pt")
    C.save_json(tdir / "config.json", config)
    (tdir / "DONE").write_text("ok\n")
    print(f"wrote teacher run {tdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["download", "extract", "teacher"],
                    choices=["download", "extract", "teacher"])
    ap.add_argument("--download-dir", default=None, help="default $DATA_ROOT/downloads/imagenet100")
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    ap.add_argument("--workers", type=int, default=max(1, C.cpu_count() // 2))
    ap.add_argument("--no-pretrained", action="store_true", help="[test only] random teacher weights")
    ap.add_argument("--check-n", type=int, default=2000)
    a = ap.parse_args()
    dl = Path(a.download_dir) if a.download_dir else C.data_root(a.data_root) / "downloads" / DS
    if "download" in a.steps:
        download(dl)
    if "extract" in a.steps:
        extract(dl, a.data_root, a.workers)
    if "teacher" in a.steps:
        make_teacher(a.output_root, a.data_root, pretrained=not a.no_pretrained, check_n=a.check_n)


if __name__ == "__main__":
    main()
