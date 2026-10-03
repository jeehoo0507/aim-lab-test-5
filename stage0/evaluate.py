"""Evaluate one run (4-3): clean test, Waterbirds groups / WGA, corruption accuracy.

    uv run python -m stage0.evaluate --dataset cub --mode kd --seed 1

Checkpoint: students -> last.pt of a finished run (final epoch, never val-best); teacher -> best.pt.
Corruption accuracy = mean over the cached conditions (15 corruptions x 5 severities = 75).
Writes eval.json into the run directory.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from stage0 import common as C
from stage0.datasets import Stage0Dataset, build_eval_transform, normalize
from stage0.engine import accuracy, get_device, make_loader, predict
from stage0.models import create_model


class CorruptionCache(torch.utils.data.Dataset):
    def __init__(self, cache_dir, manifest):
        self.dir = Path(cache_dir)
        self.tf = normalize()
        self.conds = [(c, s) for c in manifest["corruptions"] for s in manifest["severities"]]
        self.labels = [it[1] for it in manifest["subset"]]
        self.groups = [it[2] for it in manifest["subset"]]
        self.n = len(self.labels)

    def __len__(self):
        return len(self.conds) * self.n

    def __getitem__(self, k):
        cond, i = divmod(k, self.n)
        c, s = self.conds[cond]
        with Image.open(self.dir / c / str(s) / f"{i:04d}.png") as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i], cond


def group_metrics(logits, labels, groups):
    pred = logits.argmax(1)
    accs = []
    for g in range(4):
        m = groups == g
        accs.append(((pred[m] == labels[m]).float().mean().item() * 100.0) if m.any() else float("nan"))
    return accs, min(accs)


def load_run_model(rd, mode, device):
    cfg = C.load_json(rd / "config.json")
    if not (rd / "DONE").exists():
        sys.exit(f"[error] {rd} is not finished (no DONE)")
    ck_name = "best.pt" if cfg["mode"] == "teacher" else "last.pt"
    ck = torch.load(rd / ck_name, map_location="cpu", weights_only=False)
    if cfg["mode"] != "teacher" and ck["epoch"] != cfg["epochs"] - 1:
        sys.exit(f"[error] {rd}/last.pt is epoch {ck['epoch'] + 1}, expected the final epoch {cfg['epochs']}")
    model = create_model(cfg["arch"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck_name, ck["epoch"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--mode", required=True, help="teacher | ce | kd | kd_alpha1 (run directory name)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="[smoke] clean test images")
    ap.add_argument("--allow-partial-cache", action="store_true",
                    help="[smoke] accept a corruption cache with fewer than 15x5 conditions")
    args = ap.parse_args(argv)

    out_root, data_root = C.output_root(args.output_root), C.data_root(args.data_root)
    rd = Path(out_root) / args.dataset / args.mode / f"seed{args.seed}"
    C.set_cpu_threads()
    device = get_device()
    nw = args.num_workers if args.num_workers is not None else C.default_num_workers()
    model, cfg, ck_name, epoch = load_run_model(rd, args.mode, device)
    if cfg["dataset"] != args.dataset:
        sys.exit(f"[error] {rd} was trained on {cfg['dataset']}")

    res = {"dataset": args.dataset, "mode": args.mode, "seed": args.seed, "checkpoint": ck_name,
           "epoch": epoch + 1, "pretrained": cfg["pretrained"], "lr": cfg["lr"]}

    test = Stage0Dataset(data_root, args.dataset, "test", build_eval_transform(), limit=args.limit)
    logits, labels, groups = predict(model, make_loader(test, args.batch_size, nw), device)
    res["clean_acc"] = accuracy(logits, labels)
    res["n_test"] = len(test)
    if args.dataset == "waterbirds":
        res["group_acc"], res["wga"] = group_metrics(logits, labels, groups)
    print(f"{rd}: clean {res['clean_acc']:.2f}" + (f"  WGA {res['wga']:.2f}  groups "
          f"{[round(a, 2) for a in res['group_acc']]}" if "wga" in res else ""))

    cache = C.corruption_root(data_root) / args.dataset
    if not (cache / "DONE").exists():
        sys.exit(f"[error] corruption cache {cache} incomplete. Run: uv run python -m stage0.make_corruptions")
    manifest = C.load_json(cache / "manifest.json")
    n_cond = len(manifest["corruptions"]) * len(manifest["severities"])
    if n_cond != 75 and not args.allow_partial_cache:
        sys.exit(f"[error] cache has {n_cond} conditions, expected 75")
    ds = CorruptionCache(cache, manifest)
    clogits, clabels, conds = predict(model, make_loader(ds, args.batch_size, nw), device)
    correct = (clogits.argmax(1) == clabels).float()
    per = {}
    cgroups = torch.as_tensor(ds.groups).repeat(len(ds.conds))
    wgas = []
    for k, (c, s) in enumerate(ds.conds):
        m = conds == k
        per.setdefault(c, {})[str(s)] = correct[m].mean().item() * 100.0
        if args.dataset == "waterbirds":
            wgas.append(group_metrics(clogits[m], clabels[m], cgroups[m])[1])
    res["corruption_acc"] = sum(v for d in per.values() for v in d.values()) / n_cond
    res["corruption_per_condition"] = per
    res["corruption_n_conditions"] = n_cond
    res["corruption_n_images"] = ds.n
    if wgas:
        res["corruption_wga"] = sum(wgas) / len(wgas)
    print(f"  corruption acc ({n_cond} conditions x {ds.n} images): {res['corruption_acc']:.2f}")
    C.save_json(rd / "eval.json", res)
    print(f"wrote {rd / 'eval.json'}")


if __name__ == "__main__":
    main()
