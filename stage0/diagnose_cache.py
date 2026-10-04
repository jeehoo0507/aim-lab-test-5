"""Stage 3 pilot: how accurate is the crop-mapped attribution cache? (no training, a few minutes)

    uv run python -m stage0.diagnose_cache --dataset cub   -> results/stage3_cache_diag.{csv,md}

For every train image, V = 4 RandomResizedCrop views (scale 0.08-1, ratio 3/4-4/3, bicubic, random flip) are drawn
from a private RNG seeded per (image, view) -- the global RNG is never touched. No RandAugment / erasing, so the
crop mapping is isolated. Per view the teacher attribution (attn_last by default) on the view itself ("oracle") is
compared with:
  cache_vs_view   cached whole-image map (float16) -> attribution.crop_maps  (what tam / tam_var use)
  whole_vs_view   freshly computed whole-image map  -> crop_maps             (limit of the crop mapping itself:
                  even a perfect whole-image map cannot predict where the teacher looks once it sees only the crop)
  view_vs_view    the view's oracle computed twice (sanity check: must be 100% overlap, Spearman 1)
Metrics: top-k overlap (|top-k ∩ top-k| / k) at keep 0.3 / 0.15 and the Spearman rank correlation over the 196
patches, by crop-area ratio (box area / image area) bin and by flip.

Conclusion rule (written into the md): on cache_vs_view at keep 0.3, if the largest-area bin [0.7, 1.0] reaches
>= 70% overlap and exceeds the smallest bin [0.08, 0.2) by >= 15 points, the loss is concentrated in small crops
(resolution: a multi-scale cache could help); if even the largest bin is below 70%, overlap is low everywhere
(context dependence: a limit of any whole-image cache); otherwise mixed.
"""
import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from stage0 import common as C
from stage0.attribution import AttributionCache, crop_maps, teacher_attribution
from stage0.datasets import load_split, normalize
from stage0.engine import amp_ctx, get_device, make_loader
from stage0.masking import num_keep
from stage0.models import load_finetuned_teacher

BINS = ((0.08, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.0001))
BIN_NAMES = ("[0.08,0.2)", "[0.2,0.4)", "[0.4,0.7)", "[0.7,1.0]")
COMPARISONS = ("cache_vs_view", "whole_vs_view", "view_vs_view")


def rrc_params(rng, W, H, scale=C.RRC_SCALE, ratio=C.RRC_RATIO):
    """torchvision / timm RandomResizedCrop.get_params, drawing from `rng` instead of the global RNG."""
    area = W * H
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = rng.uniform(*scale) * area
        aspect = math.exp(rng.uniform(*log_ratio))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= W and 0 < h <= H:
            return rng.randint(0, H - h), rng.randint(0, W - w), h, w
    in_ratio = W / H
    if in_ratio < min(ratio):
        w, h = W, int(round(W / min(ratio)))
    elif in_ratio > max(ratio):
        h, w = H, int(round(H * max(ratio)))
    else:
        w, h = W, H
    return (H - h) // 2, (W - w) // 2, h, w


class ViewDataset(torch.utils.data.Dataset):
    """Item i -> (whole image 224x224, V views (V,3,224,224), boxes (V,7), i)."""

    def __init__(self, data_root, dataset, n_views, seed, limit=None):
        self.root, items = load_split(data_root, dataset, "train")
        if limit is not None and limit < len(items):
            items = sorted(random.Random(0).sample(items, limit))   # same subset as Stage0Dataset(limit=...)
        self.items, self.V, self.seed = items, n_views, seed
        self.norm = normalize()
        self.whole = transforms.Resize((C.IMG_SIZE, C.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        with Image.open(self.root / self.items[i][0]) as im:
            img = im.convert("RGB")
        W, H = img.size
        views, boxes = [], []
        for v in range(self.V):
            rng = random.Random((self.seed * 1_000_003 + i * 101 + v) % (2**31 - 1))
            t, l, h, w = rrc_params(rng, W, H)
            flip = rng.random() < 0.5
            crop = transforms.functional.resized_crop(img, t, l, h, w, [C.IMG_SIZE, C.IMG_SIZE],
                                                      transforms.InterpolationMode.BICUBIC)
            if flip:
                crop = transforms.functional.hflip(crop)
            views.append(self.norm(crop).numpy())
            boxes.append([t, l, h, w, float(flip), W, H])
        return (self.norm(self.whole(img)).numpy(), np.stack(views), np.asarray(boxes, np.float32), i)


def collate(batch):
    whole, views, boxes, idx = zip(*batch)
    return np.stack(whole), np.stack(views), np.stack(boxes), np.asarray(idx, dtype=np.int64)


def spearman(a, b):
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-12)


def overlap(a, b, k):
    ia, ib = torch.topk(a.float(), k, 1).indices, torch.topk(b.float(), k, 1).indices
    return (ia.unsqueeze(2) == ib.unsqueeze(1)).any(2).float().sum(1) / k


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--kind", default="attn_last", choices=("attn_last", "rollout"))
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.3, 0.15])
    ap.add_argument("--batch-size", type=int, default=32, help="images per batch (x views teacher forwards)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="[smoke] images")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    args = ap.parse_args(argv)

    out_root, data_root = C.output_root(args.output_root), C.data_root(args.data_root)
    tdir = C.teacher_dir(out_root, args.dataset)
    if not (tdir / "DONE").exists():
        sys.exit(f"[error] teacher run {tdir} is not finished (no DONE)")
    C.set_cpu_threads()
    device = get_device()
    amp = (lambda: amp_ctx(device, not args.no_amp))
    teacher, _ = load_finetuned_teacher(tdir / "best.pt", device)
    cache = AttributionCache(out_root, args.dataset, kinds=(args.kind,))
    ds = ViewDataset(data_root, args.dataset, args.views, args.seed, args.limit)
    nw = args.num_workers if args.num_workers is not None else C.default_num_workers()
    loader = make_loader(ds, args.batch_size, nw, shuffle=False, collate_fn=collate)
    ks = {keep: num_keep(keep) for keep in args.keeps}

    acc = {}   # (comparison, group, keep) -> [overlap_sum, spearman_sum, n]
    t0, done = time.time(), 0
    for b, (whole, views, boxes, idx) in enumerate(loader):
        B, V = views.shape[:2]
        whole, views = whole.to(device), views.flatten(0, 1).to(device)
        boxes = boxes.flatten(0, 1).to(device)
        rels = [ds.items[i][0] for i in idx.tolist()]
        oracle = teacher_attribution(teacher, views, args.kind, amp, args.batch_size)          # (B*V, 196)
        whole_map = teacher_attribution(teacher, whole, args.kind, amp).reshape(B, 14, 14)
        cached = cache.lookup(args.kind, rels).to(device)
        rep = lambda m: m.repeat_interleave(V, 0)  # noqa: E731
        cand = {"cache_vs_view": crop_maps(rep(cached), boxes).flatten(1),
                "whole_vs_view": crop_maps(rep(whole_map), boxes).flatten(1)}
        if b == 0:   # sanity: the same view's oracle twice
            cand["view_vs_view"] = teacher_attribution(teacher, views, args.kind, amp, args.batch_size)
        area = (boxes[:, 2] * boxes[:, 3] / (boxes[:, 5] * boxes[:, 6])).cpu()
        flip = boxes[:, 4].cpu() > 0.5
        bin_id = torch.tensor([next(j for j, (lo, hi) in enumerate(BINS) if lo <= a < hi) for a in area.tolist()])
        for comp, m in cand.items():
            sp = spearman(m, oracle).cpu()
            for keep, k in ks.items():
                ov = overlap(m, oracle, k).cpu()
                groups = {"all": torch.ones_like(flip)}
                groups.update({f"area {BIN_NAMES[j]}": bin_id == j for j in range(len(BINS))})
                groups.update({"flip": flip, "no flip": ~flip})
                for g, mask in groups.items():
                    if mask.any():
                        st = acc.setdefault((comp, g, keep), [0.0, 0.0, 0])
                        st[0] += ov[mask].sum().item()
                        st[1] += sp[mask].sum().item()
                        st[2] += int(mask.sum())
        done += B
        if b % 20 == 0:
            print(f"  {done}/{len(ds)} images ({time.time() - t0:.0f}s)", flush=True)

    rows = [{"comparison": c, "group": g, "keep": keep, "k": ks[keep], "n": n, "overlap": o / n, "spearman": s / n}
            for (c, g, keep), (o, s, n) in acc.items()]
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "stage3_cache_diag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    write_md(res / "stage3_cache_diag.md", rows, args, len(ds))
    print(f"wrote {res / 'stage3_cache_diag.csv'} / .md ({time.time() - t0:.0f}s)")


def get(rows, comp, group, keep, key):
    r = [x for x in rows if x["comparison"] == comp and x["group"] == group and x["keep"] == keep]
    return r[0][key] if r else None


def write_md(path, rows, args, n_images):
    keeps = sorted(args.keeps, reverse=True)
    groups = ["all"] + [f"area {b}" for b in BIN_NAMES] + ["flip", "no flip"]
    md = [f"# Stage 3 pilot: attribution cache diagnosis ({args.dataset}, {args.kind})", "",
          f"{n_images} train images × {args.views} RRC views (seed {args.seed}); oracle = teacher {args.kind} on the view.",
          "Cell: top-k overlap % / Spearman. cache_vs_view = what tam uses; whole_vs_view = fresh whole-image map "
          "through the same crop mapping (crop-mapping limit; the gap to cache_vs_view is the cache's storage error).", ""]
    if n_images < 1000:
        md += ["> ⚠️ small-N run (smoke / --limit) — not a real measurement.", ""]
    s = get(rows, "view_vs_view", "all", keeps[0], "overlap")
    md += [f"Sanity (same view's oracle twice, first batch): overlap {100 * s:.1f}%.", ""]
    for comp in ("cache_vs_view", "whole_vs_view"):
        md += [f"## {comp}", "", "| group | n | " + " | ".join(f"keep {k:g}" for k in keeps) + " |",
               "|---|---|" + "---|" * len(keeps)]
        for g in groups:
            n = get(rows, comp, g, keeps[0], "n")
            if n is None:
                continue
            cells = [f"{100 * get(rows, comp, g, k, 'overlap'):.1f} / {get(rows, comp, g, k, 'spearman'):.3f}"
                     for k in keeps]
            md.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
        md.append("")
    k0 = 0.3 if 0.3 in args.keeps else keeps[0]
    lo = get(rows, "cache_vs_view", f"area {BIN_NAMES[0]}", k0, "overlap")
    hi = get(rows, "cache_vs_view", f"area {BIN_NAMES[-1]}", k0, "overlap")
    md += ["## 결론", ""]
    if lo is None or hi is None:
        md.append("- 판정 불가: 가장 작은/큰 면적 구간에 view가 없음 (N이 너무 작음).")
    elif hi >= 0.7 and hi - lo >= 0.15:
        md.append(f"- **작은 크롭에서만 낮음 → 해상도 문제** (keep {k0:g}: 면적 [0.7,1.0] {100 * hi:.1f}% vs "
                  f"[0.08,0.2) {100 * lo:.1f}%). 다중 스케일 캐시로 개선 여지.")
    elif hi < 0.7:
        md.append(f"- **모든 구간에서 낮음 → 문맥 의존 (캐시 방식의 한계)** (keep {k0:g}: 가장 큰 크롭도 {100 * hi:.1f}%, "
                  f"가장 작은 크롭 {100 * lo:.1f}%).")
    else:
        md.append(f"- **혼합** (keep {k0:g}: 큰 크롭 {100 * hi:.1f}%, 작은 크롭 {100 * lo:.1f}%; 차이 15%p 미만).")
    md.append("- 판정 기준(사전 고정): cache_vs_view, keep 0.3. 가장 큰 구간 ≥ 70% 이고 (큰 − 작은) ≥ 15%p → 해상도; "
              "가장 큰 구간 < 70% → 문맥 의존; 그 외 혼합.")
    gap = [(g, get(rows, "whole_vs_view", g, k0, "overlap") - get(rows, "cache_vs_view", g, k0, "overlap"))
           for g in ("all",) if get(rows, "whole_vs_view", g, k0, "overlap") is not None]
    if gap:
        md.append(f"- 캐시 저장 오차 (whole_vs_view − cache_vs_view, 전체, keep {k0:g}): {100 * gap[0][1]:+.1f}%p.")
    path.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
