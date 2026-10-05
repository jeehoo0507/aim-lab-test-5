"""Stage 1: how faithful is the token-reduced teacher? (no training)

    uv run python -m stage0.fidelity --dataset cub [--student-ckpt PATH] [--out-name stage1_fidelity]

For every criterion x keep ratio x view, the teacher is run on only the selected patch tokens and compared
with the full teacher on the same images:
  kl     KL(p_full || p_masked), tau = 1, mean over images
  agree  argmax p_masked == argmax p_full
  acc    argmax p_masked == label
  teacher_gflops    one teacher forward with k tokens (flops.py, DeiT convention)
  criterion_gflops  extra compute to pick the tokens (beyond the student forward training already does)
  oracle_overlap    teacher_cache rows only: |top-k(cache) ∩ top-k(oracle)| / k for the same attribution kind
                    (error of the whole-image-cache -> crop approximation)

Views: train_view = train split, train augmentation before mixup (RRC+flip+RandAugment+erasing), seed 0, 1 pass;
       test_clean = test split, Resize(256) -> CenterCrop(224).
teacher_cache is train_view only (the cache covers the train split). random is repeated for 3 seeds.
The student for maskedkd / rollout runs in eval mode (deterministic); during Stage 2 training the student's
train-mode forward (drop path on) is used instead, as in MaskedKD.
keep = 1.0 is the reference point (all tokens -> identical to the full teacher: kl 0, agree 1).

Outputs: results/{out-name}.csv (long format), .md (per view: rows criterion, cols keep, cell "agree / kl"),
         .png (agree vs keep per criterion, one panel per view).
"""
import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from stage0 import common as C
from stage0.attention import AttentionRecorder
from stage0.attribution import AttributionCache, crop_maps
from stage0.datasets import Stage0Dataset, build_eval_transform, build_train_transform
from stage0.engine import amp_ctx, get_device, make_loader, numpy_collate, numpy_collate_box
from stage0.flops import criterion_gflops, teacher_gflops
from stage0.make_attribution_cache import teacher_attributions
from stage0.masking import num_keep, random_idx, topk_idx
from stage0.models import load_finetuned_teacher, load_student_weights, teacher_forward

KEEPS = (1.0, 0.7, 0.5, 0.3, 0.15)
RANDOM_SEEDS = (0, 1, 2)
VIEW_SEED = 0
# (row label, base criterion for FLOPs, source)
ROWS = [("maskedkd", "maskedkd"), ("rollout", "rollout"), ("random", "random"),
        ("teacher_cache:attn_last", "teacher_cache"), ("teacher_cache:rollout", "teacher_cache"),
        ("teacher_oracle:attn_last", "teacher_oracle"), ("teacher_oracle:rollout", "teacher_oracle")]
FIELDS = ["criterion", "keep", "k", "view", "seed", "n", "kl", "agree", "acc", "teacher_gflops",
          "criterion_gflops", "oracle_overlap"]


def run_view(view, ds, teacher, student, cache, keeps, seeds, device, bs, nw, amp):
    boxes = view == "train_view"
    loader = make_loader(ds, bs, nw, shuffle=False, collate_fn=numpy_collate_box if boxes else numpy_collate)
    stats = {}   # (criterion, keep, seed) -> [kl, agree, acc, overlap, n]
    full_correct, n_total, t0 = 0, 0, time.time()
    for b, batch in enumerate(loader):
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device)
        B = x.shape[0]
        rels = [it[0] for it in ds.items[n_total:n_total + B]]
        full, oracle = teacher_attributions(teacher, x, device, amp)
        logp_full = F.log_softmax(full, 1)
        pred_full = full.argmax(1)
        full_correct += (pred_full == y).sum().item()
        with AttentionRecorder(student, need=("cls_last", "rollout")) as rec, amp_ctx(device, amp), torch.no_grad():
            student(x)
        scores = {"maskedkd": rec.cls_last, "rollout": rec.rollout,
                  "teacher_oracle:attn_last": oracle["attn_last"], "teacher_oracle:rollout": oracle["rollout"]}
        if boxes and cache is not None:
            for kind in ("attn_last", "rollout"):
                scores[f"teacher_cache:{kind}"] = crop_maps(cache.lookup(kind, rels).to(device),
                                                            batch[3].to(device)).flatten(1)
        for keep in keeps:
            if keep == 1.0:
                continue
            k = num_keep(keep)
            jobs = [(name, 0, topk_idx(s, k)) for name, s in scores.items()]
            for sd in seeds:
                g = torch.Generator().manual_seed(C.epoch_seed(sd, b, 4))
                jobs.append(("random", sd, random_idx(B, k, g, device)))
            for name, sd, idx in jobs:
                with amp_ctx(device, amp):
                    logits = teacher_forward(teacher, x, idx).float()
                logp = F.log_softmax(logits, 1)
                pred = logits.argmax(1)
                st = stats.setdefault((name, keep, sd), [0.0, 0, 0, 0.0, 0])
                st[0] += (logp_full.exp() * (logp_full - logp)).sum(1).sum().item()
                st[1] += (pred == pred_full).sum().item()
                st[2] += (pred == y).sum().item()
                if name.startswith("teacher_cache:"):
                    o = topk_idx(scores["teacher_oracle:" + name.split(":")[1]], k)
                    hit = (idx.unsqueeze(2) == o.unsqueeze(1)).any(2).float().sum(1) / k
                    st[3] += hit.sum().item()
                st[4] += B
        n_total += B
        if b % 10 == 0:
            print(f"  {view}: {n_total}/{len(ds)} images ({time.time() - t0:.0f}s)", flush=True)
    return stats, full_correct / n_total, n_total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.ALL_DATASETS)
    ap.add_argument("--student-ckpt", default=None, help="default: $OUTPUT_ROOT/{dataset}/kd/seed0/last.pt")
    ap.add_argument("--keeps", type=float, nargs="+", default=list(KEEPS))
    ap.add_argument("--random-seeds", type=int, nargs="+", default=list(RANDOM_SEEDS))
    ap.add_argument("--views", nargs="+", default=["train_view", "test_clean"], choices=["train_view", "test_clean"])
    ap.add_argument("--out-name", default="stage1_fidelity", help="results/{out-name}.csv/.md/.png")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="[smoke] images per view")
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    args = ap.parse_args(argv)
    if 1.0 not in args.keeps:
        args.keeps = [1.0] + args.keeps

    out_root, data_root = C.output_root(args.output_root), C.data_root(args.data_root)
    tdir = C.teacher_dir(out_root, args.dataset)
    if not (tdir / "DONE").exists():
        sys.exit(f"[error] teacher run {tdir} is not finished (no DONE)")
    sck = Path(args.student_ckpt) if args.student_ckpt else C.run_dir(out_root, args.dataset, "kd", 0) / "last.pt"
    if not args.student_ckpt and not (sck.parent / "DONE").exists():
        sys.exit(f"[error] {sck.parent} is not finished (no DONE); pass --student-ckpt explicitly")
    if not sck.exists():
        sys.exit(f"[error] student checkpoint {sck} not found")
    C.set_cpu_threads()
    device = get_device()
    amp = not args.no_amp
    teacher, _ = load_finetuned_teacher(tdir / "best.pt", device)
    student = load_student_weights(sck, device)
    cache = AttributionCache(out_root, args.dataset) if "train_view" in args.views else None
    nw = args.num_workers if args.num_workers is not None else C.default_num_workers()
    print(f"teacher {tdir / 'best.pt'}\nstudent {sck}")

    tflops = {k: teacher_gflops(num_keep(k), C.NUM_CLASSES[args.dataset]) for k in args.keeps}
    cflops = {base: criterion_gflops(base, C.NUM_CLASSES[args.dataset]) for _, base in ROWS}
    rows = []
    for view in args.views:
        if view == "train_view":
            ds = Stage0Dataset(data_root, args.dataset, "train", build_train_transform(return_box=True),
                               limit=args.limit)
            ds.set_aug_seed(VIEW_SEED)
        else:
            ds = Stage0Dataset(data_root, args.dataset, "test", build_eval_transform(), limit=args.limit)
        stats, full_acc, n = run_view(view, ds, teacher, student, cache if view == "train_view" else None,
                                      args.keeps, args.random_seeds, device, args.batch_size, nw, amp)
        print(f"{view}: full-teacher acc {100 * full_acc:.2f}% on {n} images")
        for name, base in ROWS:
            if name.startswith("teacher_cache") and view != "train_view":
                continue
            for keep in args.keeps:
                for sd in (args.random_seeds if name == "random" else [0]):
                    common = {"criterion": name, "keep": keep, "k": num_keep(keep), "view": view,
                              "seed": sd if name == "random" else "", "n": n,
                              "teacher_gflops": round(tflops[keep], 4), "criterion_gflops": round(cflops[base], 4)}
                    if keep == 1.0:
                        rows.append({**common, "kl": 0.0, "agree": 1.0, "acc": full_acc,
                                     "oracle_overlap": 1.0 if name.startswith("teacher_cache") else ""})
                        continue
                    kl, ag, ac, ov, cnt = stats[(name, keep, sd)]
                    rows.append({**common, "kl": kl / cnt, "agree": ag / cnt, "acc": ac / cnt,
                                 "oracle_overlap": ov / cnt if name.startswith("teacher_cache") else ""})

    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    with open(res / f"{args.out_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    write_markdown(res / f"{args.out_name}.md", rows, args, sck, tflops, cflops)
    plot(res / f"{args.out_name}.png", rows, args.views, args.keeps)
    print(f"wrote {res / args.out_name}.csv/.md/.png")


def aggregate(rows, view, name, keep, key):
    vals = [r[key] for r in rows if r["view"] == view and r["criterion"] == name and r["keep"] == keep
            and r[key] != ""]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def write_markdown(path, rows, args, sck, tflops, cflops):
    keeps = sorted(args.keeps, reverse=True)
    md = [f"# Stage 1 fidelity ({args.dataset})", "",
          f"teacher: `{args.dataset}/teacher/seed0/best.pt`; student for maskedkd / rollout: `{sck}` (eval mode).",
          "Cell: **agree** (masked-teacher argmax = full-teacher argmax, %) / **KL**(p_full ‖ p_masked), τ=1. "
          "random: mean ± std over seeds " + ", ".join(map(str, args.random_seeds)) + ".", "",
          "Teacher GFLOPs per image (DeiT convention): "
          + ", ".join(f"keep {k:g} (k={num_keep(k)}) = {tflops[k]:.2f}" for k in keeps) + ".  ",
          "Selection cost (GFLOPs/img beyond the student forward): "
          + ", ".join(f"{b} {v:.3f}" for b, v in cflops.items()) + ".", ""]
    if any(not r["n"] or r["n"] < 1000 for r in rows):
        md += ["> ⚠️ small-N run (smoke / --limit) — not a real measurement.", ""]
    for view in args.views:
        n = next((r["n"] for r in rows if r["view"] == view), 0)
        md += [f"## {view} (N={n})", "", "| criterion | " + " | ".join(f"keep {k:g}" for k in keeps) + " |",
               "|---|" + "---|" * len(keeps)]
        names = [nm for nm, _ in ROWS if any(r["criterion"] == nm and r["view"] == view for r in rows)]
        for name in names:
            cells = []
            for k in keeps:
                a, a_sd = aggregate(rows, view, name, k, "agree")
                kl, kl_sd = aggregate(rows, view, name, k, "kl")
                cell = f"{100 * a:.1f}" + (f" ± {100 * a_sd:.1f}" if a_sd else "") + f" / {kl:.3f}" \
                    + (f" ± {kl_sd:.3f}" if kl_sd else "")
                cells.append(cell)
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        acc_full = aggregate(rows, view, "random", 1.0, "acc")[0]
        md += ["", f"Full-teacher accuracy on this view: {100 * acc_full:.2f}%.", ""]
        if view == "train_view":
            ov = [(name, k, aggregate(rows, view, name, k, "oracle_overlap")[0]) for name, _ in ROWS
                  if name.startswith("teacher_cache") for k in keeps if k < 1.0]
            if ov:
                md += ["Cache approximation (top-k overlap of teacher_cache with teacher_oracle, same kind): "
                       + "; ".join(f"{n.split(':')[1]} keep {k:g} {100 * v:.1f}%" for n, k, v in ov if v is not None)
                       + ".", ""]
    path.write_text("\n".join(md) + "\n")


# categorical slots 1-7 of the reference palette, fixed order; oracle rows dashed (secondary encoding)
STYLE = {"maskedkd": ("#2a78d6", "o", "-"), "random": ("#eb6834", "s", "-"), "rollout": ("#1baf7a", "^", "-"),
         "teacher_cache:attn_last": ("#eda100", "D", "-"), "teacher_cache:rollout": ("#e87ba4", "v", "-"),
         "teacher_oracle:attn_last": ("#008300", "D", "--"), "teacher_oracle:rollout": ("#4a3aa7", "v", "--")}


def plot(path, rows, views, keeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(views), figsize=(5.2 * len(views), 4.2), sharey=True, squeeze=False)
    ks = sorted(keeps)
    for ax, view in zip(axes[0], views):
        for name, _ in ROWS:
            pts = [(k, aggregate(rows, view, name, k, "agree")) for k in ks]
            pts = [(k, m, sd) for k, (m, sd) in pts if m is not None]
            if not pts:
                continue
            color, marker, ls = STYLE[name]
            xs, ys = [p[0] for p in pts], [100 * p[1] for p in pts]
            ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=6, label=name)
            if name == "random":
                lo = [100 * (m - (sd or 0)) for _, m, sd in pts]
                hi = [100 * (m + (sd or 0)) for _, m, sd in pts]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_title(view, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("kept patch tokens (fraction of 196)", color="#52514e")
        ax.set_xticks(ks)
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    axes[0][0].set_ylabel("agreement with full teacher (%)", color="#52514e")
    handles = {}
    for ax in axes[0]:                      # one legend for all panels (teacher_cache only exists in train_view)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [n for n, _ in ROWS if n in handles]
    fig.legend([handles[n] for n in order], order, fontsize=8, frameon=False, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
