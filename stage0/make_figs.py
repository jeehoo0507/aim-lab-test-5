"""Key result figures for the slides (Waterbirds), from files the earlier steps already wrote. No GPU needed except for
fig 6 (skipped with --no-patches or when no GPU).   -> $RESULTS_DIR/figs/*.png

  1_wga_vs_keep.png      WGA (and clean) vs keep: MaskedKD, rollout, random, oracle bird-only (+ mk_L* if run), Full KD line
  2_keep015_bars.png     keep 0.15: WGA per method, seeds as dots, Full KD line (causal ordering: bg-only ... bird-only)
  3_dose_response.png    bird share of kept patches -> masked-teacher P(true) on waterbird/land -> student WGA, per method
  4_layers.png           same student, CLS attention of each layer: bird share and teacher P(true) on g2 (why the last layer)
  5_bg_errors.png        masked-teacher error, conflict vs aligned groups, per bird-share bin (background-driven target)
  6_patches.png          kept patches on waterbird/land test images: MaskedKD (last layer) / best layer / rollout / oracle

    uv run python -m stage0.make_figs                 # reads $OUTPUT_ROOT runs, $RESULTS_DIR/why_more/*
"""
import argparse
import csv
import gzip
import os
import re
from pathlib import Path

import numpy as np

from stage0 import common as C
from stage0.summarize_wga import load_runs

COL = {"maskedkd": "#d1495b", "rollout": "#00798c", "random": "#8d8d8d", "oracle_bird": "#2e4057",
       "oracle_bg": "#c9a227", "tam_r_g0": "#66a182", "mk_filt": "#e0868f", "kd": "#333333", "layer": "#edae49"}
NAME = {"maskedkd": "MaskedKD", "rollout": "rollout", "random": "random", "oracle_bird": "oracle: bird only",
        "oracle_bg": "oracle: background only", "tam_r_g0": "TrustMask", "mk_filt": "MaskedKD + sink filter"}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.dpi": 200})
    return plt


def runs(root, ds, crit, keep, seeds):
    return load_runs(root, ds, C.mask_dirname(crit, keep), seeds)


def md_table(path, heading):
    """Rows (list of cell lists) of the first markdown table after the line that starts with `heading`."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines()
    try:
        i = next(j for j, l in enumerate(lines) if l.startswith(heading))
    except StopIteration:
        return []
    out, started = [], False
    for l in lines[i + 1:]:
        if l.startswith("|"):
            cells = [c.strip().strip("*") for c in l.strip("|").split("|")]
            if started and not set("".join(cells)) <= set("-: "):
                out.append(cells)
            started = True
        elif started:
            break
    return out


def num(x):
    try:
        return float(re.sub(r"[^\d.\-+]", "", x))
    except ValueError:
        return float("nan")


def layer_runs(root, ds):
    return sorted({p.parent.name.split("_k")[0] for p in (Path(root) / ds).glob("mk_L*_k*/seed*")})


# ------------------------------------------------------------------------------------------- figs ---
def fig1(plt, root, ds, seeds, out):
    kd = load_runs(root, ds, "kd", seeds)
    kd_w, kd_c = np.mean([e["wga"] for e in kd.values()]), np.mean([e["clean_acc"] for e in kd.values()])
    keeps = [0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    series = ["maskedkd", "rollout", "random", "oracle_bird"] + layer_runs(root, ds)
    for c in series:
        pts = [(k, runs(root, ds, c, k, seeds)) for k in keeps]
        pts = [(k, r) for k, r in pts if r]
        if not pts:
            continue
        col = COL.get(c, COL["layer"])
        lab = NAME.get(c, f"MaskedKD @ {c[3:]}")
        for j, key in enumerate(("clean_acc", "wga")):
            m = [np.mean([e[key] for e in r.values()]) for _, r in pts]
            s = [np.std([e[key] for e in r.values()]) for _, r in pts]
            ax[j].errorbar([k for k, _ in pts], m, yerr=s, marker="o", color=col, label=lab, capsize=3, lw=2)
    for j, (v, t) in enumerate(((kd_c, "Clean accuracy"), (kd_w, "Worst-group accuracy (WGA)"))):
        ax[j].axhline(v, ls="--", color=COL["kd"], lw=1.2, label=f"Full KD ({v:.1f})")
        ax[j].set(title=t, xlabel="keep ratio (teacher tokens kept)", ylabel="%")
        ax[j].invert_xaxis()
    ax[1].legend(fontsize=9, loc="lower left")
    fig.suptitle("Fewer teacher tokens: clean stays, WGA collapses unless the kept tokens are on the bird", y=1.02)
    fig.savefig(out / "1_wga_vs_keep.png")
    plt.close(fig)


def fig2(plt, root, ds, seeds, out):
    order = ["oracle_bg", "random", "maskedkd", "mk_filt", "rollout", "tam_r_g0", "oracle_bird"] + \
        [c for c in layer_runs(root, ds)]
    kd = load_runs(root, ds, "kd", seeds)
    kd_w = np.mean([e["wga"] for e in kd.values()])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs, labels = [], []
    for c in order:
        r = runs(root, ds, c, 0.15, seeds)
        if not r:
            continue
        v = [e["wga"] for e in r.values()]
        x = len(xs)
        ax.bar(x, np.mean(v), color=COL.get(c, COL["layer"]), alpha=0.85, width=0.65)
        ax.scatter([x] * len(v), v, color="k", s=14, zorder=3)
        ax.text(x, np.mean(v) + 1.5, f"{np.mean(v):.1f}", ha="center", fontsize=10)
        xs.append(x)
        labels.append(NAME.get(c, f"MaskedKD\n@ {c[3:]}").replace(": ", ":\n").replace(" + ", "\n+ "))
    ax.axhline(kd_w, ls="--", color=COL["kd"], lw=1.2)
    ax.text(-0.45, kd_w + 1.5, f"Full KD {kd_w:.1f}", ha="left", fontsize=10, color=COL["kd"])
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set(ylabel="WGA (%)", title="keep 0.15 (15% of teacher tokens): what is kept decides WGA", ylim=(0, 90))
    fig.savefig(out / "2_keep015_bars.png")
    plt.close(fig)


def fig3(plt, root, ds, seeds, res, out):
    p = res / "why_more" / "records.csv.gz"
    if not p.exists():
        print("[skip] fig 3: no why_more/records.csv.gz")
        return
    rows = [r for r in csv.DictReader(gzip.open(p, "rt")) if r["part"] == "C" and r["split"] == "test"]
    pts = []
    for (c, k) in sorted({(r["score"], float(r["keep"])) for r in rows}):
        R = [r for r in rows if r["score"] == c and float(r["keep"]) == k]
        wb = [float(r["bird_sel"]) for r in R if r["group"] in ("2", "3") and r["bird_sel"] != "nan"]
        t2 = [float(r["p_t_mask"]) for r in R if r["group"] == "2"]
        st = runs(root, ds, c, k, seeds)
        if wb and st:
            pts.append((c, k, np.mean(wb), 100 * np.mean(t2), np.mean([e["wga"] for e in st.values()])))
    if not pts:
        print("[skip] fig 3: no matching runs")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for c, k, b, t, w in pts:
        col = COL.get(c, COL["layer"])
        ax[0].scatter(b, t, s=70, color=col)
        ax[1].scatter(t, w, s=70, color=col)
        lab = f"{NAME.get(c, c)} {k:g}"
        ax[0].annotate(lab, (b, t), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax[1].annotate(lab, (t, w), textcoords="offset points", xytext=(5, 4), fontsize=8)
    b, t, w = (np.array([p[i] for p in pts]) for i in (2, 3, 4))
    r1, r2 = np.corrcoef(b, t)[0, 1], np.corrcoef(t, w)[0, 1]
    ax[0].set(xlabel="bird share of kept patches (waterbird images)",
              ylabel="masked teacher P(true), waterbird/land (%)", title=f"selection -> teacher target  (r = {r1:.2f})")
    ax[1].set(xlabel="masked teacher P(true), waterbird/land (%)", ylabel="student WGA (%)",
              title=f"teacher target -> student WGA  (r = {r2:.2f})")
    fig.suptitle("The chain: kept bird patches -> teacher target on the minority group -> student WGA", y=1.02)
    fig.savefig(out / "3_dose_response.png")
    plt.close(fig)


def fig4(plt, res, out):
    rows = md_table(res / "why_more" / "why_more_AB.md", "## A. 같은 student, 신호만 바꾸기 (KD student")
    L = [r for r in rows if re.fullmatch(r"L\d+", r[0])]
    if not L:
        print("[skip] fig 4: no layer table in why_more_AB.md")
        return
    x = [int(r[0][1:]) for r in L]
    bird, t2 = [num(r[1]) for r in L], [num(r[5]) for r in L]
    roll = next((r for r in rows if r[0] == "rollout"), None)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(x, bird, "o-", color=COL["oracle_bird"], lw=2, label="bird share of kept patches")
    ax.set(xlabel="student layer whose CLS attention selects the patches", ylabel="bird share", xticks=x)
    ax2 = ax.twinx()
    ax2.plot(x, t2, "s--", color=COL["maskedkd"], lw=2, label="masked teacher P(true), waterbird/land (%)")
    ax2.set_ylabel("teacher P(true) on g2 (%)")
    ax2.spines["right"].set_visible(True)
    if roll:
        ax.axhline(num(roll[1]), color=COL["rollout"], ls=":", lw=1.5, label=f"rollout ({num(roll[1]):.2f})")
    ax.annotate("MaskedKD\n(last layer)", (x[-1], bird[-1]), textcoords="offset points", xytext=(-60, -30),
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.set_title("Same Full-KD student, keep 0.15: attention moves off the bird in the last layers")
    fig.savefig(out / "4_layers.png")
    plt.close(fig)


def fig5(plt, res, out):
    rows = md_table(res / "why_more" / "why_more_AB.md", "## B. teacher 오류는 배경 때문인가")
    want = [("MaskedKD (자기 student)", "MaskedKD", COL["maskedkd"]), ("rollout 신호 (KD student)", "rollout", COL["rollout"])]
    fig, axes = plt.subplots(1, len(want), figsize=(12, 4.2), sharey=True)
    ok = False
    for ax, (src, title, col) in zip(np.atleast_1d(axes), want):
        R = [r for r in rows if r[0] == src]
        if not R:
            continue
        ok = True
        x = np.arange(len(R))
        ax.bar(x - 0.2, [num(r[3]) for r in R], 0.4, color=col, label="conflict groups (bird ≠ background)")
        ax.bar(x + 0.2, [num(r[5]) for r in R], 0.4, color="#bbbbbb", label="aligned groups")
        ax.set_xticks(x, [r[1] for r in R])
        ax.set(xlabel="bird share of kept patches", title=title)
    if not ok:
        plt.close(fig)
        print("[skip] fig 5: no table B")
        return
    np.atleast_1d(axes)[0].set_ylabel("masked teacher error (%)")
    np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.suptitle("Same bird share, far more errors when the background disagrees: the target follows the background", y=1.02)
    fig.savefig(out / "5_bg_errors.png")
    plt.close(fig)


def fig6(plt, root, ds, res, out, n=5, seed=0):
    import random

    import torch
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    from stage0.masking import num_keep
    from stage0.models import load_finetuned_teacher, load_student_weights
    from stage0.why_collapse import bird_maps, ensure_masks, load_images, p_true, pick_items, run_teacher
    from stage0.why_more import run_selection, student_signals
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = C.data_root(None)
    img_root, items = pick_items(data_root, ds, "test", 0)
    g2 = [it for it in items if it[2] == 2]
    items = sorted(random.Random(seed).sample(g2, min(n, len(g2))))
    X = load_images(items, img_root)
    y = torch.tensor([it[1] for it in items])
    bm = bird_maps(items, img_root, ensure_masks(Path(data_root) / "cub_segmentations"))
    B = torch.stack([m if m is not None else torch.full((196,), float("nan")) for m in bm])
    teacher, _ = load_finetuned_teacher(C.teacher_dir(root, ds) / "best.pt", dev)
    sig = student_signals(load_student_weights(Path(root) / ds / "kd" / f"seed{seed}" / "last.pt", dev), X, dev, 16)
    k = num_keep(0.15)
    sel_file = res / "why_more" / "layer_select.json"
    best = C.load_json(sel_file)["layer"] if sel_file.exists() else None
    cols = [("maskedkd", "MaskedKD (last layer)")] + ([(f"mk_L{best}", f"layer {best} (val-picked)")] if best is not None else []) \
        + [("rollout", "rollout"), ("oracle_bird", "oracle: bird only")]
    sels = {c: run_selection(c, k, sig, B, None, seed) for c, _ in cols}
    pts = {c: p_true(run_teacher(teacher, X, s, dev, 16), y) for c, s in sels.items()}
    mean, std = torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None], torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None]
    fig, axes = plt.subplots(len(items), 1 + len(cols), figsize=(2.4 * (1 + len(cols)), 2.5 * len(items)), squeeze=False)
    for r in range(len(items)):
        img = (X[r].float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        bird = B[r].reshape(14, 14).numpy()
        for c in range(1 + len(cols)):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                ax.imshow(img)
                ax.set_title("waterbird on land" if r == 0 else "", fontsize=9)
            else:
                key, lab = cols[c - 1]
                keep = np.zeros(196)
                keep[sels[key][r].numpy()] = 1
                ax.imshow(img * (0.2 + 0.8 * np.kron(keep.reshape(14, 14), np.ones((16, 16)))[..., None]))
                ax.set_title((lab + "\n" if r == 0 else "") + f"teacher P(waterbird) {100 * float(pts[key][r]):.0f}%", fontsize=8)
            if not np.isnan(bird).all():
                ax.contour(np.kron(bird, np.ones((16, 16))), levels=[0.5], colors="lime", linewidths=0.7)
    fig.suptitle("keep 0.15, same Full-KD student: bright = patches the teacher sees, green = bird", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "6_patches.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-patches", action="store_true")
    ap.add_argument("--n-patches", type=int, default=5)
    ap.add_argument("--output-root")
    a = ap.parse_args()
    root, ds = C.output_root(a.output_root), a.dataset
    res = Path(os.environ.get("RESULTS_DIR", "results"))
    out = res / "figs"
    out.mkdir(parents=True, exist_ok=True)
    plt = setup()
    for name, fn in (("1", lambda: fig1(plt, root, ds, a.seeds, out)), ("2", lambda: fig2(plt, root, ds, a.seeds, out)),
                     ("3", lambda: fig3(plt, root, ds, a.seeds, res, out)), ("4", lambda: fig4(plt, res, out)),
                     ("5", lambda: fig5(plt, res, out))):
        try:
            fn()
        except Exception as e:      # one broken figure must not stop the others
            print(f"[warn] fig {name} failed: {e!r}")
    if not a.no_patches:
        try:
            fig6(plt, root, ds, res, out, n=a.n_patches)
        except Exception as e:
            print(f"[warn] fig 6 failed: {e!r}")
    print("wrote", sorted(p.name for p in out.glob("*.png")), "->", out)


if __name__ == "__main__":
    main()
