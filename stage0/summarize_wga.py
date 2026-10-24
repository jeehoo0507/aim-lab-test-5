"""WGA collapse summary (Waterbirds): where does MaskedKD's worst-group accuracy break as keep shrinks, and does
rollout hold at the same keep? Also checks the rebuilt Stage 0 against the archived cv2 numbers.

    uv run python -m stage0.summarize_wga                     # -> $RESULTS_DIR/wga_collapse.{md,csv,png}

Pre-registered rules (fixed before the rerun):
  collapse point  = largest keep where MaskedKD's paired WGA difference vs Full KD is <= -3.0 %p on average and
                    negative for every seed
  "clean is flat" = |mean paired clean difference vs KD| < 1.0 %p at the collapse point
  reproduction    = |new mean - archived mean| <= 1.0 %p (clean) and <= 3.0 %p (WGA) for teacher / CE / KD
Groups: g = 2*y + place  (0 landbird-land, 1 landbird-water, 2 waterbird-land, 3 waterbird-water).
"""
import argparse
import csv
import os
import statistics
from pathlib import Path

from stage0 import common as C

GROUPS = ("landbird/land", "landbird/water", "waterbird/land", "waterbird/water")
METRICS = (("clean_acc", "clean"), ("wga", "WGA"), ("corruption_acc", "corruption"))
COLLAPSE_DROP, CLEAN_FLAT, REPRO_CLEAN, REPRO_WGA = -3.0, 1.0, 1.0, 3.0


def load_runs(root, ds, d, seeds=None):
    base = Path(root) / ds / d
    out = {}
    for p in sorted(base.glob("seed*/eval.json")):
        s = int(p.parent.name[4:])
        if seeds is None or s in seeds:
            out[s] = C.load_json(p)
    return out


def mean(xs):
    return statistics.mean(xs) if xs else None


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def fmt(x, nd=2, sign=False):
    return "–" if x is None else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def paired(a, b, key):
    return {s: a[s][key] - b[s][key] for s in a if s in b and key in a[s] and key in b[s]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--criteria", nargs="+", default=["maskedkd", "rollout"])
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.5, 0.3, 0.25, 0.2, 0.15])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[1] / "results" / "wga_reference_cv2.json"))
    ap.add_argument("--output-root")
    a = ap.parse_args()
    root, ds = C.output_root(a.output_root), a.dataset
    keeps = sorted(a.keeps, reverse=True)
    res_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    res_dir.mkdir(parents=True, exist_ok=True)
    kd = load_runs(root, ds, "kd", a.seeds)
    runs = {(c, k): load_runs(root, ds, C.mask_dirname(c, k), a.seeds) for c in a.criteria for k in keeps}
    md = [f"# WGA collapse: {ds}", "",
          f"output root `{root}`, seeds {a.seeds}. Paired differences use the same seed (identical batches / mixup).",
          "Groups: " + ", ".join(f"g{i} {g}" for i, g in enumerate(GROUPS)), ""]

    # ---- 1. reproduction check vs the archived cv2 runs
    md += ["## 1. 재현 확인 (예전 cv2 결과와 비교)", "",
           f"기준: |새 평균 − 예전 평균| ≤ {REPRO_CLEAN} (clean), ≤ {REPRO_WGA} (WGA)", "",
           "| run | n | clean 새 / 예전 | WGA 새 / 예전 | corruption 새 / 예전 | 판정 |", "|---|---|---|---|---|---|"]
    ref = C.load_json(a.ref)["runs"] if Path(a.ref).exists() else {}
    repro_ok = True
    for d in ["teacher", "ce", "kd"] + [C.mask_dirname(c, k) for c in a.criteria for k in keeps]:
        new = load_runs(root, ds, d, None if d == "teacher" else a.seeds)
        if not new or d not in ref:
            continue
        old = ref[d]
        cells, ok = [], True
        for key, _ in METRICS:
            nv, ov = mean([e[key] for e in new.values() if key in e]), mean([e[key] for e in old.values() if key in e])
            cells.append(f"{fmt(nv)} / {fmt(ov)}")
            tol = {"clean_acc": REPRO_CLEAN, "wga": REPRO_WGA}.get(key)
            if tol is not None and nv is not None and ov is not None and abs(nv - ov) > tol:
                ok = False
        if d in ("teacher", "ce", "kd"):
            repro_ok &= ok
        md.append(f"| {d} | {len(new)} | " + " | ".join(cells) + f" | {'OK' if ok else '**다름**'} |")
    md += ["", "- Stage 0 재현: " + ("**OK**" if repro_ok else "**실패: 환경·데이터 확인 전에는 아래 결과를 해석하지 말 것**"), ""]

    # ---- 2. main table
    md += ["## 2. keep별 결과 (평균 ± 표준편차, Full KD 대비 같은 seed 차이)", ""]
    kd_row = {key: mean([e[key] for e in kd.values() if key in e]) for key, _ in METRICS}
    md += [f"Full KD (n={len(kd)}): clean {fmt(kd_row['clean_acc'])}, WGA {fmt(kd_row['wga'])}, "
           f"corruption {fmt(kd_row['corruption_acc'])}", "",
           "| 기준 | keep | n | clean | Δclean | WGA | ΔWGA | WGA 하락 seed | corruption | Δcorr |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    rows_csv = []
    for c in a.criteria:
        for k in keeps:
            r = runs[(c, k)]
            if not r:
                md.append(f"| {c} | {k:g} | 0 | – | – | – | – | – | – | – |")
                continue
            cells, rec = [], {"criterion": c, "keep": k, "n": len(r)}
            for key, name in METRICS:
                vals = [e[key] for e in r.values() if key in e]
                dif = paired(r, kd, key)
                dm = mean(list(dif.values()))
                cells += [f"{fmt(mean(vals))} ± {fmt(sd(vals))}", fmt(dm, sign=True)]
                if key == "wga":
                    cells.append(f"{sum(v < 0 for v in dif.values())}/{len(dif)}")
                rec.update({name: mean(vals), f"d_{name}": dm})
            for g in range(4):
                rec[f"g{g}"] = mean([e["group_acc"][g] for e in r.values() if "group_acc" in e])
            rows_csv.append(rec)
            md.append(f"| {c} | {k:g} | {len(r)} | " + " | ".join(cells) + " |")
    md.append("")

    # ---- 3. group accuracies
    md += ["## 3. 그룹별 정확도 (평균): 어느 그룹이 무너지나", "",
           "| 기준 | keep | " + " | ".join(f"g{i} {g}" for i, g in enumerate(GROUPS)) + " |",
           "|---|---|" + "---|" * 4]
    if kd and all("group_acc" in e for e in kd.values()):
        md.append("| Full KD | – | " + " | ".join(fmt(mean([e["group_acc"][g] for e in kd.values()])) for g in range(4)) + " |")
    for rec in rows_csv:
        md.append(f"| {rec['criterion']} | {rec['keep']:g} | " + " | ".join(fmt(rec[f'g{g}']) for g in range(4)) + " |")
    md.append("")

    # ---- 4. verdicts (pre-registered)
    md += ["## 4. 판정 (사전 고정)", ""]
    collapse = None
    if "maskedkd" in a.criteria:
        for k in keeps:
            dif = paired(runs[("maskedkd", k)], kd, "wga")
            if len(dif) == len(a.seeds) and mean(list(dif.values())) <= COLLAPSE_DROP and all(v < 0 for v in dif.values()):
                collapse = k
                break
        incomplete = [k for k in keeps if len(runs[("maskedkd", k)]) < len(a.seeds)]
        if collapse is None:
            md.append(f"- **MaskedKD 붕괴 지점: 없음** (ΔWGA ≤ {COLLAPSE_DROP}%p이고 모든 seed 음수인 keep 없음)"
                      + (f" — 미완료 keep {incomplete}" if incomplete else ""))
        else:
            dc = mean(list(paired(runs[("maskedkd", collapse)], kd, "clean_acc").values()))
            dw = mean(list(paired(runs[("maskedkd", collapse)], kd, "wga").values()))
            flat = dc is not None and abs(dc) < CLEAN_FLAT
            md.append(f"- **MaskedKD 붕괴 지점: keep {collapse:g}** (ΔWGA {dw:+.2f}%p, Δclean {dc:+.2f}%p → "
                      + ("clean은 유지, WGA만 하락" if flat else "clean도 같이 하락") + ")"
                      + (f" — 미완료 keep {incomplete}" if incomplete else ""))
        if "rollout" in a.criteria:
            for k in keeps:
                d = paired(runs[("rollout", k)], runs[("maskedkd", k)], "wga")
                if d:
                    md.append(f"- keep {k:g}: rollout − MaskedKD WGA {mean(list(d.values())):+.2f}%p "
                              f"({sum(v > 0 for v in d.values())}/{len(d)} seed에서 rollout이 높음)")
    md += ["", "그림: `wga_collapse.png` (keep에 따른 clean / WGA, 점선 = Full KD)", ""]

    (res_dir / "wga_collapse.md").write_text("\n".join(md) + "\n")
    if rows_csv:
        with open(res_dir / "wga_collapse.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    plot(rows_csv, kd_row, a.criteria, res_dir / "wga_collapse.png")
    print("\n".join(md))
    print(f"wrote {res_dir / 'wga_collapse.md'}")


def plot(rows, kd_row, criteria, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    colors = {"maskedkd": "#2a78d6", "rollout": "#1baf7a"}
    for ax, (name, key) in zip(axes, (("clean", "clean_acc"), ("WGA", "wga"))):
        for c in criteria:
            pts = sorted((r["keep"], r[name]) for r in rows if r["criterion"] == c and r[name] is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=c, color=colors.get(c))
        if kd_row.get(key) is not None:
            ax.axhline(kd_row[key], ls="--", color="gray", label="Full KD")
        ax.set_title(name)
        ax.set_xlabel("keep ratio (teacher tokens kept)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
