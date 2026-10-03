"""Stage 2 summary -> results/stage2_summary.md

    uv run python -m stage0.summarize_stage2 [--datasets cub]

Rows: Full KD (Stage 0 kd/seed*, keep 1.0), maskedkd / random at each keep, CE (Stage 0 ce/seed*).
Metric: clean test top-1 of the last-epoch checkpoint (eval.json); corruption accuracy alongside.
Paired differences use the same seed (kd/seedN <-> {criterion}_k{keep}/seedN; maskedkd <-> random).

Margin rule (fixed in advance): the "MaskedKD limit ratio" is the largest keep at which the maskedkd mean is
at least 0.5 %p below Full KD. If no keep reaches that, the summary says no loss even at the smallest keep and
that lower ratios (0.1, 0.05) need to be added.
"""
import argparse
import os
import statistics
from pathlib import Path

from stage0 import common as C
from stage0.flops import teacher_gflops
from stage0.masking import TRAIN_CRITERIA, num_keep

SEEDS = (0, 1, 2)
LIMIT_DROP = 0.5


def load(out_root, ds, dirname):
    out = {}
    for s in SEEDS:
        p = Path(out_root) / ds / dirname / f"seed{s}" / "eval.json"
        if p.exists():
            out[s] = C.load_json(p)
    return out


def ms(vals):
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.2f} (n=1)"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}" + ("" if len(vals) == 3 else f" (n={len(vals)})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cub"], choices=C.DATASETS)
    ap.add_argument("--keeps", type=float, nargs="+", default=list(C.MASK_KEEPS))
    ap.add_argument("--output-root")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    args = ap.parse_args(argv)
    out_root = C.output_root(args.output_root)
    keeps = sorted(args.keeps, reverse=True)

    md = ["# Stage 2 summary: MaskedKD / Random teacher-token reduction vs Full KD", "",
          "Test = clean top-1 at the last epoch (mean ± std over seeds). \"vs Full KD\" = mean difference; per-seed "
          "differences pair runs with the same seed (same data order, augmentation and mixup). Teacher GFLOPs: one "
          "DeiT-B forward per image with k = round(keep·196) patch tokens (DeiT convention, FlopCounterMode/2).", ""]
    smoke = False
    for ds in args.datasets:
        kd, ce = load(out_root, ds, "kd"), load(out_root, ds, "ce")
        runs = {(c, k): load(out_root, ds, C.mask_dirname(c, k)) for c in TRAIN_CRITERIA for k in keeps}
        smoke |= any(not e.get("pretrained", True) for v in [kd, ce, *runs.values()] for e in v.values())
        kd_mean = statistics.mean(e["clean_acc"] for e in kd.values()) if kd else None
        name = "CUB" if ds == "cub" else "Waterbirds"
        md += [f"## {name}", "",
               "| 기준 | 비율 | teacher GFLOPs/img | test acc (mean ± std) | Full KD 대비 | seed별 차이 (vs KD) | "
               "corruption acc |",
               "|---|---|---|---|---|---|---|",
               f"| Full KD | 1.0 | {teacher_gflops(196, C.NUM_CLASSES[ds]):.2f} | "
               f"{ms([e['clean_acc'] for e in kd.values()])} | 0 | | {ms([e['corruption_acc'] for e in kd.values()])} |"]
        diffs = {}
        for c in TRAIN_CRITERIA:
            for k in keeps:
                r = runs[(c, k)]
                vals = [e["clean_acc"] for e in r.values()]
                paired = {s: r[s]["clean_acc"] - kd[s]["clean_acc"] for s in SEEDS if s in r and s in kd}
                diffs[(c, k)] = paired
                d = f"{statistics.mean(vals) - kd_mean:+.2f}" if vals and kd_mean is not None else "–"
                per = ", ".join(f"s{s} {v:+.2f}" for s, v in paired.items())
                md.append(f"| {c} | {k:g} | {teacher_gflops(num_keep(k), C.NUM_CLASSES[ds]):.2f} | {ms(vals)} | {d} | "
                          f"{per} | {ms([e['corruption_acc'] for e in r.values()])} |")
        ce_vals = [e["clean_acc"] for e in ce.values()]
        md += [f"| CE | – | 0 | {ms(ce_vals)} | "
               + (f"{statistics.mean(ce_vals) - kd_mean:+.2f}" if ce_vals and kd_mean is not None else "–")
               + " | " + ", ".join(f"s{s} {ce[s]['clean_acc'] - kd[s]['clean_acc']:+.2f}" for s in SEEDS
                                   if s in ce and s in kd)
               + f" | {ms([e['corruption_acc'] for e in ce.values()])} |", ""]

        md += ["### maskedkd − random (same seed, same keep)", ""]
        for k in keeps:
            a, b = runs[("maskedkd", k)], runs[("random", k)]
            per = {s: a[s]["clean_acc"] - b[s]["clean_acc"] for s in SEEDS if s in a and s in b}
            md.append(f"- keep {k:g}: " + (f"mean {statistics.mean(per.values()):+.2f}%p ("
                                            + ", ".join(f"s{s} {v:+.2f}" for s, v in per.items()) + ")"
                                            if per else "–"))
        md.append("")

        md += ["### 여유 구간 판정 (사전 고정: Full KD 대비 maskedkd 평균 −0.5%p 이상 하락하는 가장 큰 비율)", ""]
        complete = kd_mean is not None and all(len(runs[("maskedkd", k)]) == len(SEEDS) for k in keeps) \
            and len(kd) == len(SEEDS)
        drops = {k: statistics.mean(e["clean_acc"] for e in runs[("maskedkd", k)].values()) - kd_mean
                 for k in keeps if runs[("maskedkd", k)] and kd_mean is not None}
        limit = next((k for k in keeps if k in drops and drops[k] <= -LIMIT_DROP), None)
        status = "" if complete else " (미완료: 일부 seed 결과 없음 — 판정 보류)"
        if limit is not None:
            md.append(f"- **{name}: MaskedKD 한계 비율 = {limit:g}** (Full KD 대비 {drops[limit]:+.2f}%p){status}")
        else:
            md.append(f"- **{name}: {min(keeps):g}에서도 손실 없음 → 더 낮은 비율(0.1, 0.05) 추가 필요**{status}")
        md.append("")
    if smoke:
        md.insert(2, "> ⚠️ SMOKE TEST — random-init models and/or reduced data. Not real results.\n")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.results_dir) / "stage2_summary.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
