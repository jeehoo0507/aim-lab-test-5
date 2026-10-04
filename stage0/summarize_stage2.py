"""Stage 2 summary -> results/stage2_summary.md

    uv run python -m stage0.summarize_stage2 [--datasets cub]

Rows: Full KD (Stage 0 kd/seed*, keep 1.0), maskedkd / random at each keep, CE (Stage 0 ce/seed*).
Metric: clean test top-1 of the last-epoch checkpoint (eval.json); corruption accuracy alongside.
Paired differences use the same seed (kd/seedN <-> {criterion}_k{keep}/seedN; maskedkd <-> random).

Stage 3 pilot (rollout / tam / tam_var / tam_oracle rows and a selection-cost column appear only when such runs
exist): paired differences (tam, tam_var, tam_oracle vs maskedkd; tam, tam_var vs rollout), measured tam_var
teacher GFLOPs, per-epoch cache-vs-oracle top-k overlap, and the pilot verdict (fixed in advance; "beats" =
higher on every paired seed (>= 2) and mean >= +0.3 %p):
  tam or tam_var beats maskedkd                      -> promising: extend to the 3-seed main experiment
                                                        (also report whether it beats rollout)
  neither does, tam_oracle beats maskedkd (keep 0.15) -> idea valid, the cache is the bottleneck
  neither does, tam_oracle does not either           -> unlikely: revisit the selection rule (pool size, g)

Margin rule (fixed in advance): the "MaskedKD limit ratio" is the largest keep at which the maskedkd mean is
at least 0.5 %p below Full KD. If no keep reaches that, the summary says no loss even at the smallest keep and
that lower ratios (0.1, 0.05) need to be added.
"""
import argparse
import csv
import os
import statistics
from pathlib import Path

from stage0 import common as C
from stage0.flops import selection_gflops, teacher_gflops
from stage0.masking import STAGE2_CRITERIA, TAM_CACHE_CRITERIA, TRAIN_CRITERIA, num_keep

SEEDS = (0, 1, 2)
LIMIT_DROP = 0.5
PILOT_GAIN = 0.3
PILOT_CRITERIA = ("rollout", "tam", "tam_var", "tam_oracle", "tam_r")


def paired(a, b):
    return {s: a[s]["clean_acc"] - b[s]["clean_acc"] for s in SEEDS if s in a and s in b}


def beats(per):
    """Pre-registered "beats": >= 2 paired seeds, all positive, mean >= +0.3 %p."""
    return len(per) >= 2 and all(v > 0 for v in per.values()) and statistics.mean(per.values()) >= PILOT_GAIN - 1e-9


def fmt_pair(per):
    return (f"mean {statistics.mean(per.values()):+.2f}%p (" + ", ".join(f"s{s} {v:+.2f}" for s, v in per.items())
            + ")") if per else "–"


def overlap_by_epoch(out_root, ds, dirname):
    """Per-epoch mean of log.csv cache_oracle_overlap over the run's seeds."""
    per_epoch = {}
    for s in SEEDS:
        p = Path(out_root) / ds / dirname / f"seed{s}" / "log.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                if r.get("cache_oracle_overlap"):
                    per_epoch.setdefault(int(r["epoch"]), []).append(float(r["cache_oracle_overlap"]))
    return {e: statistics.mean(v) for e, v in sorted(per_epoch.items())}


def pilot_section(out_root, ds, name, runs, keeps, pilot):
    md = ["### Stage 3 pilot: 같은 seed 짝 차이", ""]
    pairs = [("tam", "maskedkd"), ("tam_var", "maskedkd"), ("tam_oracle", "maskedkd"), ("tam", "rollout"),
             ("tam_var", "rollout"), ("tam_r", "maskedkd"), ("tam_r", "rollout"), ("tam_r", "tam")]
    for a, b in pairs:
        if a not in pilot or (b != "maskedkd" and b not in pilot):
            continue
        for k in keeps:
            if runs[(a, k)]:
                md.append(f"- {a} − {b}, keep {k:g}: {fmt_pair(paired(runs[(a, k)], runs[(b, k)]))}")
    md.append("")
    if "tam_var" in pilot:
        md += ["tam_var teacher GFLOPs/img (measured, bucket-weighted) vs fixed k:", ""]
        for k in keeps:
            if runs[("tam_var", k)]:
                gm = measured_gflops(out_root, ds, C.mask_dirname("tam_var", k), SEEDS)
                g = teacher_gflops(num_keep(k), C.NUM_CLASSES[ds])
                md.append(f"- keep {k:g}: " + (f"{gm:.3f} vs {g:.3f} ({100 * (gm / g - 1):+.1f}%)" if gm else "–"))
        md.append("")
    ov_lines = []
    for c in TAM_CACHE_CRITERIA:
        for k in keeps:
            ov = overlap_by_epoch(out_root, ds, C.mask_dirname(c, k)) if runs[(c, k)] else {}
            if ov:
                vals = list(ov.values())
                marks = {e: v for e, v in ov.items() if e in (0, len(ov) // 2, len(ov) - 1)}
                ov_lines.append(f"- {c}, keep {k:g}: mean {100 * statistics.mean(vals):.1f}% over {len(vals)} epochs ("
                                + ", ".join(f"epoch {e + 1}: {100 * v:.1f}%" for e, v in marks.items()) + ")")
    if ov_lines:
        md += ["학습 중 cache_oracle_overlap (캐시 top-k ∩ 같은 view의 teacher top-k, 에포치마다 첫 micro-batch):", "",
               *ov_lines, ""]

    # verdict
    win, incomplete = [], []
    for c in ("tam", "tam_var", "tam_r"):
        for k in keeps:
            if c in pilot and runs[(c, k)]:
                per = paired(runs[(c, k)], runs[("maskedkd", k)])
                if beats(per):
                    win.append((c, k, statistics.mean(per.values())))
                elif len(per) < 2:
                    incomplete.append(f"{c} @ {k:g}")
    oracle_k = [k for k in keeps if runs[("tam_oracle", k)]]
    oracle_win = [k for k in oracle_k if beats(paired(runs[("tam_oracle", k)], runs[("maskedkd", k)]))]
    oracle_incomplete = not oracle_k or any(len(paired(runs[("tam_oracle", k)], runs[("maskedkd", k)])) < 2
                                            for k in oracle_k)
    md += ["### 파일럿 판단 (사전 고정; 이긴다 = 짝 seed 모두(≥2) 높고 평균 +0.3%p 이상)", ""]
    if win:
        txt = "; ".join(f"{c} @ keep {k:g} ({m:+.2f}%p)" for c, k, m in win)
        md.append(f"- **{name}: 가능성 있음 → 본 실험(3 seed)으로 확장** — maskedkd 대비 이김: {txt}")
        for c, k, _ in win:
            if "rollout" in pilot and runs[("rollout", k)]:
                per = paired(runs[(c, k)], runs[("rollout", k)])
                if beats(per):
                    md.append(f"  - {c} @ keep {k:g}: rollout도 이김 ({fmt_pair(per)})")
                else:
                    md.append(f"  - {c} @ keep {k:g}: **rollout 대비 우위 없음** ({fmt_pair(per)}) → 본 실험에서 S를 "
                              "rollout으로 바꾼 TAM 검토 근거")
            else:
                md.append(f"  - {c} @ keep {k:g}: rollout 결과 없음")
    elif incomplete or ("tam_oracle" in pilot and oracle_incomplete) or "tam_oracle" not in pilot:
        missing = incomplete + ([] if "tam_oracle" in pilot and not oracle_incomplete else ["tam_oracle (짝 seed ≥2)"])
        md.append(f"- **{name}: 미완료 — 판단 보류** (부족: {', '.join(missing)})")
    elif oracle_win:
        md.append(f"- **{name}: 아이디어는 유효, 캐시가 병목** — tam/tam_var는 maskedkd를 못 이기고 tam_oracle은 이김 "
                  f"(keep {', '.join(f'{k:g}' for k in oracle_win)}) → 캐시 진단 결과(results/stage3_cache_diag.md)로 캐시 "
                  "개선 후 재실험")
    else:
        md.append(f"- **{name}: 가능성 낮음 → 선택 규칙(풀 크기, g 스케줄) 재검토** — tam/tam_var, tam_oracle 모두 "
                  "maskedkd를 못 이김")
    md.append("")
    return md


def measured_gflops(out_root, ds, dirname, seeds):
    """Mean per-image teacher GFLOPs over all epochs of the given runs (TAM runs log it per epoch)."""
    vals = []
    for s in seeds:
        p = Path(out_root) / ds / dirname / f"seed{s}" / "log.csv"
        if p.exists():
            vals += [float(r["teacher_gflops"]) for r in csv.DictReader(open(p)) if r.get("teacher_gflops")]
    return statistics.mean(vals) if vals else None


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


def wga_section(kd, ce, runs, criteria, keeps):
    """Worst-group accuracy table (Waterbirds), same layout as the clean table; paired by seed vs Full KD."""
    kd_w = [e["wga"] for e in kd.values() if "wga" in e]
    kd_m = statistics.mean(kd_w) if kd_w else None
    md = ["### Worst-group accuracy (WGA)", "",
          "| 기준 | 비율 | WGA (mean ± std) | Full KD 대비 | seed별 차이 (vs KD) |", "|---|---|---|---|---|",
          f"| Full KD | 1.0 | {ms(kd_w)} | 0 | |"]
    rows = [(c, k, runs[(c, k)]) for c in criteria for k in keeps if runs[(c, k)]] + [("CE", None, ce)]
    for c, k, r in rows:
        vals = [e["wga"] for e in r.values() if "wga" in e]
        paired = {s: r[s]["wga"] - kd[s]["wga"] for s in SEEDS if s in r and s in kd and "wga" in r[s]}
        d = f"{statistics.mean(vals) - kd_m:+.2f}" if vals and kd_m is not None else "–"
        per = ", ".join(f"s{s} {v:+.2f}" for s, v in paired.items())
        md.append(f"| {c} | {'–' if k is None else f'{k:g}'} | {ms(vals)} | {d} | {per} |")
    return md + [""]


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
        pilot = [c for c in PILOT_CRITERIA if any(runs[(c, k)] for k in keeps)]
        criteria = [c for c in TRAIN_CRITERIA if c in STAGE2_CRITERIA or c in pilot]   # pilot rows only if run
        cost = {c: selection_gflops(c, num_classes=C.NUM_CLASSES[ds]) for c in criteria} if pilot else {}
        sel_hdr, sel_sep = (" 선택 비용 GFLOPs/img |", "---|") if pilot else ("", "")
        md += [f"## {name}", "",
               f"| 기준 | 비율 | teacher GFLOPs/img |{sel_hdr} test acc (mean ± std) | Full KD 대비 | "
               "seed별 차이 (vs KD) | corruption acc |",
               f"|---|---|---|{sel_sep}---|---|---|---|",
               f"| Full KD | 1.0 | {teacher_gflops(196, C.NUM_CLASSES[ds]):.2f} |{' 0 |' if pilot else ''} "
               f"{ms([e['clean_acc'] for e in kd.values()])} | 0 | | {ms([e['corruption_acc'] for e in kd.values()])} |"]
        for c in criteria:
            for k in keeps:
                r = runs[(c, k)]
                if c in pilot and not r:
                    continue                      # pilot criteria: only the keeps that were run
                vals = [e["clean_acc"] for e in r.values()]
                paired = {s: r[s]["clean_acc"] - kd[s]["clean_acc"] for s in SEEDS if s in r and s in kd}
                d = f"{statistics.mean(vals) - kd_mean:+.2f}" if vals and kd_mean is not None else "–"
                per = ", ".join(f"s{s} {v:+.2f}" for s, v in paired.items())
                g = teacher_gflops(num_keep(k), C.NUM_CLASSES[ds])
                if c == "tam_var":
                    gm = measured_gflops(out_root, ds, C.mask_dirname(c, k), list(r))
                    g_txt = f"{gm:.2f} (measured)" if gm is not None else f"{g:.2f} (nominal)"
                else:
                    g_txt = f"{g:.2f}"
                sel = ""
                if pilot:
                    sel = f" {cost[c]:.3f}" + (" (진단용, 비용 비교 대상 아님)" if c == "tam_oracle" else "") + " |"
                md.append(f"| {c} | {k:g} | {g_txt} |{sel} {ms(vals)} | {d} | "
                          f"{per} | {ms([e['corruption_acc'] for e in r.values()])} |")
        ce_vals = [e["clean_acc"] for e in ce.values()]
        md += [f"| CE | – | 0 |{' – |' if pilot else ''} {ms(ce_vals)} | "
               + (f"{statistics.mean(ce_vals) - kd_mean:+.2f}" if ce_vals and kd_mean is not None else "–")
               + " | " + ", ".join(f"s{s} {ce[s]['clean_acc'] - kd[s]['clean_acc']:+.2f}" for s in SEEDS
                                   if s in ce and s in kd)
               + f" | {ms([e['corruption_acc'] for e in ce.values()])} |", ""]
        if pilot:
            md += ["선택 비용 = 토큰 선택에 드는 추가 연산 (student 학습 forward와 마스킹 teacher forward 제외): maskedkd·tam·"
                   "tam_var는 student 마지막 블록 attention 재계산, rollout은 전 블록 attention + rollout 곱, tam_oracle은 "
                   "teacher 전체 forward 1회 추가.", ""]

        md += ["### maskedkd − random (same seed, same keep)", ""]
        for k in keeps:
            a, b = runs[("maskedkd", k)], runs[("random", k)]
            per = {s: a[s]["clean_acc"] - b[s]["clean_acc"] for s in SEEDS if s in a and s in b}
            md.append(f"- keep {k:g}: " + (f"mean {statistics.mean(per.values()):+.2f}%p ("
                                            + ", ".join(f"s{s} {v:+.2f}" for s, v in per.items()) + ")"
                                            if per else "–"))
        md.append("")

        if any("wga" in e for e in kd.values()):   # Waterbirds: worst-group accuracy (eval.json "wga")
            md += wga_section(kd, ce, runs, criteria, keeps)
        if pilot:
            md += pilot_section(out_root, ds, name, runs, keeps, pilot)
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
