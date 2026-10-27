"""Follow-up sweep summary (Waterbirds): student results of the control / candidate criteria, paired with Full KD
(same seed), pre-registered verdicts, and the efficiency table.  -> $RESULTS_DIR/more.{md,csv}

    uv run python -m stage0.summarize_more

Runs read (dir {name}_k{keep}/seed*/eval.json): maskedkd, rollout, random, mk_filt, tam_r_g0 (TrustMask),
oracle_bird, oracle_bg at the keeps given below. Missing runs are shown as "–" (the sweep may still be going).

Pre-registered rules (fixed before the runs; paired dWGA = run - Full KD, same seed; "3/3" = every seed):
  R1 random control   random collapses too            : dWGA(random) <= -3 and 3/3 negative
                      MaskedKD selection is worse      : WGA(MaskedKD) - WGA(random) <= -3 and 3/3 negative
  R2 oracle (causal)  bird patches suffice             : dWGA(oracle_bird) > -3
                      background patches cause it      : dWGA(oracle_bg) <= -3 and 3/3 negative
                      -> "background tokens cause the collapse" if both hold
  R3 sink filter      sinks contribute                 : WGA(mk_filt) - WGA(MaskedKD) >= +5 and 3/3 positive
  R4 TrustMask        vs rollout                       : WGA(tam_r_g0) - WGA(rollout) mean and seeds (reported, no
                                                         threshold: the g = 0 setting was chosen after seeing earlier
                                                         test results, so this is not a clean confirmatory test)
  R5 rollout limit    rollout collapses at keep k      : dWGA(rollout, k) <= -3 and 3/3 negative, k in (0.1, 0.05)
"""
import argparse
import csv
import os
import statistics
from pathlib import Path

from stage0 import common as C
from stage0.summarize_wga import GROUPS, fmt, load_runs, mean, paired, sd

DROP, SINK_GAIN = -3.0, 5.0
RUNS = [("maskedkd", 0.15), ("maskedkd", 0.25), ("random", 0.25), ("random", 0.15), ("mk_filt", 0.15),
        ("tam_r_g0", 0.15), ("oracle_bird", 0.15), ("oracle_bg", 0.15),
        ("rollout", 0.25), ("rollout", 0.15), ("rollout", 0.1), ("rollout", 0.05)]
LABEL = {"maskedkd": "MaskedKD", "random": "random", "mk_filt": "MaskedKD + sink 필터", "tam_r_g0": "TrustMask",
         "oracle_bird": "oracle: 새만", "oracle_bg": "oracle: 배경만", "rollout": "rollout"}


def epoch_time(root, ds, d):
    ts = []
    for p in (Path(root) / ds / d).glob("seed*/log.csv"):
        with open(p) as fh:
            ts += [float(r["epoch_time_s"]) for r in csv.DictReader(fh) if r.get("epoch_time_s")]
    return statistics.mean(ts) if ts else None


def verdict(dif, rule):
    v = list(dif.values())
    return bool(v) and rule(statistics.mean(v)) and len(v) >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-flops", action="store_true", help="skip the GFLOPs table (smoke)")
    ap.add_argument("--output-root")
    a = ap.parse_args()
    root, ds = C.output_root(a.output_root), a.dataset
    res = Path(os.environ.get("RESULTS_DIR", "results"))
    res.mkdir(parents=True, exist_ok=True)
    kd = load_runs(root, ds, "kd", a.seeds)
    R = {(c, k): load_runs(root, ds, C.mask_dirname(c, k), a.seeds) for c, k in RUNS}
    md = ["# 후속 실험: 대조군·인과·후보 방법 (Waterbirds)", "",
          f"seed {a.seeds}, Full KD 대비 같은 seed 차이. 비어 있는 칸은 아직 안 끝났거나 돌리지 않은 run.", ""]
    if kd:
        md.append(f"Full KD (n={len(kd)}): clean {fmt(mean([e['clean_acc'] for e in kd.values()]))}, "
                  f"WGA {fmt(mean([e['wga'] for e in kd.values()]))}, 그룹 "
                  + " / ".join(fmt(mean([e['group_acc'][g] for e in kd.values()]), 1) for g in range(4)))
    md += ["", "## 1. 결과", "",
           "| 방법 | keep | n | clean | Δclean | WGA | ΔWGA | WGA 하락 seed | g1 | g2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for c, k in RUNS:
        r = R[(c, k)]
        if not r:
            md.append(f"| {LABEL[c]} | {k:g} | 0 | – | – | – | – | – | – | – |")
            continue
        dc, dw = paired(r, kd, "clean_acc"), paired(r, kd, "wga")
        g = [mean([e["group_acc"][i] for e in r.values()]) for i in range(4)]
        rec = {"criterion": c, "keep": k, "n": len(r), "clean": mean([e["clean_acc"] for e in r.values()]),
               "wga": mean([e["wga"] for e in r.values()]), "wga_sd": sd([e["wga"] for e in r.values()]),
               "d_clean": mean(list(dc.values())), "d_wga": mean(list(dw.values())), **{f"g{i}": g[i] for i in range(4)}}
        csv_rows.append(rec)
        md.append(f"| {LABEL[c]} | {k:g} | {len(r)} | {fmt(rec['clean'])} | {fmt(rec['d_clean'], sign=True)} | "
                  f"{fmt(rec['wga'])} ± {fmt(rec['wga_sd'])} | {fmt(rec['d_wga'], sign=True)} | "
                  f"{sum(v < 0 for v in dw.values())}/{len(dw)} | {fmt(g[1], 1)} | {fmt(g[2], 1)} |")

    def dif(c, k, ref=None):
        return paired(R[(c, k)], kd if ref is None else R[ref], "wga")

    def line(d, pos=False):
        v = list(d.values())
        if not v:
            return "run 없음"
        n = sum((x > 0) if pos else (x < 0) for x in v)
        return f"{statistics.mean(v):+.1f}%p ({n}/{len(v)} seed {'양수' if pos else '음수'})"

    md += ["", "## 2. 판정 (사전 고정)", ""]
    for k in (0.15, 0.25):
        d1 = dif("random", k)
        d2 = dif("maskedkd", k, ("random", k))
        if d1:
            md.append(f"- **R1 random 대조 (keep {k:g})**: random ΔWGA {line(d1)} → "
                      + ("random도 붕괴" if verdict(d1, lambda m: m <= DROP) and all(x < 0 for x in d1.values())
                         else "random은 붕괴 안 함"))
        if d2:
            md.append(f"  - MaskedKD − random WGA {line(d2)} → "
                      + ("**MaskedKD의 선택이 무작위보다 나쁨**" if verdict(d2, lambda m: m <= DROP)
                         and all(x < 0 for x in d2.values()) else "무작위보다 나쁘다고 할 수 없음"))
    ob, og = dif("oracle_bird", 0.15), dif("oracle_bg", 0.15)
    if ob or og:
        bird_ok = verdict(ob, lambda m: m > DROP)
        bg_bad = verdict(og, lambda m: m <= DROP) and all(x < 0 for x in og.values())
        md.append(f"- **R2 oracle (keep 0.15)**: 새만 ΔWGA {line(ob)}, 배경만 ΔWGA {line(og)} → "
                  + ("**배경 token이 붕괴의 원인 (인과)**" if bird_ok and bg_bad else
                     f"새만 {'회복' if bird_ok else '회복 안 됨'}, 배경만 {'붕괴' if bg_bad else '붕괴 안 함'}"))
    d3 = paired(R[("mk_filt", 0.15)], R[("maskedkd", 0.15)], "wga")
    if d3:
        md.append(f"- **R3 sink 필터**: (MaskedKD + sink 필터) − MaskedKD WGA {line(d3, pos=True)} → "
                  + ("**sink 토큰이 붕괴에 기여**" if verdict(d3, lambda m: m >= SINK_GAIN) and all(x > 0 for x in d3.values())
                     else "기여 확인 안 됨"))
    d4 = paired(R[("tam_r_g0", 0.15)], R[("rollout", 0.15)], "wga")
    if d4:
        md.append(f"- **R4 TrustMask vs rollout (keep 0.15)**: WGA 차이 {line(d4, pos=True)} (보고만; g=0 설정은 예전 test 결과를 보고 "
                  "고른 것이라 확증 검정이 아님)")
    for k in (0.1, 0.05):
        d5 = dif("rollout", k)
        if d5:
            md.append(f"- **R5 rollout 한계 (keep {k:g})**: ΔWGA {line(d5)} → "
                      + ("rollout도 붕괴" if verdict(d5, lambda m: m <= DROP) and all(x < 0 for x in d5.values())
                         else "rollout 유지"))

    md += ["", "## 3. 효율 (teacher 계산량, 1장당)", ""]
    if not a.no_flops:
        from stage0.flops import selection_gflops, teacher_gflops
        full = teacher_gflops(196, 2)
        md += [f"Full teacher {full:.2f} GFLOPs (DeiT-B). 학습 시간은 log.csv의 epoch 평균 (같은 GPU에 동시에 돈 run 수에 따라 "
               "달라지므로 참고용).", "",
               "| 방법 | keep | k | teacher GFLOPs | Full 대비 | 선택 GFLOPs | epoch 시간 (s) |", "|---|---|---|---|---|---|---|"]
        kd_t = epoch_time(root, ds, "kd")
        md.append(f"| Full KD | 1 | 196 | {full:.2f} | 100% | 0 | {fmt(kd_t, 1)} |")
        for c, k in RUNS:
            kk = round(k * 196)
            tg = teacher_gflops(kk, 2)
            base = "tam_r" if c == "tam_r_g0" else c
            try:
                sg = f"{selection_gflops(base, 'attn_last', 2):.3f}"
            except ValueError:
                sg = "0 (분할 마스크)" if c.startswith("oracle") else "–"
            md.append(f"| {LABEL[c]} | {k:g} | {kk} | {tg:.2f} | {100 * tg / full:.0f}% | {sg} | "
                      f"{fmt(epoch_time(root, ds, C.mask_dirname(c, k)), 1)} |")
    with open(res / "more.csv", "w", newline="") as fh:
        if csv_rows:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    md.append("")
    (res / "more.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {res / 'more.md'}")


if __name__ == "__main__":
    main()
