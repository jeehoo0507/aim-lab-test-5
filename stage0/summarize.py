"""Stage 0 summary table and pre-registered verdict (5절) -> results/stage0_summary.md

    uv run python -m stage0.summarize

Verdict (fixed in advance):
  metric      CUB: clean top-1   Waterbirds: WGA
  PASS        mean over 3 seeds of (KD - CE) >= 1.0 %p  AND  all 3 same-seed differences > 0
  OOD PASS    clean verdict fails but corruption accuracy satisfies the same rule
  warning     teacher metric < student CE mean
"""
import argparse
import os
import statistics
from pathlib import Path

from stage0 import common as C

SEEDS = (0, 1, 2)
METRIC = {"cub": "clean_acc", "waterbirds": "wga"}
THRESH = 1.0


def load_evals(out_root, dataset, mode):
    out = {}
    for s in SEEDS if mode != "teacher" else (0,):
        p = Path(out_root) / dataset / mode / f"seed{s}" / "eval.json"
        if p.exists():
            out[s] = C.load_json(p)
    return out


def fmt(evals, key):
    vals = [e[key] for e in evals.values() if e.get(key) is not None]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.2f}" + ("" if len(evals) == 1 else " (n=1)")
    sd = statistics.stdev(vals)
    return f"{statistics.mean(vals):.2f} ± {sd:.2f}" + ("" if len(vals) == 3 else f" (n={len(vals)})")


def verdict(ce, kd, key):
    """Returns (status, detail): status in pass / fail / incomplete."""
    seeds = [s for s in SEEDS if s in ce and s in kd]
    if len(seeds) < len(SEEDS):
        return "incomplete", f"paired seeds available: {seeds}"
    diffs = [kd[s][key] - ce[s][key] for s in seeds]
    mean = statistics.mean(diffs)
    ok = mean >= THRESH and all(d > 0 for d in diffs)
    return ("pass" if ok else "fail"), f"KD−CE = {mean:+.2f}%p (per seed {', '.join(f'{d:+.2f}' for d in diffs)})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    args = ap.parse_args(argv)
    out_root = C.output_root(args.output_root)

    rows, verdict_lines, notes, n_corr = [], [], [], set()
    smoke = False
    for ds in C.DATASETS:
        evals = {m: load_evals(out_root, ds, m) for m in ("teacher", "ce", "kd", "kd_alpha1")}
        smoke |= any(not e.get("pretrained", True) for v in evals.values() for e in v.values())
        n_corr |= {(e["corruption_n_conditions"], e["corruption_n_images"]) for v in evals.values() for e in v.values()}
        for label, mode in (("Teacher", "teacher"), ("CE", "ce"), ("KD", "kd"), ("KD (α=1.0)", "kd_alpha1")):
            if mode == "kd_alpha1" and not evals[mode]:
                continue
            e = evals[mode]
            rows.append(f"| {'CUB' if ds == 'cub' else 'Waterbirds'} | {label} | {fmt(e, 'clean_acc')} | "
                        f"{fmt(e, 'wga') if ds == 'waterbirds' else '–'} | {fmt(e, 'corruption_acc')} |")

        key = METRIC[ds]
        name = "CUB" if ds == "cub" else "Waterbirds"
        for kd_mode, tag in (("kd", ""), ("kd_alpha1", " [α=1.0 fallback]")):
            if kd_mode == "kd_alpha1" and not evals[kd_mode]:
                continue
            st, detail = verdict(evals["ce"], evals[kd_mode], key)
            if st == "pass":
                final = "통과"
            elif st == "incomplete":
                final = "미완료"
            else:
                st2, detail2 = verdict(evals["ce"], evals[kd_mode], "corruption_acc")
                final = "OOD 통과" if st2 == "pass" else "실패"
                detail += f"; corruption {detail2}"
            verdict_lines.append(f"- **{name}{tag}: {final}** — {key}: {detail}")

        t, ce = evals["teacher"].get(0), [e[key] for e in evals["ce"].values()]
        if t is not None and ce and t[key] < statistics.mean(ce):
            notes.append(f"⚠️ {name}: teacher {key} ({t[key]:.2f}) < student CE mean ({statistics.mean(ce):.2f})")
        diag = Path(out_root) / ds / "teacher" / "seed0" / "diagnosis.json"
        if diag.exists():
            d = C.load_json(diag)
            for view in ("train_view", "test_clean"):
                v = d[view]
                e1, e4 = (v[f"wrong_entropy_norm_tau{t}"] for t in (1, 4))
                ent = "n/a (C=2)" if e1 is None else f"{e1:.3f} / τ4 {e4:.3f}"
                notes.append(f"{name} teacher {view}: acc {v['acc'] * 100:.2f}%, p_T(y) τ1 {v['p_true_tau1']:.3f} / "
                             f"τ4 {v['p_true_tau4']:.3f}, wrong-class entropy/log(C−1) τ1 {ent}")

    md = ["# Stage 0 summary: CE vs Full KD", ""]
    if smoke:
        md += ["> ⚠️ SMOKE TEST — random-init models and/or reduced data. Not real results.", ""]
    md += ["Values: 3-seed mean ± std (teacher: single run). Students at the last epoch; teacher at best val.",
           "Corruption = mean accuracy over the cached conditions (15 corruptions × 5 severities) on a fixed test "
           "subset: " + ", ".join(f"{c} conditions × {n} images" for c, n in sorted(n_corr)) + ".", "",
           "| 데이터셋 | 방법 | Clean | WGA | Corruption |", "|---|---|---|---|---|", *rows, ""]
    if notes:
        md += ["## Notes", "", *[f"- {n}" for n in notes], ""]
    md += ["## 판정", "",
           "규칙: CUB=clean top-1, Waterbirds=WGA. 통과 = (KD−CE) 3-seed 평균 ≥ 1.0%p 그리고 같은 seed 차이 3개 모두 > 0. "
           "clean 실패 시 corruption 정확도로 같은 규칙 → OOD 통과.", "", *verdict_lines, ""]
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.results_dir) / "stage0_summary.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
