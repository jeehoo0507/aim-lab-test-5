"""Surgical fine-tuning summary: per train_blocks, WGA / clean of KD vs MaskedKD (vs rollout), paired by seed.
Full-network rows ("all") come from the main runs (kd, maskedkd_k{keep}, rollout_k{keep}).

    uv run python -m stage0.summarize_surgical          # -> $RESULTS_DIR/why/surgical.md
Pre-registered reading: if MaskedKD − KD WGA at train_blocks=head is <= -3 %p on average with every seed negative,
the collapse needs no feature change -> it is carried by the masked teacher's target.
"""
import argparse
import os
import statistics
from pathlib import Path

from stage0 import common as C
from stage0.summarize_wga import load_runs, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--blocks", nargs="+", default=["head", "last2", "first4"])
    ap.add_argument("--crits", nargs="+", default=["maskedkd", "rollout"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--output-root")
    a = ap.parse_args()
    root, ds = C.output_root(a.output_root), "waterbirds"
    out = Path(os.environ.get("RESULTS_DIR", "results")) / "why"
    out.mkdir(parents=True, exist_ok=True)
    k = f"{a.keep:g}"
    md = [f"# Surgical fine-tuning: 일부 층만 학습해도 MaskedKD (keep {k})의 WGA가 무너지는가", "",
          "같은 seed끼리 비교 (같은 배치·mixup). 값은 seed 평균. all = 전체 학습 (본 실험 run).", "",
          "| 학습 범위 | KD clean | KD WGA | " + " | ".join(f"{c} clean | {c} WGA | {c}−KD WGA (seed별)" for c in a.crits) + " |",
          "|---|---|---|" + "---|---|---|" * len(a.crits)]
    verdict = []
    for tb in ["all"] + a.blocks:
        suf = "" if tb == "all" else f"_tb-{tb}"
        kd = load_runs(root, ds, "kd" + suf, a.seeds)
        cells = [f"{statistics.mean(e['clean_acc'] for e in kd.values()):.2f}" if kd else "–",
                 f"{statistics.mean(e['wga'] for e in kd.values()):.2f}" if kd else "–"]
        for c in a.crits:
            r = load_runs(root, ds, f"{c}_k{k}{suf}", a.seeds)
            d = paired(r, kd, "wga")
            cells += [f"{statistics.mean(e['clean_acc'] for e in r.values()):.2f}" if r else "–",
                      f"{statistics.mean(e['wga'] for e in r.values()):.2f}" if r else "–",
                      (f"{statistics.mean(d.values()):+.2f} (" + ", ".join(f"{v:+.1f}" for v in d.values()) + ")")
                      if d else "–"]
            if c == "maskedkd" and tb == "head" and d:
                hit = statistics.mean(d.values()) <= -3.0 and all(v < 0 for v in d.values())
                verdict.append(f"- head만 학습 (특징 고정): MaskedKD − KD WGA {statistics.mean(d.values()):+.2f}%p → "
                               + ("**붕괴 유지 → 원인은 teacher target 자체 (특징 변화 불필요)**" if hit
                                  else "붕괴 약함/없음 → student 특징 학습이 필요한 현상"))
        md.append(f"| {tb} | " + " | ".join(cells) + " |")
    md += ["", "## 판정 (사전 고정)", ""] + (verdict or ["- head run 미완료"])
    (out / "surgical.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
