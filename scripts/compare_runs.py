"""Paired comparison of finished runs (eval.json) at one keep ratio: clean / WGA / corruption, per seed and vs a reference.

    uv run python scripts/compare_runs.py --dataset waterbirds --keep 0.15
    uv run python scripts/compare_runs.py --dataset cub --keep 0.15 --ref rollout --runs maskedkd tam tam_r tam_r_g0 tam_r_g0.5
"""
import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402


def load(root, ds, d, seeds):
    out = {}
    for s in seeds:
        p = root / ds / d / f"seed{s}" / "eval.json"
        if p.exists():
            out[s] = C.load_json(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--ref", default="rollout")
    ap.add_argument("--runs", nargs="+", default=["maskedkd", "rollout", "tam", "tam_r", "tam_r_g0", "tam_r_g0.5", "tam_g0", "tam_r_var_g0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "outputs"))
    a = ap.parse_args()
    root = Path(a.output_root)
    dirs = {"kd": "kd", **{r: f"{r}_k{a.keep:g}" for r in a.runs}}
    runs = {r: load(root, a.dataset, d, a.seeds) for r, d in dirs.items()}
    ref = runs.get(a.ref, {})
    metrics = [("clean_acc", "clean"), ("wga", "WGA"), ("corruption_acc", "corruption")]
    metrics = [(k, n) for k, n in metrics if any(k in e for v in runs.values() for e in v.values())]
    print(f"{a.dataset}, keep {a.keep:g}; per-seed values, mean, and paired difference vs {a.ref} (same seed)\n")
    for key, name in metrics:
        print(f"## {name}")
        print(f"{'run':<14}" + "".join(f"{'s' + str(s):>8}" for s in a.seeds) + f"{'mean':>9}{'vs ' + a.ref:>14}  per-seed diff")
        for r, v in runs.items():
            vals = [v[s][key] for s in a.seeds if s in v and key in v[s]]
            row = "".join(f"{v[s][key]:8.2f}" if s in v and key in v[s] else f"{'–':>8}" for s in a.seeds)
            mean = f"{statistics.mean(vals):9.2f}" if vals else f"{'–':>9}"
            diffs = {s: v[s][key] - ref[s][key] for s in a.seeds if s in v and s in ref and key in v[s] and key in ref[s]}
            d = f"{statistics.mean(diffs.values()):+14.2f}" if diffs and r != a.ref else f"{'':>14}"
            per = ", ".join(f"s{s} {x:+.2f}" for s, x in diffs.items()) if r != a.ref else ""
            print(f"{r:<14}{row}{mean}{d}  {per}")
        print()


if __name__ == "__main__":
    main()
