"""lr selection (3-1) from finished lr-sweep runs, then promotion of the winning seed-0 run.

    uv run python -m stage0.select_lr --dataset cub --which teacher
    uv run python -m stage0.select_lr --dataset cub --which student

Sweep runs live in outputs/{dataset}/lrsel/{teacher|ce}_lr{lr}/seed0/.
  teacher: highest best-epoch val accuracy over lr in {5e-5, 1e-4}  (the teacher checkpoint is val-best)
  student: highest final-epoch val accuracy of CE over lr in {5e-5, 1e-4, 3e-4}  (students are
           evaluated at the last epoch); the chosen lr is used for both CE and KD.
  ties -> smaller lr.
The winning sweep run has exactly the config of the seed-0 main run, so it is promoted (hard links)
to outputs/{dataset}/{teacher|ce}/seed0/ instead of being trained a second time.
Writes outputs/{dataset}/lr_selection.json and regenerates results/lr_selection.md.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

from stage0 import common as C

WHICH = {"teacher": ("teacher", C.TEACHER_LRS, "best_val_acc"),
         "student": ("ce", C.STUDENT_LRS, "final_val_acc")}
PROMOTE_FILES = ("config.json", "log.csv", "last.pt", "best.pt", "status.json", "train.log", "DONE")


def select(out_root, dataset, which):
    mode, lrs, metric = WHICH[which]
    scores = {}
    for lr in lrs:
        rd = C.run_dir(out_root, dataset, mode, 0, subdir=C.lrsel_subdir(mode, lr))
        if not (rd / "DONE").exists():
            sys.exit(f"[error] lr sweep run {rd} not finished")
        scores[lr] = C.load_json(rd / "DONE")[metric]
    best = max(scores.values())
    lr = min(l for l, s in scores.items() if s == best)
    return {"lr": lr, "metric": metric, "scores": {f"{l:g}": s for l, s in scores.items()},
            "source": str(C.run_dir(out_root, dataset, mode, 0, subdir=C.lrsel_subdir(mode, lr)))}


def promote(src, dst):
    src, dst = Path(src), Path(dst)
    src_cfg = C.load_json(src / "config.json")
    if dst.exists() and any(dst.iterdir()):
        if (dst / "DONE").exists() and C.load_json(dst / "config.json").get("run_uid") == src_cfg["run_uid"]:
            return "already promoted"
        sys.exit(f"[error] {dst} already exists and is not the promoted run {src}. Remove it or rerun selection.")
    dst.mkdir(parents=True, exist_ok=True)
    for name in PROMOTE_FILES:
        if (src / name).exists():
            try:
                os.link(src / name, dst / name)
            except OSError:
                shutil.copy2(src / name, dst / name)
    C.save_json(dst / "PROMOTED", {"from": str(src)})
    return "promoted"


def write_markdown(out_root, results_dir):
    lines = ["# lr selection (seed 0)", "",
             "teacher: best-epoch val acc, lr ∈ {5e-5, 1e-4}.  student: CE final-epoch val acc, "
             "lr ∈ {5e-5, 1e-4, 3e-4}; the chosen student lr is used for CE and KD.  ties → smaller lr.", "",
             "| dataset | run | " + " | ".join(f"lr={l:g}" for l in C.STUDENT_LRS) + " | chosen |",
             "|---|---|" + "---|" * len(C.STUDENT_LRS) + "---|"]
    for ds in C.DATASETS:
        p = Path(out_root) / ds / "lr_selection.json"
        if not p.exists():
            continue
        sel = C.load_json(p)
        for which in ("teacher", "student"):
            if which not in sel:
                continue
            s = sel[which]
            cells = [f"{s['scores'][f'{l:g}']:.2f}" if f"{l:g}" in s["scores"] else "–" for l in C.STUDENT_LRS]
            lines.append(f"| {ds} | {which} | " + " | ".join(cells) + f" | **{s['lr']:g}** |")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "lr_selection.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--which", required=True, choices=list(WHICH))
    ap.add_argument("--output-root")
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    args = ap.parse_args(argv)
    out_root = C.output_root(args.output_root)

    sel = select(out_root, args.dataset, args.which)
    path = out_root / args.dataset / "lr_selection.json"
    allsel = C.load_json(path) if path.exists() else {}
    if args.which in allsel and allsel[args.which]["lr"] != sel["lr"]:
        sys.exit(f"[error] {path} already records {args.which} lr={allsel[args.which]['lr']} "
                 f"but the sweep now picks {sel['lr']}")
    allsel[args.which] = sel
    C.save_json(path, allsel)
    mode = WHICH[args.which][0]
    status = promote(sel["source"], C.run_dir(out_root, args.dataset, mode, 0))
    write_markdown(out_root, args.results_dir)
    print(f"{args.dataset} {args.which}: lr={sel['lr']:g}  scores={sel['scores']}  ({status})")


if __name__ == "__main__":
    main()
