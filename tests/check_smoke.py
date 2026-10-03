"""Assertions for smoke_test.sh (pipeline outputs, parallel isolation, resume)."""
import csv
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def pipeline(out_root):
    out_root = Path(out_root)
    print("[pipeline outputs]")
    runs = []
    for ds in C.DATASETS:
        for mode in C.MODES:
            seeds = (0,) if mode == "teacher" else (0, 1, 2)
            for s in seeds:
                rd = C.run_dir(out_root, ds, mode, s)
                runs.append(rd)
                cfg = C.load_json(rd / "config.json") if (rd / "config.json").exists() else {}
                check((rd / "DONE").exists() and not (rd / "FAILED").exists(), f"{rd.relative_to(out_root)} DONE")
                check(cfg.get("dataset") == ds and cfg.get("mode") == mode and cfg.get("seed") == s,
                      f"{rd.relative_to(out_root)} config matches its path")
                rows = list(csv.DictReader(open(rd / "log.csv"))) if (rd / "log.csv").exists() else []
                check(len(rows) == cfg.get("epochs", -1) and all(r["val_acc"] for r in rows),
                      f"{rd.relative_to(out_root)} log.csv has one row per epoch")
                if mode == "kd":
                    check(all(r["train_kd"] for r in rows), f"{rd.relative_to(out_root)} logs the KD term")
                    check(cfg.get("teacher_ckpt") == str(C.teacher_dir(out_root, ds) / "best.pt"),
                          f"{rd.relative_to(out_root)} used the {ds} teacher")
                check(cfg.get("smoothing") == C.RECIPE[mode]["smoothing"] and cfg.get("mixup") == C.RECIPE[mode]["mixup"],
                      f"{rd.relative_to(out_root)} recipe (LS={cfg.get('smoothing')}, mixup={cfg.get('mixup')})")
                ev = rd / "eval.json"
                check(ev.exists(), f"{rd.relative_to(out_root)} eval.json")
                if ev.exists():
                    e = C.load_json(ev)
                    check(e["corruption_n_conditions"] == 75, f"{rd.relative_to(out_root)} 75 corruption conditions")
                    if ds == "waterbirds":
                        check(len(e["group_acc"]) == 4 and e["wga"] == min(e["group_acc"]),
                              f"{rd.relative_to(out_root)} 4 group accs + WGA")
                    ck = "best.pt" if mode == "teacher" else "last.pt"
                    check(e["checkpoint"] == ck, f"{rd.relative_to(out_root)} evaluated {ck}")
                # the run's own log only ever mentions its own command
                log = (rd / "train.log").read_text()
                heads = [line for line in log.splitlines() if line.startswith("===== ")]
                ok = all(f"--mode {mode}" in h and f"--dataset {ds}" in h and f"--seed {s}" in h for h in heads)
                check(heads and ok, f"{rd.relative_to(out_root)} train.log contains only its own run(s)")
        d = C.teacher_dir(out_root, ds) / "diagnosis.json"
        check(d.exists(), f"{ds} teacher diagnosis.json")
    uids = {}
    for rd in runs:
        if (rd / "config.json").exists():
            uids.setdefault(C.load_json(rd / "config.json")["run_uid"], []).append(rd)
    promoted = {C.run_dir(out_root, ds, m, 0) for ds in C.DATASETS for m in ("teacher", "ce")}
    dup = [v for v in uids.values() if len(v) > 1]
    check(not dup, f"every main run dir holds its own run (duplicates: {dup})")
    for rd in promoted:
        check((rd / "PROMOTED").exists(), f"{rd.relative_to(out_root)} promoted from the lr sweep")


def parallel(out_root, min_overlap):
    print("[parallel execution]")
    st = C.load_json(Path(out_root) / "_scheduler" / "state.json")
    t_cost = st.get("teacher_slots", st["slots"])
    jobs = [j for j in st["jobs"] if j["kind"] in ("gpu", "gpu_exclusive") and j["started"] and j["ended"]]
    max_small = max_load = max_teach = 0
    for g in {j["gpu"] for j in jobs}:
        ev = sorted(itertools.chain.from_iterable(
            ((j["started"], 1, j["kind"]), (j["ended"], -1, j["kind"])) for j in jobs if j["gpu"] == g),
            key=lambda e: (e[0], e[1]))  # ends before starts at equal timestamps
        small = teach = 0
        for _, d, kind in ev:
            if kind == "gpu":
                small += d
            else:
                teach += d
            max_small, max_teach = max(max_small, small), max(max_teach, teach)
            max_load = max(max_load, small + teach * t_cost)
    check(max_small >= min_overlap, f"max concurrent student/eval jobs on one GPU = {max_small} (want >= {min_overlap})")
    check(max_load <= st["slots"], f"slot load never above RUNS_PER_GPU={st['slots']} (max {max_load}, teacher={t_cost})")
    check(max_teach <= 1, "at most one teacher fine-tune per GPU at a time")
    check(all(j["state"] == "done" for j in st["jobs"]), "all scheduler jobs done")


def resume(resumed_dir, reference_dir, strict):
    print("[resume]")
    r, ref = Path(resumed_dir), Path(reference_dir)
    rows = list(csv.DictReader(open(r / "log.csv")))
    check([int(x["epoch"]) for x in rows] == list(range(len(rows))) and (r / "DONE").exists(),
          f"resumed run finished with epochs {[x['epoch'] for x in rows]} exactly once each")
    check("resuming from" in (r / "train.log").read_text(), "train.log shows the run resumed from last.pt")
    a = torch.load(r / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(ref / "last.pt", map_location="cpu", weights_only=False)["model"]
    diff = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
    ra = list(csv.DictReader(open(ref / "log.csv")))
    print(f"  info max |w_resumed - w_uninterrupted| = {diff:.3g}; val acc resumed "
          f"{[x['val_acc'] for x in rows]} vs uninterrupted {[x['val_acc'] for x in ra]}")
    if strict:
        check(diff < 1e-4, "resumed run reproduces the uninterrupted run (CPU, deterministic)")
    else:
        print("  info GPU kernels are not bit-deterministic; weight equality is reported, not enforced")


if __name__ == "__main__":
    what = sys.argv[1]
    {"pipeline": lambda: pipeline(sys.argv[2]),
     "parallel": lambda: parallel(sys.argv[2], int(sys.argv[3])),
     "resume": lambda: resume(sys.argv[2], sys.argv[3], sys.argv[4] == "strict")}[what]()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        sys.exit(1)
