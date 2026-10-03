"""One-table progress view of a run_stage0.sh launch (called by status.sh).

States: PENDING(대기) RUNNING(실행 중) DONE(완료) FAILED(실패) BLOCKED(의존 run 실패로 보류)
"""
import argparse
import os
import time
from pathlib import Path

from stage0 import common as C


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root")
    ap.add_argument("--all", action="store_true", help="also list finished jobs")
    args = ap.parse_args(argv)
    out_root = C.output_root(args.output_root)
    sp = out_root / "_scheduler" / "state.json"
    if not sp.exists():
        print(f"no scheduler state at {sp} (run_stage0.sh not started yet?)")
        return
    st = C.load_json(sp)
    alive = pid_alive(st["pid"])
    now = time.time()
    counts = {}
    rows = []
    for j in st["jobs"]:
        state = j["state"]
        if state == "running" and not alive:
            state = "stale"
        counts[state] = counts.get(state, 0) + 1
        if state == "done" and not args.all:
            continue
        epoch = val = ""
        if j["run_dir"] and j["kind"] in ("gpu", "gpu_exclusive") and "/eval/" not in j["name"] \
                and "diagnose" not in j["name"]:
            sj = Path(j["run_dir"]) / "status.json"
            if sj.exists():
                s = C.load_json(sj)
                if "epoch" in s:
                    epoch = f"{s['epoch']}/{s['epochs']}"
                    val = f"{s['val_acc']:.2f}" if s.get("val_acc") is not None else ""
        elapsed = ""
        if j["started"]:
            elapsed = hms((j["ended"] or now) - j["started"]) if state != "pending" else ""
        gpu = j["gpu"] if j["gpu"] is not None else ("cpu" if j["kind"] == "cpu" else "")
        rows.append((j["name"], state.upper(), gpu if state in ("running", "stale") else "", epoch, val, elapsed,
                     (j["log"] or "") if state in ("failed", "stale") else ""))
    hdr = ("JOB", "STATE", "GPU", "EPOCH", "VAL", "ELAPSED", "LOG (failed)")
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    print(f"scheduler pid {st['pid']} {'running' if alive else 'NOT running'}; "
          f"updated {hms(now - st['updated'])} ago; GPUs {st['gpus']} x {st['slots']} slot(s)")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    order = {"RUNNING": 0, "STALE": 0, "FAILED": 1, "BLOCKED": 2, "PENDING": 3, "DONE": 4}
    for r in sorted(rows, key=lambda r: order.get(r[1], 9)):
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print("totals: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + ("" if args.all else "   (use --all to list finished jobs)"))


if __name__ == "__main__":
    main()
