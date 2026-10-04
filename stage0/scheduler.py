"""Dependency-aware parallel runner behind run_stage0.sh.

Dependencies (only these are ordered):
    teacher lr sweep -> select teacher lr (promotes seed-0 teacher) -> teacher diagnosis, teacher eval, KD runs
    CE lr sweep      -> select student lr (promotes CE seed 0)      -> CE seeds 1-2, KD seeds 0-2
    corruption cache (CPU)                                          -> every eval
    every train run  -> its eval;   everything -> summarize
Everything else starts immediately (both datasets' teacher and CE sweeps, the corruption cache).

GPU placement: a queue over the GPUs listed in GPUS. Each GPU has RUNS_PER_GPU slots. Student runs,
diagnosis and eval take one slot. A DeiT-B teacher fine-tune is limited to one per GPU and takes
TEACHER_SLOTS slots (preflight.json: teacher peak memory / student peak memory; without preflight the
whole GPU). When a teacher is waiting, the least-loaded GPU stops taking new small jobs until there
is room (no starvation).
Finished work is detected from files (DONE, eval.json, diagnosis.json, cache DONE), so rerunning the
same command after an interruption only runs what is left; failed runs are retried (train.py resumes
from last.pt). A failing job never stops unrelated jobs; its dependents are marked blocked.
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from stage0 import common as C
from stage0.attribution import cache_dir as attribution_cache_dir
from stage0.masking import STAGE2_CRITERIA, TAM_CACHE_CRITERIA, TRAIN_CRITERIA

PY = [sys.executable, "-m"]


@dataclass
class Job:
    name: str
    kind: str                    # gpu | gpu_exclusive | cpu | local | final
    cmd: object                  # list[str] or callable -> list[str]
    done: object                 # callable -> bool
    deps: list = field(default_factory=list)
    priority: int = 50
    run_dir: Path = None
    marker: str = None           # failure marker file written into run_dir
    state: str = "pending"       # pending | running | done | failed | blocked
    gpu: str = None
    proc: object = None
    started: float = None
    ended: float = None
    rc: int = None
    log: Path = None


def lr_of(out_root, ds, which):
    return C.load_json(Path(out_root) / ds / "lr_selection.json")[which]["lr"]


def build_stage2_jobs(a, out_root, data_root):
    """Stage 2: maskedkd-mode student runs (criterion x keep x seed) + their eval + stage-2 summary.
    Needs the finished Stage 0 teacher and student lr selection (read only); refuses otherwise."""
    ds_list = a.datasets
    tr_extra, ev_extra = shlex.split(a.train_args), shlex.split(a.eval_args)
    co_extra = shlex.split(a.corruption_args)
    cache_root = C.corruption_root(data_root)
    jobs = []
    for ds in ds_list:
        tdir = C.teacher_dir(out_root, ds)
        sel = Path(out_root) / ds / "lr_selection.json"
        if not (tdir / "DONE").exists():
            sys.exit(f"[error] Stage 2 needs the finished Stage 0 teacher {tdir} (no DONE). Run Stage 0 first.")
        if not sel.exists() or "student" not in C.load_json(sel):
            sys.exit(f"[error] Stage 2 needs the Stage 0 student lr selection {sel}. Run Stage 0 first.")
        lr = C.load_json(sel)["student"]["lr"]
        jobs.append(Job(f"{ds}/corruptions", "cpu",
                        PY + ["stage0.make_corruptions", "--datasets", ds, "--workers", str(a.corruption_workers)]
                        + co_extra, lambda ds=ds: (cache_root / ds / "DONE").exists(), [], 10))
        attr_job = f"{ds}/attribution_cache"
        if any(c in TAM_CACHE_CRITERIA for c in a.criteria):   # tam / tam_var read the attribution cache
            adir = attribution_cache_dir(out_root, ds)
            tuid = C.load_json(tdir / "config.json")["run_uid"]
            jobs.append(Job(attr_job, "gpu", PY + ["stage0.make_attribution_cache", "--dataset", ds]
                            + shlex.split(a.attribution_args),
                            lambda adir=adir, tuid=tuid: (adir / "DONE").exists()
                            and C.load_json(adir / "manifest.json")["teacher_run_uid"] == tuid, [], 95))
        for crit in a.criteria:
            for keep in a.keeps:
                md = C.mask_dirname(crit, keep)
                for seed in a.seeds:
                    rd = C.run_dir(out_root, ds, C.MASK_MODE, seed, subdir=md)
                    name = f"{ds}/{md}/seed{seed}"
                    jobs.append(Job(name, "gpu",
                                    PY + ["stage0.train", "--mode", C.MASK_MODE, "--mask-criterion", crit,
                                          "--keep", f"{keep:g}", "--dataset", ds, "--seed", str(seed),
                                          "--lr", f"{lr:g}"] + tr_extra,
                                    lambda rd=rd: (rd / "DONE").exists(),
                                    [attr_job] if crit in TAM_CACHE_CRITERIA else [],
                                    60 if crit == "tam_oracle" else 70,   # slowest run starts last
                                    rd, "FAILED"))
                    jobs.append(Job(f"{ds}/eval/{md}/seed{seed}", "gpu",
                                    PY + ["stage0.evaluate", "--dataset", ds, "--mode", md, "--seed", str(seed)]
                                    + ev_extra, lambda rd=rd: (rd / "eval.json").exists(),
                                    [name, f"{ds}/corruptions"], 30, rd, "FAILED.eval"))
    summary = Path(os.environ.get("RESULTS_DIR", "results")) / "stage2_summary.md"
    # default keeps + requested ones, so a later partial launch (e.g. --keeps 0.15) does not shrink the table
    summary_keeps = sorted(set(C.MASK_KEEPS) | set(a.keeps), reverse=True)
    jobs.append(Job("summarize_stage2", "final", PY + ["stage0.summarize_stage2", "--datasets", *ds_list,
                                                        "--keeps", *[f"{k:g}" for k in summary_keeps]],
                    lambda: summary.exists(), [], 0))
    return jobs


def build_jobs(a, out_root, data_root):
    jobs = []
    tr_extra, ev_extra = shlex.split(a.train_args), shlex.split(a.eval_args)
    dg_extra, co_extra = shlex.split(a.diag_args), shlex.split(a.corruption_args)
    cache_root = C.corruption_root(data_root)

    def train_job(name, ds, mode, seed, lr_fn, deps, prio, subdir=None, alpha=C.KD_ALPHA):
        rd = C.run_dir(out_root, ds, mode, seed, alpha, subdir)

        def cmd():
            c = PY + ["stage0.train", "--mode", mode, "--dataset", ds, "--seed", str(seed),
                      "--lr", f"{lr_fn():g}"] + tr_extra
            if subdir:
                c += ["--subdir", subdir]
            if mode == "kd" and alpha != C.KD_ALPHA:
                c += ["--alpha", str(alpha)]
            return c
        jobs.append(Job(name, "gpu_exclusive" if mode == "teacher" else "gpu", cmd,
                        lambda rd=rd: (rd / "DONE").exists(), deps, prio, rd, "FAILED"))

    def eval_job(ds, mode_dir, seed, deps):
        rd = Path(out_root) / ds / mode_dir / f"seed{seed}"
        jobs.append(Job(f"{ds}/eval/{mode_dir}/seed{seed}", "gpu",
                        PY + ["stage0.evaluate", "--dataset", ds, "--mode", mode_dir, "--seed", str(seed)] + ev_extra,
                        lambda rd=rd: (rd / "eval.json").exists(), deps + [f"{ds}/corruptions"], 30, rd,
                        "FAILED.eval"))

    for ds in a.datasets:
        jobs.append(Job(f"{ds}/corruptions", "cpu",
                        PY + ["stage0.make_corruptions", "--datasets", ds, "--workers", str(a.corruption_workers)]
                        + co_extra,
                        lambda ds=ds: (cache_root / ds / "DONE").exists(), [], 10))
        t_sweep = []
        for lr in C.TEACHER_LRS:
            n = f"{ds}/lrsel/teacher_lr{lr:g}"
            train_job(n, ds, "teacher", 0, lambda lr=lr: lr, [], 90, subdir=C.lrsel_subdir("teacher", lr))
            t_sweep.append(n)
        s_sweep = []
        for lr in C.STUDENT_LRS:
            n = f"{ds}/lrsel/ce_lr{lr:g}"
            train_job(n, ds, "ce", 0, lambda lr=lr: lr, [], 80, subdir=C.lrsel_subdir("ce", lr))
            s_sweep.append(n)
        sel_t, sel_s = f"{ds}/select_lr/teacher", f"{ds}/select_lr/student"
        for name, which, deps in ((sel_t, "teacher", t_sweep), (sel_s, "student", s_sweep)):
            mode = "teacher" if which == "teacher" else "ce"
            jobs.append(Job(name, "local", PY + ["stage0.select_lr", "--dataset", ds, "--which", which],
                            lambda ds=ds, which=which, mode=mode: (
                                (Path(out_root) / ds / "lr_selection.json").exists()
                                and which in C.load_json(Path(out_root) / ds / "lr_selection.json")
                                and (C.run_dir(out_root, ds, mode, 0) / "DONE").exists()),
                            deps, 100))
        t_lr = lambda ds=ds: lr_of(out_root, ds, "teacher")  # noqa: E731
        s_lr = lambda ds=ds: lr_of(out_root, ds, "student")  # noqa: E731

        tdir = C.teacher_dir(out_root, ds)
        jobs.append(Job(f"{ds}/diagnose_teacher", "gpu",
                        PY + ["stage0.diagnose_teacher", "--dataset", ds] + dg_extra,
                        lambda tdir=tdir: (tdir / "diagnosis.json").exists(), [sel_t], 40, tdir,
                        "FAILED.diagnose"))
        eval_job(ds, "teacher", 0, [sel_t])
        eval_job(ds, "ce", 0, [sel_s])
        for s in (1, 2):
            train_job(f"{ds}/ce/seed{s}", ds, "ce", s, s_lr, [sel_s], 60)
            eval_job(ds, "ce", s, [f"{ds}/ce/seed{s}"])
        kd_alphas = [C.KD_ALPHA] + ([1.0] if a.fallback_alpha1 else [])
        for alpha in kd_alphas:
            md = C.mode_dirname("kd", alpha)
            for s in (0, 1, 2):
                train_job(f"{ds}/{md}/seed{s}", ds, "kd", s, s_lr, [sel_s, sel_t], 70, alpha=alpha)
                eval_job(ds, md, s, [f"{ds}/{md}/seed{s}"])
    summary = Path(os.environ.get("RESULTS_DIR", "results")) / "stage0_summary.md"
    jobs.append(Job("summarize", "final", PY + ["stage0.summarize"], lambda: summary.exists(), [], 0))
    return jobs


class Scheduler:
    def __init__(self, jobs, gpus, slots, teacher_slots, out_root, max_cpu_jobs, poll, sched_name="_scheduler"):
        self.jobs = {j.name: j for j in jobs}
        self.gpus, self.slots = gpus, slots
        self.teacher_slots = max(1, min(teacher_slots, slots))
        self.used = {g: 0 for g in gpus}
        self.teacher_on = {g: 0 for g in gpus}
        self.reserved = {}
        self.out_root = Path(out_root)
        self.sched_dir = self.out_root / sched_name   # Stage 2 uses its own: never clobbers a running Stage 0
        (self.sched_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.max_cpu_jobs, self.poll = max_cpu_jobs, poll
        self.env_base = dict(os.environ, STAGE0_CONCURRENT_RUNS=str(len(gpus) * slots), PYTHONUNBUFFERED="1")
        prev = {}
        if (self.sched_dir / "state.json").exists():
            prev = {x["name"]: x for x in C.load_json(self.sched_dir / "state.json")["jobs"]}
        for j in jobs:
            if j.kind != "final" and j.done():  # the summary is always regenerated
                j.state = "done"
                p = prev.get(j.name, {})
                if p.get("state") == "done":   # keep timing history of earlier launches
                    j.gpu, j.started, j.ended, j.rc = p["gpu"], p["started"], p["ended"], p["rc"]
                    j.log = Path(p["log"]) if p.get("log") else None

    # ---------------------------------------------------------------- run ---
    def start(self, j, gpu=None):
        safe = j.name.replace("/", "__")
        j.log = self.sched_dir / "logs" / f"{safe}.log"
        try:
            cmd = j.cmd() if callable(j.cmd) else j.cmd
        except Exception as e:  # noqa: BLE001 - e.g. unreadable lr_selection.json: fail this job only
            with open(j.log, "a") as f:
                f.write(f"\n===== {time.strftime('%F %T')} could not build command: {e!r}\n")
            if gpu is not None:
                self.used[gpu] -= self.teacher_slots if j.kind == "gpu_exclusive" else 1
                if j.kind == "gpu_exclusive":
                    self.teacher_on[gpu] -= 1
            j.state, j.started, j.ended, j.rc = "failed", time.time(), time.time(), None
            print(f"[{time.strftime('%T')}] FAILED  {j.name} (could not build command: {e!r})", flush=True)
            return False
        env = dict(self.env_base)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
        f = open(j.log, "a")
        f.write(f"\n===== {time.strftime('%F %T')} gpu={gpu} {' '.join(cmd)}\n")
        f.flush()
        j.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        j.proc._logfile = f
        j.state, j.gpu, j.started, j.ended, j.rc = "running", gpu, time.time(), None, None
        print(f"[{time.strftime('%T')}] start   {j.name}" + (f"  (gpu {gpu})" if gpu is not None else ""),
              flush=True)
        return True

    def finish(self, j, rc):
        j.proc._logfile.close()
        j.rc, j.ended, j.proc = rc, time.time(), None
        if j.gpu is not None:
            self.used[j.gpu] -= self.teacher_slots if j.kind == "gpu_exclusive" else 1
            if j.kind == "gpu_exclusive":
                self.teacher_on[j.gpu] -= 1
        if rc == 0 and j.done():
            j.state = "done"
            if j.marker and j.run_dir is not None:
                (j.run_dir / j.marker).unlink(missing_ok=True)
            print(f"[{time.strftime('%T')}] done    {j.name}", flush=True)
            return
        j.state = "failed"
        if j.marker and j.run_dir is not None and j.run_dir.exists():
            marker = j.run_dir / j.marker
            # train.py writes FAILED with the traceback itself; only add one if it could not (e.g. killed)
            if not marker.exists() or marker.stat().st_mtime < j.started:
                tail = j.log.read_text(errors="replace").splitlines()[-40:] if j.log.exists() else []
                marker.write_text(f"{j.name} exited with code {rc}\nlog: {j.log}\n\n" + "\n".join(tail))
        print(f"[{time.strftime('%T')}] FAILED  {j.name} (rc={rc}); log: {j.log}", flush=True)

    def place_gpu(self, j):
        free = [g for g in self.gpus if self.reserved.get(g) in (None, j.name)]
        if j.kind == "gpu_exclusive":
            room = [g for g in free if self.teacher_on[g] == 0
                    and self.used[g] + self.teacher_slots <= self.slots]
            if room:
                g = min(room, key=lambda g: self.used[g])
                self.reserved = {k: v for k, v in self.reserved.items() if v != j.name}
                self.used[g] += self.teacher_slots
                self.teacher_on[g] += 1
                return g
            if j.name not in self.reserved.values():
                cands = [g for g in free if self.reserved.get(g) is None and self.teacher_on[g] == 0]
                if cands:
                    g = min(cands, key=lambda g: self.used[g])
                    self.reserved[g] = j.name
            return None
        cands = [g for g in free if self.reserved.get(g) is None and self.used[g] < self.slots]
        if not cands:
            return None
        g = min(cands, key=lambda g: self.used[g])
        self.used[g] += 1
        return g

    def step(self):
        for j in self.jobs.values():
            if j.state == "running" and j.proc.poll() is not None:
                self.finish(j, j.proc.returncode)
        changed = True
        while changed:
            changed = False
            for j in self.jobs.values():
                if j.state == "pending" and any(self.jobs[d].state in ("failed", "blocked") for d in j.deps):
                    j.state, changed = "blocked", True
                    print(f"[{time.strftime('%T')}] blocked {j.name} (dependency failed)", flush=True)
        others_active = any(j.state in ("pending", "running") for j in self.jobs.values() if j.kind != "final")
        ready = sorted((j for j in self.jobs.values() if j.state == "pending"
                        and all(self.jobs[d].state == "done" for d in j.deps)), key=lambda j: -j.priority)
        for j in ready:
            if j.kind == "final":
                if not others_active:
                    self.run_blocking(j)
            elif j.kind == "local":
                self.run_blocking(j)
            elif j.kind == "cpu":
                if sum(1 for x in self.jobs.values() if x.kind == "cpu" and x.state == "running") < self.max_cpu_jobs:
                    self.start(j)
            else:
                g = self.place_gpu(j)
                if g is not None:
                    self.start(j, g)
        self.write_state()

    def run_blocking(self, j):
        if self.start(j):
            self.finish(j, j.proc.wait())

    def write_state(self):
        C.save_json(self.sched_dir / "state.json", {
            "pid": os.getpid(), "updated": time.time(), "gpus": self.gpus, "slots": self.slots,
            "teacher_slots": self.teacher_slots,
            "jobs": [{"name": j.name, "kind": j.kind, "state": j.state, "gpu": j.gpu, "started": j.started,
                      "ended": j.ended, "rc": j.rc, "run_dir": str(j.run_dir) if j.run_dir else None,
                      "log": str(j.log) if j.log else None, "deps": j.deps} for j in self.jobs.values()]})

    def run(self):
        def stop(signum, _frame):
            print(f"\nsignal {signum}: stopping running jobs (they resume on the next launch)", flush=True)
            for j in self.jobs.values():
                if j.state == "running" and j.proc is not None:
                    j.proc.terminate()
            for j in self.jobs.values():
                if j.state == "running" and j.proc is not None:
                    try:
                        j.proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        j.proc.kill()
                    self.finish(j, -signum)
            self.write_state()
            sys.exit(130)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        n_done = sum(j.state == "done" for j in self.jobs.values())
        print(f"{len(self.jobs)} jobs, {n_done} already done. GPUs {self.gpus} x {self.slots} slot(s); "
              f"a teacher fine-tune uses {self.teacher_slots}.", flush=True)
        while any(j.state in ("pending", "running") for j in self.jobs.values()):
            self.step()
            if any(j.state in ("pending", "running") for j in self.jobs.values()):
                time.sleep(self.poll)
        self.write_state()
        bad = [j.name for j in self.jobs.values() if j.state in ("failed", "blocked")]
        print(f"\nfinished. failed/blocked: {bad or 'none'}")
        return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=os.environ.get("GPUS"),
                    help='comma-separated GPU ids, e.g. "0,1" (required; "cpu" = CPU-only test mode)')
    ap.add_argument("--runs-per-gpu", type=int, default=int(os.environ["RUNS_PER_GPU"])
                    if os.environ.get("RUNS_PER_GPU") else None)
    ap.add_argument("--teacher-slots", type=int, default=int(os.environ["TEACHER_SLOTS"])
                    if os.environ.get("TEACHER_SLOTS") else None,
                    help="slots a teacher fine-tune occupies (default: preflight.json, else the whole GPU)")
    ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS), choices=C.DATASETS)
    ap.add_argument("--corruption-workers", type=int,
                    default=int(os.environ.get("CORRUPTION_WORKERS", max(1, C.cpu_count() // 2))))
    ap.add_argument("--max-cpu-jobs", type=int, default=1)
    ap.add_argument("--fallback-alpha1", action="store_true", help="also run KD alpha=1.0 x 3 seeds (5절 fallback)")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--stage", type=int, choices=(0, 2), default=0,
                    help="0 = Stage 0 pipeline (default); 2 = MaskedKD / random baselines (needs Stage 0 done)")
    ap.add_argument("--criteria", nargs="+", default=list(STAGE2_CRITERIA), choices=TRAIN_CRITERIA,
                    help="[stage 2] token-selection criteria")
    ap.add_argument("--keeps", type=float, nargs="+", default=list(C.MASK_KEEPS), help="[stage 2] keep ratios")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="[stage 2] seeds")
    ap.add_argument("--train-args", default="", help="extra args for every train.py call (smoke test)")
    ap.add_argument("--eval-args", default="")
    ap.add_argument("--diag-args", default="")
    ap.add_argument("--corruption-args", default="")
    ap.add_argument("--attribution-args", default="", help="[stage 2, tam] extra make_attribution_cache args")
    ap.add_argument("--output-root")
    ap.add_argument("--data-root")
    a = ap.parse_args(argv)

    if not a.gpus:
        sys.exit('[error] set GPUS explicitly, e.g. GPUS="0,1" (all GPUs are never taken by default)')
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    out_root, data_root = C.output_root(a.output_root), C.data_root(a.data_root)
    os.environ["OUTPUT_ROOT"], os.environ["DATA_ROOT"] = str(out_root), str(data_root)
    slots = a.runs_per_gpu
    pf = C.preflight_settings(out_root) or {}
    if slots is None:
        slots = pf.get("suggested_runs_per_gpu")
        if slots is None:
            print("[warn] no RUNS_PER_GPU and no preflight.json -> 1 run per GPU. Run preflight.py first.")
            slots = 1
    slots = max(1, slots)
    teacher_slots = a.teacher_slots or pf.get("teacher_slot_cost") or slots
    if a.stage == 2:
        jobs, sched_name = build_stage2_jobs(a, out_root, data_root), "_scheduler_stage2"
    else:
        jobs, sched_name = build_jobs(a, out_root, data_root), "_scheduler"
    sys.exit(Scheduler(jobs, gpus, slots, teacher_slots, out_root, a.max_cpu_jobs, a.poll, sched_name).run())


if __name__ == "__main__":
    main()
