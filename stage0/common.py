"""Shared constants, paths and run-directory helpers for Stage 0.

The training recipe is fixed here (see docs/maskedkd_settings.md, "Stage 0 채택 설정").
Nothing in this file is meant to be changed per run; train.py asserts against it.
"""
import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

DATASETS = ("cub", "waterbirds")
MODES = ("teacher", "ce", "kd")
NUM_CLASSES = {"cub": 200, "waterbirds": 2}

TEACHER_ARCH = "deit_base_patch16_224"
STUDENT_ARCH = "deit_tiny_patch16_224"
IMG_SIZE = 224
RESIZE_SIZE = 256  # Resize(256) -> CenterCrop(224)  (MaskedKD eval_crop_ratio 0.875)

# ---------------------------------------------------------------- recipe ----
# Common to all runs
EPOCHS = 100
BATCH_SIZE = 128
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-6          # MaskedKD main.py:78
MIN_LR_RATIO = 0.01       # min_lr = base lr / 100 (team decision; MaskedKD uses 1e-5 absolute)
COOLDOWN_EPOCHS = 0       # team decision; MaskedKD default is 10
WEIGHT_DECAY = 0.05
OPT_EPS = 1e-8
DROP_PATH = 0.1           # MaskedKD main.py:51
INTERPOLATION = "bicubic"  # MaskedKD main.py:101 / datasets.py:104

# Train augmentation (teacher and students): MaskedKD transforms_imagenet_train
RRC_SCALE = (0.08, 1.0)
RRC_RATIO = (3.0 / 4.0, 4.0 / 3.0)
HFLIP = 0.5
RAND_AUGMENT = "rand-m9-mstd0.5-inc1"
REPROB = 0.25
REMODE = "pixel"
RECOUNT = 1
REPEATED_AUG = False      # team decision: off for every run

# Per-mode label smoothing / mixup
RECIPE = {
    # teacher fine-tune: RRC+flip+RA+erasing, NO label smoothing, NO mixup/cutmix
    "teacher": dict(smoothing=0.0, mixup=0.0, cutmix=0.0),
    # students (CE and KD identical): MaskedKD defaults
    "ce": dict(smoothing=0.1, mixup=0.8, cutmix=1.0),
    "kd": dict(smoothing=0.1, mixup=0.8, cutmix=1.0),
}
MIXUP_PROB = 1.0
MIXUP_SWITCH_PROB = 0.5
MIXUP_MODE = "batch"

# KD (MaskedKD losses.py)
KD_ALPHA = 0.5
KD_TAU = 1.0

# lr grids (3-1)
TEACHER_LRS = (5e-5, 1e-4)
STUDENT_LRS = (5e-5, 1e-4, 3e-4)

# Corruptions (4-3)
CORRUPTION_SUBSET = 1000
CORRUPTION_SEED = 0
SEVERITIES = (1, 2, 3, 4, 5)


def assert_recipe(mode, smoothing, mixup, cutmix):
    """Teacher: LS=0 and mixup/cutmix off (previous failure cause). Students: MaskedKD values."""
    if mode == "teacher":
        assert smoothing == 0.0, f"teacher fine-tune must use label_smoothing == 0.0, got {smoothing}"
        assert mixup == 0.0 and cutmix == 0.0, f"teacher fine-tune must not use mixup/cutmix ({mixup}, {cutmix})"
    else:
        want = RECIPE[mode]
        got = dict(smoothing=smoothing, mixup=mixup, cutmix=cutmix)
        assert got == want, f"{mode} must use the MaskedKD student recipe {want}, got {got}"


# ----------------------------------------------------------------- paths ----
def _env_path(value, env, what):
    v = value or os.environ.get(env)
    if not v:
        sys.exit(f"[error] {what} not set: pass --{env.lower().replace('_', '-')} or export {env}")
    return Path(v).expanduser().resolve()


def data_root(arg=None):
    return _env_path(arg, "DATA_ROOT", "DATA_ROOT")


def output_root(arg=None):
    return _env_path(arg, "OUTPUT_ROOT", "OUTPUT_ROOT")


def corruption_root(data_root_path):
    return Path(os.environ.get("CORRUPTION_ROOT", data_root_path / "corruptions")).expanduser().resolve()


def mode_dirname(mode, alpha=KD_ALPHA):
    if mode == "kd" and alpha != KD_ALPHA:
        return f"kd_alpha{alpha:g}"
    return mode


def run_dir(out_root, dataset, mode, seed, alpha=KD_ALPHA, subdir=None):
    """outputs/{dataset}/{mode}/seed{seed}/  (lr selection: outputs/{dataset}/lrsel/{mode}_lr{lr}/seed0/)."""
    return Path(out_root) / dataset / (subdir or mode_dirname(mode, alpha)) / f"seed{seed}"


def lrsel_subdir(mode, lr):
    return f"lrsel/{mode}_lr{lr:g}"


def teacher_dir(out_root, dataset):
    return run_dir(out_root, dataset, "teacher", 0)


# --------------------------------------------------------------- helpers ----
def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_seed(seed, epoch, stream=0):
    """Deterministic per-(seed, epoch, stream) seed, so a resumed run replays the same epoch."""
    return (seed * 1_000_003 + epoch * 7_919 + stream * 104_729) % (2**31 - 1)


def worker_init_fn(worker_id):
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)


def concurrent_runs():
    return max(1, int(os.environ.get("STAGE0_CONCURRENT_RUNS", "1")))


def _read(path):
    try:
        return Path(path).read_text().split()
    except OSError:
        return None


def cpu_count():
    """CPUs this process may actually use: min(affinity, cgroup CPU quota).

    os.cpu_count() reports the whole host inside a container (Kubernetes / Coder pods), which would
    start far more data-loader workers than the pod's CPU limit allows.
    $STAGE0_CPUS overrides the detection.
    """
    if os.environ.get("STAGE0_CPUS"):
        return max(1, int(os.environ["STAGE0_CPUS"]))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = None
    v2 = _read("/sys/fs/cgroup/cpu.max")                       # cgroup v2: "max 100000" | "800000 100000"
    if v2 and v2[0] != "max":
        quota = int(v2[0]) / int(v2[1])
    else:
        q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and int(q[0]) > 0:                          # cgroup v1: quota -1 = unlimited
            quota = int(q[0]) / int(p[0])
    if quota is not None:
        n = min(n, max(1, int(quota)))
    return max(1, n)


def mem_limit_gb():
    """RAM this process may use in GiB: min(cgroup memory limit, physical RAM). $STAGE0_MEM_GB overrides."""
    if os.environ.get("STAGE0_MEM_GB"):
        return float(os.environ["STAGE0_MEM_GB"])
    limits = []
    meminfo = _read("/proc/meminfo")
    if meminfo and "MemTotal:" in meminfo:
        limits.append(int(meminfo[meminfo.index("MemTotal:") + 1]) * 1024)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read(path)
        if v and v[0].isdigit():
            limits.append(int(v[0]))
    return min(limits) / 2**30 if limits else float("inf")


def default_num_workers():
    return max(1, min(8, cpu_count() // concurrent_runs()))


def set_cpu_threads():
    torch.set_num_threads(max(1, cpu_count() // concurrent_runs()))


def preflight_settings(out_root):
    p = Path(out_root) / "preflight.json"
    return load_json(p) if p.exists() else None


class Tee:
    """Duplicate a stream into a log file (append mode)."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


def tee_output(log_path):
    f = open(log_path, "a", buffering=1)
    f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} {' '.join(sys.argv)}\n")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    return f


class RunLock:
    """Exclusive lock on a run directory so two processes can never write to the same run."""

    def __init__(self, directory):
        self.path = Path(directory) / "RUNNING.lock"
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"[error] {self.path.parent} is in use by another process (lock held). Refusing to run.")
        self.f.seek(0)
        self.f.truncate()
        self.f.write(str(os.getpid()))
        self.f.flush()
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
