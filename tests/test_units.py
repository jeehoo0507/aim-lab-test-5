"""Fast unit checks (CPU, a few seconds). Run: DATA_ROOT=... uv run python tests/test_units.py"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402
from stage0.datasets import Stage0Dataset, build_train_transform  # noqa: E402
from stage0.engine import make_loader  # noqa: E402
from stage0.losses import HardCE, Stage0Loss, soft_kd_loss  # noqa: E402
from stage0.models import create_teacher, teacher_forward  # noqa: E402


def expect_raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


def test_teacher_forward():
    t = create_teacher(10, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(t(x), teacher_forward(t, x))
    t.train()
    expect_raises(lambda: teacher_forward(t, x))


def test_kd_loss_matches_maskedkd():
    torch.manual_seed(0)
    s, t = torch.randn(8, 5), torch.randn(8, 5)
    for tau in (1.0, 4.0):
        ref = torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(s / tau, 1), F.softmax(t / tau, 1)) * tau * tau
        assert torch.allclose(soft_kd_loss(s, t, tau), ref)
    y = F.one_hot(torch.randint(0, 5, (8,)), 5).float()
    total, ce, kd = Stage0Loss("kd")(s, y, t)
    assert torch.allclose(total, 0.5 * ce + 0.5 * kd)
    total, ce, kd = Stage0Loss("kd", alpha=1.0)(s, y, t)
    assert torch.allclose(total, kd)


def test_recipe_asserts():
    C.assert_recipe("teacher", 0.0, 0.0, 0.0)
    expect_raises(lambda: C.assert_recipe("teacher", 0.1, 0.0, 0.0))
    expect_raises(lambda: C.assert_recipe("teacher", 0.0, 0.8, 1.0))
    C.assert_recipe("ce", 0.1, 0.8, 1.0)
    C.assert_recipe("kd", 0.1, 0.8, 1.0)
    expect_raises(lambda: C.assert_recipe("kd", 0.0, 0.8, 1.0))
    expect_raises(lambda: HardCE(0.1))
    expect_raises(lambda: Stage0Loss("teacher")(torch.randn(4, 3), torch.rand(4, 3)))  # soft targets rejected
    assert C.REPEATED_AUG is False


def test_augmentation_independent_of_workers(data_root):
    ds = Stage0Dataset(data_root, "cub", "train", build_train_transform(), limit=24)

    def batches(seed, nw):
        ds.set_aug_seed(C.epoch_seed(seed, 0, 3))
        bs = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(
            ds, generator=torch.Generator().manual_seed(C.epoch_seed(seed, 0, 1))), 8, drop_last=True)
        return torch.cat([x for x, _, _ in make_loader(ds, None, nw, batch_sampler=bs,
                                                       generator=torch.Generator().manual_seed(1))])
    a = batches(0, 0)
    assert torch.equal(a, batches(0, 2)), "augmentation must not depend on num_workers"
    assert not torch.equal(a, batches(1, 0)), "different seeds must give different batches"


def test_container_limits():
    """cpu_count / mem_limit_gb follow the cgroup limits, not the host totals."""
    real_read = C._read
    host_cpus = len(os.sched_getaffinity(0))
    meminfo = ["MemTotal:", str(64 * 2**20), "kB"]  # 64 GiB host
    cases = [
        # (files, expected cpus, expected mem GiB)
        ({"/sys/fs/cgroup/cpu.max": ["max", "100000"]}, host_cpus, 64.0),
        ({"/sys/fs/cgroup/cpu.max": ["150000", "100000"], "/sys/fs/cgroup/memory.max": [str(16 * 2**30)]},
         min(host_cpus, 1), 16.0),
        ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": ["-1"], "/sys/fs/cgroup/cpu/cpu.cfs_period_us": ["100000"],
          "/sys/fs/cgroup/memory/memory.limit_in_bytes": ["9223372036854771712"]}, host_cpus, 64.0),
        ({"/sys/fs/cgroup/memory.max": ["max"]}, host_cpus, 64.0),
    ]
    try:
        for files, want_cpu, want_mem in cases:
            files = {**files, "/proc/meminfo": meminfo}
            C._read = lambda p, files=files: files.get(str(p))
            assert C.cpu_count() == want_cpu, (files, C.cpu_count())
            assert abs(C.mem_limit_gb() - want_mem) < 1e-6, (files, C.mem_limit_gb())
        C._read = lambda p: {"/sys/fs/cgroup/cpu.max": ["400000", "100000"]}.get(str(p))
        assert C.cpu_count() == min(host_cpus, 4)
    finally:
        C._read = real_read
    os.environ["STAGE0_CPUS"], os.environ["STAGE0_MEM_GB"] = "3", "12"
    try:
        assert C.cpu_count() == 3 and C.mem_limit_gb() == 12.0
    finally:
        del os.environ["STAGE0_CPUS"], os.environ["STAGE0_MEM_GB"]


if __name__ == "__main__":
    test_teacher_forward()
    test_kd_loss_matches_maskedkd()
    test_recipe_asserts()
    test_container_limits()
    dr = os.environ.get("DATA_ROOT")
    if dr and (Path(dr) / "cub" / "stage0_split.json").exists():
        test_augmentation_independent_of_workers(dr)
    else:
        print("skip augmentation test (DATA_ROOT not prepared)")
    print("unit tests passed")
