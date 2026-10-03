"""Server pre-check (6-2). Run before anything else:

    GPUS="0" DATA_ROOT=... OUTPUT_ROOT=... HF_HOME=... uv run python -m stage0.preflight

Checks CUDA / GPUs / arch compatibility, disk, write permissions, internet (or local copies of the
weights and datasets), and measures memory: one DeiT-B teacher training step and one KD student step
at batch 128, halving the micro-batch on OOM. The largest fitting micro-batch (gradient accumulation
keeps the effective batch at 128) and the suggested RUNS_PER_GPU go to $OUTPUT_ROOT/preflight.json,
which train.py and run_stage0.sh use as defaults. Exits non-zero with fixes if any check fails.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

from stage0 import common as C

MIN_DISK_GB = 30
HF_REPOS = {C.TEACHER_ARCH: "timm/deit_base_patch16_224.fb_in1k",
            C.STUDENT_ARCH: "timm/deit_tiny_patch16_224.fb_in1k"}
CONTEXT_OVERHEAD_GB = 0.6   # CUDA context + allocator slack per process
RAM_PER_RUN_GB = 4.0        # host RAM per concurrent run (torch/CUDA process + its data-loader workers)
RAM_RESERVE_GB = 4.0        # scheduler, corruption-cache workers, OS


class Report:
    def __init__(self):
        self.rows, self.fails, self.warns = [], [], []

    def ok(self, what, detail=""):
        self.rows.append(("OK", what, detail))

    def warn(self, what, detail, fix=""):
        self.rows.append(("WARN", what, detail))
        self.warns.append((what, fix))

    def fail(self, what, detail, fix):
        self.rows.append(("FAIL", what, detail))
        self.fails.append((what, fix))

    def print(self):
        w = max(len(r[1]) for r in self.rows)
        print("\n" + "=" * 78 + "\nStage 0 preflight\n" + "=" * 78)
        for s, what, detail in self.rows:
            print(f"[{s:4}] {what.ljust(w)}  {detail}")
        for title, items in (("FIX", self.fails), ("NOTE", self.warns)):
            for what, fix in items:
                if fix:
                    print(f"\n{title} ({what}):\n  " + fix.replace("\n", "\n  "))
        print("=" * 78)


def nvidia_smi_busy():
    """Map GPU index -> list of (pid, MiB) of compute processes, via nvidia-smi if present."""
    try:
        idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    uuid2idx = {u.strip(): i.strip() for i, u in (line.split(",") for line in idx.strip().splitlines())}
    busy = {}
    for line in apps.strip().splitlines():
        u, pid, mem = (x.strip() for x in line.split(","))
        busy.setdefault(uuid2idx.get(u, "?"), []).append((pid, mem))
    return busy


def check_gpus(rep, gpus):
    if not torch.cuda.is_available():
        rep.fail("CUDA", f"torch {torch.__version__} sees no GPU",
                 "Check `nvidia-smi`, the NVIDIA driver (>= 570 for this CUDA 12.8 build) and CUDA_VISIBLE_DEVICES.")
        return []
    rep.ok("torch / CUDA build", f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
                                 f"archs {' '.join(torch.cuda.get_arch_list())}")
    busy = nvidia_smi_busy() or {}
    archs = torch.cuda.get_arch_list()
    info = []
    # torch enumerates GPUs in CUDA_VISIBLE_DEVICES order; preflight must run without it set to index ids.
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        maj, mnr = torch.cuda.get_device_capability(i)
        sm = f"sm_{maj}{mnr}"
        ptx_ok = any(a.startswith("compute_") and int(a.split("_")[1]) <= maj * 10 + mnr for a in archs)
        compat = sm in archs or ptx_ok
        procs = busy.get(str(i), [])
        sel = str(i) in gpus
        detail = (f"{p.name}  {sm}  free {free / 2**30:.1f}/{total / 2**30:.1f} GiB"
                  + (f"  IN USE by pid {', '.join(f'{a}({b}MiB)' for a, b in procs)}" if procs else "")
                  + ("  [selected]" if sel else ""))
        if not compat:
            rep.fail(f"GPU {i} arch", detail, f"torch build lacks {sm}. Blackwell (sm_120) needs a CUDA >= 12.8 "
                     "build; this project pins torch 2.9.1 (cu128) via uv.lock -> `uv sync`.")
        elif sel and procs:
            rep.warn(f"GPU {i}", detail, f"GPU {i} is used by other processes; free memory above is what is left.")
        else:
            rep.ok(f"GPU {i}", detail)
        info.append({"index": i, "name": p.name, "sm": sm, "free_gb": free / 2**30, "total_gb": total / 2**30,
                     "other_procs": len(procs)})
    missing = [g for g in gpus if not g.isdigit() or int(g) >= torch.cuda.device_count()]
    if missing:
        rep.fail("GPUS", f"requested {gpus}, machine has {torch.cuda.device_count()}", "Fix GPUS=...")
    return info


def check_disk_and_perms(rep, paths):
    seen = {}
    for label, p in paths.items():
        p.mkdir(parents=True, exist_ok=True)
        try:
            t = p / f".preflight_{os.getpid()}"
            t.write_text("x")
            t.unlink()
            rep.ok(f"write {label}", str(p))
        except OSError as e:
            rep.fail(f"write {label}", f"{p}: {e}", f"Make {p} writable or point {label} elsewhere.")
        dev = os.stat(p).st_dev
        if dev in seen:
            continue
        seen[dev] = label
        free = shutil.disk_usage(p).free / 2**30
        if free < MIN_DISK_GB:
            rep.fail(f"disk ({label})", f"{free:.0f} GiB free at {p}",
                     f"Need >= {MIN_DISK_GB} GiB (datasets ~2 GB, corruption cache ~8 GB per dataset, "
                     "checkpoints/logs).")
        else:
            rep.ok(f"disk ({label})", f"{free:.0f} GiB free at {p}")


def url_ok(url, timeout=15):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "stage0-preflight"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400, str(r.status)
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__ + ": " + str(e)[:80]


def weights_cached(repo):
    try:
        from huggingface_hub import try_to_load_from_cache
        r = try_to_load_from_cache(repo, "model.safetensors")
        return isinstance(r, str) and os.path.exists(r)
    except Exception:  # noqa: BLE001
        return False


def check_network(rep, data_root):
    from stage0.datasets import SPLIT_FILE
    from stage0.prepare_data import SOURCES
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    need_hf = [r for r in HF_REPOS.values() if not weights_cached(r)]
    if not need_hf:
        rep.ok("timm weights", f"cached under {hf_home}")
    else:
        ok, why = url_ok(f"https://huggingface.co/{need_hf[0]}/resolve/main/config.json")
        if ok:
            rep.ok("Hugging Face Hub", "reachable (weights download on first use)")
        else:
            rep.fail("Hugging Face Hub", f"unreachable ({why}) and weights not cached",
                     "On a machine with internet:\n"
                     + "\n".join(f"  HF_HOME=/tmp/hf python -c \"import timm; timm.create_model('{a}', pretrained=True)\""
                                 for a in HF_REPOS)
                     + f"\nthen copy /tmp/hf/hub/models--timm--deit_*_patch16_224.fb_in1k to {hf_home}/hub/ "
                       "on the server (files: config.json, model.safetensors).")
    for ds, src in SOURCES.items():
        if (Path(data_root) / ds / SPLIT_FILE).exists():
            rep.ok(f"data {ds}", "prepared")
            continue
        if (Path(data_root) / "downloads" / src["archive"]).exists() or (Path(data_root) / ds / src["folder"]).is_dir():
            rep.ok(f"data {ds}", "archive present; run prepare_data.py")
            continue
        results = [url_ok(u) for u in src["urls"]]
        if any(ok for ok, _ in results):
            rep.ok(f"data {ds}", "download URL reachable")
        else:
            rep.fail(f"data {ds}", f"not present and URL unreachable ({results[0][1]})",
                     f"Download {src['archive']} from\n  " + "\n  ".join(src["urls"])
                     + f"\nand copy it to {Path(data_root) / 'downloads' / src['archive']}, then run prepare_data.py.")


def try_step(kind, micro, device, num_classes=200):
    from stage0.models import create_student, create_teacher, teacher_forward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = (create_teacher if kind == "teacher" else create_student)(num_classes, pretrained=False).to(device)
    teacher = None
    if kind == "kd":
        teacher = create_teacher(num_classes, pretrained=False).to(device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(micro, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    y = torch.randint(0, num_classes, (micro,), device=device)
    times = []
    try:
        for _ in range(3):
            torch.cuda.synchronize(device)
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out.float(), y)
                if teacher is not None:
                    t = teacher_forward(teacher, x)
                    loss = loss + torch.nn.functional.kl_div(out.float().log_softmax(1), t.float().softmax(1),
                                                             reduction="batchmean")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            times.append(time.time() - t0)
        return torch.cuda.max_memory_reserved(device) / 2**30, min(times)
    finally:
        del model, teacher, opt, x, y
        torch.cuda.empty_cache()


def memory_test(rep, gpu_index):
    device = torch.device(f"cuda:{gpu_index}")
    res = {}
    for kind in ("teacher", "kd"):
        micro = C.BATCH_SIZE
        while micro >= 2:
            try:
                peak, t = try_step(kind, micro, device)
                res[kind] = {"micro_batch": micro, "peak_gb": peak, "step_s": t}
                break
            except torch.OutOfMemoryError:
                micro //= 2
        if kind not in res:
            rep.fail(f"memory {kind}", "does not fit even at micro-batch 2", "Use a GPU with more free memory.")
            return None
        r = res[kind]
        accum = -(-C.BATCH_SIZE // r["micro_batch"])
        rep.ok(f"memory {kind} step", f"micro-batch {r['micro_batch']} x accum {accum} = 128, "
                                      f"peak {r['peak_gb']:.1f} GiB, {r['step_s'] * accum:.2f} s/step(128)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=os.environ.get("GPUS"), help='GPUs you plan to use, e.g. "0,1"')
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    ap.add_argument("--skip-memory-test", action="store_true")
    args = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("[note] CUDA_VISIBLE_DEVICES is set; GPU indices below are relative to it.")
    data_root, out_root = C.data_root(args.data_root), C.output_root(args.output_root)
    gpus = [g.strip() for g in (args.gpus or "").split(",") if g.strip()]
    rep = Report()
    if not gpus:
        rep.fail("GPUS", "not set", 'Pass --gpus "0,1" or export GPUS="0,1" (GPUs to use on this shared server).')

    gpu_info = check_gpus(rep, gpus)
    paths = {"DATA_ROOT": data_root, "OUTPUT_ROOT": out_root, "CORRUPTION_ROOT": C.corruption_root(data_root)}
    if os.environ.get("HF_HOME"):
        paths["HF_HOME"] = Path(os.environ["HF_HOME"]).expanduser()
    else:
        rep.warn("HF_HOME", "not set", "export HF_HOME=/path/with/space (timm weight cache); default ~/.cache/huggingface")
    check_disk_and_perms(rep, paths)
    check_network(rep, data_root)
    ncpu = C.cpu_count()
    host = os.cpu_count() or 1
    rep.ok("CPU", f"{ncpu} usable" + (f" (host reports {host}; container/affinity limit applied)" if host != ncpu else ""))
    ram = C.mem_limit_gb()
    if ram < RAM_RESERVE_GB + RAM_PER_RUN_GB:
        rep.warn("RAM", f"{ram:.1f} GiB usable", f"Less than {RAM_RESERVE_GB + RAM_PER_RUN_GB:.0f} GiB: runs may be "
                 "killed by the OOM killer. Use RUNS_PER_GPU=1 or a machine with more RAM.")
    else:
        rep.ok("RAM", f"{ram:.1f} GiB usable (container limit applied)")

    mem = None
    if gpu_info and gpus and not args.skip_memory_test and not rep.fails:
        g = int(gpus[0])
        mem = memory_test(rep, g)
    if mem:
        sel = [x for x in gpu_info if str(x["index"]) in gpus]
        free = min(x["free_gb"] for x in sel)
        per_run = mem["kd"]["peak_gb"] + CONTEXT_OVERHEAD_GB
        by_mem = int(free * 0.92 // per_run)
        by_cpu = max(1, ncpu // (2 * len(gpus)))
        by_ram = max(1, int((ram - RAM_RESERVE_GB) // RAM_PER_RUN_GB) // len(gpus))
        runs = max(1, min(8, by_mem, by_cpu, by_ram))
        t_cost = min(runs, -(-(mem["teacher"]["peak_gb"] + CONTEXT_OVERHEAD_GB) // per_run))
        rep.ok("RUNS_PER_GPU (suggested)", f"{runs}  (GPU memory allows {by_mem}, CPU allows {by_cpu}, "
                                           f"RAM allows {by_ram})")
        rep.ok("teacher slot cost", f"{int(t_cost)} of {runs} slots (max 1 teacher fine-tune per GPU; "
                                    f"student runs share the rest)")
        steps = {"cub": 5394 // 128, "waterbirds": 4795 // 128}
        th = sum(steps.values()) * C.EPOCHS * mem["teacher"]["step_s"] * -(-128 // mem["teacher"]["micro_batch"])
        sh = sum(steps.values()) * C.EPOCHS * mem["kd"]["step_s"] * -(-128 // mem["kd"]["micro_batch"])
        rep.ok("rough compute estimate", f"one teacher run (both datasets) ~{th / 3600:.1f} GPU-h of steps; "
                                         f"one KD run (both datasets) <= ~{sh / 3600:.1f} GPU-h (data loading excluded)")
        C.save_json(out_root / "preflight.json", {
            "micro_batch": {"teacher": mem["teacher"]["micro_batch"], "ce": mem["kd"]["micro_batch"],
                            "kd": mem["kd"]["micro_batch"]},
            "suggested_runs_per_gpu": runs, "teacher_slot_cost": int(t_cost), "memory": mem, "gpus": sel, "cpu_count": ncpu, "ram_gb": ram,
            "torch": torch.__version__, "cuda": torch.version.cuda, "created": time.strftime("%F %T")})
    rep.print()
    if rep.fails:
        print(f"PREFLIGHT FAILED ({len(rep.fails)} problem(s)). Fix the items above and rerun.")
        sys.exit(1)
    print(f"PREFLIGHT OK. Settings written to {out_root / 'preflight.json'}")


if __name__ == "__main__":
    main()
