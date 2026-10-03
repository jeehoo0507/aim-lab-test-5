"""Stage 0 training: teacher fine-tune, student CE, student full KD.

    uv run python -m stage0.train --mode {teacher,ce,kd} --dataset {cub,waterbirds} --seed S --lr LR

Output: $OUTPUT_ROOT/{dataset}/{mode}/seed{seed}/   (--subdir overrides "{mode}", used for lr selection)
    config.json   resolved settings (a rerun with different settings is refused)
    train.log     stdout/stderr of every attempt (appended)
    log.csv       per epoch: lr, train CE term, train KD term, train acc, val acc
    status.json   live progress (read by status.sh)
    last.pt       model / optimizer / scheduler / AMP scaler / RNG / epoch, every epoch (resume point)
    best.pt       teacher only: best val-accuracy epoch
    DONE          written when all epochs finished;  FAILED  written with the traceback on error

Rerunning the same command resumes from last.pt. --overwrite starts from scratch.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
from timm.data import Mixup
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler

from stage0 import common as C
from stage0.datasets import Stage0Dataset, build_eval_transform, build_train_transform
from stage0.engine import accuracy, amp_ctx, get_device, make_loader, predict
from stage0.losses import Stage0Loss
from stage0.models import create_student, create_teacher, load_finetuned_teacher, teacher_forward

# Keys that may differ between the original run and a resumed one.
# The teacher identity is checked separately (teacher_run_uid) when the teacher is loaded.
RESUME_TOLERANT_KEYS = {"num_workers", "micro_batch", "run_uid", "created", "argv", "teacher_ckpt", "teacher_run_uid"}
CSV_FIELDS = ["epoch", "lr", "train_loss", "train_ce", "train_kd", "train_acc", "val_acc", "epoch_time_s"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=C.MODES)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=C.KD_ALPHA, help="KD weight (kd only; 1.0 = fallback run)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE, help="effective batch")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="per-step batch (grad accumulation up to --batch-size); default from preflight.json")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="default min(8, cpu_count // $STAGE0_CONCURRENT_RUNS)")
    ap.add_argument("--subdir", default=None, help="replace the {mode} path component (lr selection)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    # smoke-test only
    ap.add_argument("--limit", type=int, default=None, help="[smoke] use N train and N val images")
    ap.add_argument("--random-init", action="store_true",
                    help="[smoke, offline] skip ImageNet weights. Never use for real runs.")
    return ap.parse_args(argv)


def build_config(args, out_root, data_root):
    rec = C.RECIPE[args.mode]
    C.assert_recipe(args.mode, **rec)
    if args.mode != "kd" and args.alpha != C.KD_ALPHA:
        sys.exit("[error] --alpha only applies to --mode kd")
    if args.batch_size % 2:
        sys.exit("[error] batch size must be even (mixup)")
    return {
        "mode": args.mode, "dataset": args.dataset, "seed": args.seed, "lr": args.lr,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "arch": C.TEACHER_ARCH if args.mode == "teacher" else C.STUDENT_ARCH,
        "pretrained": not args.random_init, "num_classes": C.NUM_CLASSES[args.dataset],
        "optimizer": "adamw", "weight_decay": C.WEIGHT_DECAY, "opt_eps": C.OPT_EPS,
        "sched": "cosine", "warmup_epochs": C.WARMUP_EPOCHS, "warmup_lr": C.WARMUP_LR,
        "min_lr": args.lr * C.MIN_LR_RATIO, "cooldown_epochs": C.COOLDOWN_EPOCHS,
        "drop_path": C.DROP_PATH, "interpolation": C.INTERPOLATION,
        "rrc_scale": list(C.RRC_SCALE), "hflip": C.HFLIP, "rand_augment": C.RAND_AUGMENT,
        "reprob": C.REPROB, "remode": C.REMODE, "repeated_aug": C.REPEATED_AUG,
        "smoothing": rec["smoothing"], "mixup": rec["mixup"], "cutmix": rec["cutmix"],
        "mixup_prob": C.MIXUP_PROB, "mixup_switch_prob": C.MIXUP_SWITCH_PROB, "mixup_mode": C.MIXUP_MODE,
        "kd_alpha": args.alpha if args.mode == "kd" else None,
        "kd_tau": C.KD_TAU if args.mode == "kd" else None,
        "amp": not args.no_amp, "limit": args.limit,
        "data_root": str(data_root), "output_root": str(out_root),
    }


def prepare_run_dir(rd, config, overwrite):
    """Unique, collision-safe run directory. Returns 'new' | 'resume' | 'done'."""
    cfg_path = rd / "config.json"
    if overwrite:
        for p in rd.iterdir():
            if p.name != "RUNNING.lock":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        return "new"
    if cfg_path.exists():
        old = C.load_json(cfg_path)
        diff = {k: (old.get(k), config.get(k)) for k in set(old) | set(config)
                if k not in RESUME_TOLERANT_KEYS and old.get(k) != config.get(k)}
        if diff:
            sys.exit(f"[error] {rd} already holds a run with different settings:\n"
                     + "\n".join(f"  {k}: existing={a!r} requested={b!r}" for k, a, b in
                                 ((k, *v) for k, v in sorted(diff.items())))
                     + "\nUse a different output path or --overwrite.")
        if (rd / "DONE").exists():
            return "done"
        return "resume" if (rd / "last.pt").exists() else "new"
    # files an earlier attempt can leave behind if it failed before writing config.json
    own = ("RUNNING.lock", "train.log", "FAILED", "status.json")
    foreign = [p.name for p in rd.iterdir() if p.name not in own]
    if foreign:
        sys.exit(f"[error] {rd} exists and was not created by this config ({foreign[:5]}). "
                 "Refusing to write into it without --overwrite.")
    return "new"


def check_teacher(out_root, dataset, num_classes):
    tdir = C.teacher_dir(out_root, dataset)
    ck = tdir / "best.pt"
    if not ck.exists():
        sys.exit(f"[error] KD needs the fine-tuned teacher {ck}, which does not exist. Train it first.")
    if not (tdir / "DONE").exists():
        sys.exit(f"[error] teacher run {tdir} has no DONE file (still training or failed).")
    tcfg = C.load_json(tdir / "config.json")
    meta = torch.load(ck, map_location="cpu", weights_only=False)
    for cfg in (tcfg, meta["config"]):
        if cfg["dataset"] != dataset or cfg["mode"] != "teacher":
            sys.exit(f"[error] teacher checkpoint {ck} was trained for dataset={cfg['dataset']} "
                     f"mode={cfg['mode']}, but this KD run is for dataset={dataset}.")
    if meta["config"]["num_classes"] != num_classes:
        sys.exit(f"[error] teacher has {meta['config']['num_classes']} classes, expected {num_classes}")
    return ck, tcfg


def rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def train(args, rd, config, state):
    device = get_device()
    amp = config["amp"]
    C.set_cpu_threads()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    num_workers = args.num_workers if args.num_workers is not None else C.default_num_workers()
    pf = C.preflight_settings(config["output_root"])
    micro = args.micro_batch or (pf or {}).get("micro_batch", {}).get(args.mode) or args.batch_size
    micro = min(micro, args.batch_size)
    config["num_workers"], config["micro_batch"] = num_workers, micro
    print(f"run dir: {rd}\nstate: {state}  device: {device}  micro_batch: {micro}  "
          f"accum: {-(-args.batch_size // micro)}  num_workers: {num_workers}")

    data_root = Path(config["data_root"])
    train_ds = Stage0Dataset(data_root, args.dataset, "train", build_train_transform(), limit=args.limit)
    val_ds = Stage0Dataset(data_root, args.dataset, "val", build_eval_transform(), limit=args.limit)
    if len(train_ds) < args.batch_size:
        sys.exit(f"[error] train set ({len(train_ds)}) smaller than batch size {args.batch_size}")
    val_loader = make_loader(val_ds, 256, num_workers)
    print(f"train {len(train_ds)}  val {len(val_ds)}  steps/epoch {len(train_ds) // args.batch_size}")

    # Same seed => same head init and data order for CE and KD (student is built before the teacher).
    C.seed_everything(args.seed)
    build = create_teacher if args.mode == "teacher" else create_student
    model = build(config["num_classes"], pretrained=config["pretrained"]).to(device)

    teacher = None
    if args.mode == "kd":
        ck, tcfg = check_teacher(config["output_root"], args.dataset, config["num_classes"])
        if config.get("teacher_run_uid") not in (None, tcfg["run_uid"]):
            sys.exit(f"[error] teacher {ck} changed since this KD run started; use --overwrite")
        config["teacher_ckpt"], config["teacher_run_uid"] = str(ck), tcfg["run_uid"]
        teacher, tmeta = load_finetuned_teacher(ck, device)
        print(f"teacher: {ck} (epoch {tmeta['epoch']}, val acc {tmeta['val_acc']:.2f})")

    optimizer = create_optimizer_v2(model, opt="adamw", lr=args.lr, weight_decay=C.WEIGHT_DECAY, eps=C.OPT_EPS)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=config["min_lr"],
                                  warmup_t=C.WARMUP_EPOCHS, warmup_lr_init=C.WARMUP_LR, cycle_limit=1,
                                  t_in_epochs=True, warmup_prefix=False)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    criterion = Stage0Loss(args.mode, alpha=args.alpha, tau=C.KD_TAU)
    mixup_fn = None
    if config["mixup"] > 0 or config["cutmix"] > 0:
        mixup_fn = Mixup(mixup_alpha=config["mixup"], cutmix_alpha=config["cutmix"], prob=C.MIXUP_PROB,
                         switch_prob=C.MIXUP_SWITCH_PROB, mode=C.MIXUP_MODE,
                         label_smoothing=config["smoothing"], num_classes=config["num_classes"])
    assert (mixup_fn is None) == (args.mode == "teacher")

    start_epoch, best_val, best_epoch = 0, -1.0, -1
    if state == "resume":
        ck = torch.load(rd / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start_epoch, best_val, best_epoch = ck["epoch"] + 1, ck["best_val"], ck["best_epoch"]
        print(f"resuming from {rd / 'last.pt'} at epoch {start_epoch}")
    C.save_json(rd / "config.json", config)

    csv_path = rd / "log.csv"
    rows = [r for r in read_csv_rows(csv_path) if int(r["epoch"]) < start_epoch]
    write_csv_rows(csv_path, rows)

    steps = len(train_ds) // args.batch_size
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]
        # Everything random in this epoch derives from (seed, epoch): data order, per-sample
        # augmentation, mixup, drop path. CE and KD with the same seed see identical batches.
        C.seed_everything(C.epoch_seed(args.seed, epoch, 0))
        train_ds.set_aug_seed(C.epoch_seed(args.seed, epoch, 3))
        sampler = torch.utils.data.RandomSampler(
            train_ds, generator=torch.Generator().manual_seed(C.epoch_seed(args.seed, epoch, 1)))
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        loader = make_loader(train_ds, None, num_workers, batch_sampler=batch_sampler,
                             generator=torch.Generator().manual_seed(C.epoch_seed(args.seed, epoch, 2)))
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "n": 0}
        for step, (x, y, _g) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target = y
            if mixup_fn is not None:
                x, target = mixup_fn(x, y)
            n = x.shape[0]
            optimizer.zero_grad(set_to_none=True)
            for xs, ts, ys in zip(x.split(micro), target.split(micro), y.split(micro)):
                with amp_ctx(device, amp):
                    out = model(xs)
                    t_logits = teacher_forward(teacher, xs) if teacher is not None else None
                loss, ce, kd = criterion(out, ts, t_logits)
                scaler.scale(loss * (xs.shape[0] / n)).backward()
                sums["loss"] += loss.item() * xs.shape[0]
                sums["ce"] += ce.item() * xs.shape[0]
                sums["kd"] += kd.item() * xs.shape[0] if kd is not None else 0.0
                sums["correct"] += (out.argmax(1) == ys).sum().item()
            if not np.isfinite(sums["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            scaler.step(optimizer)
            scaler.update()
            sums["n"] += n
        assert sums["n"] == steps * args.batch_size, (sums["n"], steps)

        model.eval()
        logits, labels, _ = predict(model, val_loader, device, amp)
        val_acc = accuracy(logits, labels)
        n = sums["n"]
        row = {"epoch": epoch, "lr": f"{lr:.6g}", "train_loss": f"{sums['loss'] / n:.5f}",
               "train_ce": f"{sums['ce'] / n:.5f}",
               "train_kd": f"{sums['kd'] / n:.5f}" if args.mode == "kd" else "",
               "train_acc": f"{100.0 * sums['correct'] / n:.3f}", "val_acc": f"{val_acc:.3f}",
               "epoch_time_s": f"{time.time() - t0:.1f}"}

        if args.mode == "teacher" and val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            C.atomic_torch_save({"model": model.state_dict(), "config": config, "epoch": epoch,
                                 "val_acc": val_acc}, rd / "best.pt")
        elif args.mode != "teacher" and val_acc > best_val:
            best_val, best_epoch = val_acc, epoch  # logged only; students are evaluated at the last epoch
        rows.append(row)
        write_csv_rows(csv_path, rows)
        C.atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                             "rng": rng_state(), "epoch": epoch, "best_val": best_val,
                             "best_epoch": best_epoch, "config": config}, rd / "last.pt")
        C.save_json(rd / "status.json", {"state": "running", "epoch": epoch + 1, "epochs": args.epochs,
                                         "val_acc": val_acc, "updated": time.time()})
        print(f"epoch {epoch + 1}/{args.epochs}  lr {lr:.3g}  loss {row['train_loss']}  ce {row['train_ce']}  "
              f"kd {row['train_kd'] or '-'}  train_acc {row['train_acc']}  val_acc {val_acc:.2f}  "
              f"({row['epoch_time_s']}s)", flush=True)

    final_val = float(rows[-1]["val_acc"])
    C.save_json(rd / "status.json", {"state": "done", "epoch": args.epochs, "epochs": args.epochs,
                                     "val_acc": final_val, "updated": time.time()})
    C.save_json(rd / "DONE", {"final_val_acc": final_val, "best_val_acc": best_val, "best_epoch": best_epoch,
                              "epochs": args.epochs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"done. final val acc {final_val:.2f}  best {best_val:.2f} @ epoch {best_epoch + 1}")


def main(argv=None):
    args = parse_args(argv)
    out_root, data_root = C.output_root(args.output_root), C.data_root(args.data_root)
    config = build_config(args, out_root, data_root)
    rd = C.run_dir(out_root, args.dataset, args.mode, args.seed, args.alpha, args.subdir)
    rd.mkdir(parents=True, exist_ok=True)
    with C.RunLock(rd):
        state = prepare_run_dir(rd, config, args.overwrite)
        if state == "done":
            print(f"{rd} already finished (DONE present); nothing to do.")
            return
        if state == "new":
            config["run_uid"] = uuid.uuid4().hex
            config["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            old = C.load_json(rd / "config.json")
            config["run_uid"], config["created"] = old["run_uid"], old["created"]
            if "teacher_run_uid" in old:
                config["teacher_run_uid"] = old["teacher_run_uid"]
        config["argv"] = sys.argv
        C.tee_output(rd / "train.log")
        (rd / "FAILED").unlink(missing_ok=True)
        try:
            train(args, rd, config, state)
        except BaseException as e:
            if isinstance(e, SystemExit) and e.code in (0, None):
                raise
            # a refusal (sys.exit("[error] ...")) is reported as its message; anything else with traceback
            tb = str(e.code) if isinstance(e, SystemExit) else traceback.format_exc()
            if not isinstance(e, SystemExit):
                print(tb, file=sys.stderr)
            (rd / "FAILED").write_text(tb)
            C.save_json(rd / "status.json", {"state": "failed", "updated": time.time(), "error": repr(e)[:500]})
            raise


if __name__ == "__main__":
    main()
