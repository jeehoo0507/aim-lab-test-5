"""Stage 0 training: teacher fine-tune, student CE, student full KD.

    uv run python -m stage0.train --mode {teacher,ce,kd} --dataset {cub,waterbirds} --seed S --lr LR
    uv run python -m stage0.train --mode maskedkd --mask-criterion {maskedkd,random} --keep 0.3 ...  (Stage 2)

Output: $OUTPUT_ROOT/{dataset}/{mode}/seed{seed}/   (--subdir overrides "{mode}", used for lr selection)
    config.json   resolved settings (a rerun with different settings is refused)
    train.log     stdout/stderr of every attempt (appended)
    log.csv       per epoch: lr, train CE term, train KD term, train acc, val acc
    status.json   live progress (read by status.sh)
    last.pt       model / optimizer / scheduler / AMP scaler / RNG / epoch, every epoch (resume point)
    best.pt       teacher only: best val-accuracy epoch
    ckpt_e{10,30,60}.pt  maskedkd only: student weights at those epochs (later fidelity measurements)
    DONE          written when all epochs finished;  FAILED  written with the traceback on error

Rerunning the same command resumes from last.pt. --overwrite starts from scratch.

Stage 2 (--mode maskedkd): identical to kd (recipe, alpha, tau, lr, teacher, data order) except that the
teacher sees only k = round(keep * 196) patch tokens chosen per image by --mask-criterion; output dir
{dataset}/{criterion}_k{keep}/seed{seed}/. log.csv gains mask_agree: masked vs full teacher argmax agreement
on the first batch of each epoch (one extra full teacher forward per epoch).
Stage 3 pilot (--mask-criterion tam / tam_var): TAM token selection from the cached teacher attribution
(attribution_cache/, crop-mapped with the sample's RRC box / flip, then mixed like the images by mixup / cutmix) and the student's last-block
attention; tam_var adds the 3-bucket per-image budget. The train loader then uses the box-returning transform
(bit-identical images) and passes the sample index. log.csv gains tam_gap_ratio, mean_k, bucket_k and
teacher_gflops (measured per-image teacher GFLOPs averaged over the epoch's images).
STAGE0_BATCH_HASH_LOG=path (any mode, off by default) appends per-step hashes of the student input batch and
mixup targets, to check that same-seed runs see identical data.
"""
import argparse
import contextlib
import csv
import hashlib
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
from stage0.attribution import KINDS as ATTR_KINDS
from stage0.attribution import AttributionCache, crop_maps, teacher_attribution, topk_overlap
from stage0.datasets import Stage0Dataset, WithIndex, build_eval_transform, build_train_transform
from stage0.engine import accuracy, amp_ctx, get_device, make_loader, numpy_collate_box_index, predict
from stage0.losses import Stage0Loss
from stage0.masking import (TAM_CACHE_CRITERIA, TAM_CRITERIA, TAM_DELTA, TAM_GAP, TRAIN_CRITERIA,
                            RecordingMixup, StudentMaskSelector, TamSelector, mix_attribution, num_keep,
                            tam_bucket_ks)
from stage0.models import (create_student, create_teacher, load_finetuned_teacher, teacher_forward,
                           teacher_forward_buckets)

# Keys that may differ between the original run and a resumed one.
# The teacher identity is checked separately (teacher_run_uid) when the teacher is loaded.
RESUME_TOLERANT_KEYS = {"num_workers", "micro_batch", "run_uid", "created", "argv", "teacher_ckpt", "teacher_run_uid"}
CSV_FIELDS = ["epoch", "lr", "train_loss", "train_ce", "train_kd", "train_acc", "val_acc", "epoch_time_s"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=C.TRAIN_MODES)
    ap.add_argument("--dataset", required=True, choices=C.ALL_DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=C.KD_ALPHA, help="KD weight (kd only; 1.0 = fallback run)")
    ap.add_argument("--mask-criterion", choices=TRAIN_CRITERIA, default=None, help="[maskedkd] token selection")
    ap.add_argument("--keep", type=float, default=None, help="[maskedkd] fraction of the 196 patch tokens kept")
    ap.add_argument("--tam-gap", type=float, nargs=2, default=None, metavar=("G0", "G1"),
                    help=f"[tam] gap ratio from G0 (first epoch) to G1 (last), linear; default {TAM_GAP}")
    ap.add_argument("--tam-kind", choices=ATTR_KINDS, default=None, help="[tam] cached attribution (attn_last)")
    ap.add_argument("--tam-budget", choices=("fixed", "bucket"), default=None,
                    help="[tam] fixed (tam) or 3-bucket (tam_var); implied by the criterion")
    ap.add_argument("--tam-delta", type=float, default=None, help=f"[tam_var] bucket spread (default {TAM_DELTA})")
    ap.add_argument("--ckpt-epochs", type=int, nargs="*", default=list(C.CKPT_EPOCHS),
                    help="[maskedkd] epochs (1-based) after which ckpt_e{epoch}.pt (student weights) is saved")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE, help="effective batch")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="per-step batch (grad accumulation up to --batch-size); default from preflight.json")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="default min(8, usable CPUs (cgroup-aware) // $STAGE0_CONCURRENT_RUNS)")
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
    if args.mode == C.MASK_MODE and (args.mask_criterion is None or args.keep is None):
        sys.exit("[error] --mode maskedkd needs --mask-criterion and --keep")
    if args.mode != C.MASK_MODE and (args.mask_criterion is not None or args.keep is not None):
        sys.exit("[error] --mask-criterion / --keep only apply to --mode maskedkd")
    tam = args.mask_criterion in TAM_CRITERIA
    if not tam and any(v is not None for v in (args.tam_gap, args.tam_kind, args.tam_budget, args.tam_delta)):
        sys.exit("[error] --tam-* options only apply to --mask-criterion tam / tam_var")
    if tam:
        implied = "bucket" if args.mask_criterion == "tam_var" else "fixed"
        if args.tam_budget not in (None, implied):
            sys.exit(f"[error] --mask-criterion {args.mask_criterion} implies --tam-budget {implied}")
        if implied == "fixed" and args.tam_delta is not None:
            sys.exit("[error] --tam-delta only applies to tam_var")
        args.tam_gap = list(args.tam_gap or TAM_GAP)
        args.tam_kind = args.tam_kind or "attn_last"
        args.tam_budget = implied
        args.tam_delta = (TAM_DELTA if args.tam_delta is None else args.tam_delta) if implied == "bucket" else None
    if args.batch_size % 2:
        sys.exit("[error] batch size must be even (mixup)")
    config = {
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
        "kd_alpha": args.alpha if args.mode in C.KD_MODES else None,
        "kd_tau": C.KD_TAU if args.mode in C.KD_MODES else None,
        "amp": not args.no_amp, "limit": args.limit,
        "data_root": str(data_root), "output_root": str(out_root),
    }
    if args.mode == C.MASK_MODE:   # extra keys only for Stage 2, so Stage 0 configs are unchanged
        config.update({"mask_criterion": args.mask_criterion, "keep": args.keep, "keep_k": num_keep(args.keep),
                       "ckpt_epochs": list(args.ckpt_epochs)})
    if args.mask_criterion in TAM_CRITERIA:   # extra keys only for TAM, so Stage 2 configs are unchanged
        config.update({"tam_gap": args.tam_gap, "tam_kind": args.tam_kind, "tam_budget": args.tam_budget,
                       "tam_delta": args.tam_delta,
                       "tam_teacher_signal": "oracle" if args.mask_criterion == "tam_oracle" else "cache"})
    if args.mask_criterion in ("rollout",) + TAM_CRITERIA:   # selection cost (GFLOPs/img), new criteria only
        from stage0.flops import selection_gflops
        config["selection_gflops"] = round(selection_gflops(args.mask_criterion, args.tam_kind or "attn_last",
                                                            config["num_classes"]), 4)
    return config


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


def write_csv_rows(path, rows, fields=CSV_FIELDS):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
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
    pf_mode = "kd" if args.mode == C.MASK_MODE else args.mode   # preflight measured the full-KD step
    micro = args.micro_batch or (pf or {}).get("micro_batch", {}).get(pf_mode) or args.batch_size
    micro = min(micro, args.batch_size)
    config["num_workers"], config["micro_batch"] = num_workers, micro
    print(f"run dir: {rd}\nstate: {state}  device: {device}  micro_batch: {micro}  "
          f"accum: {-(-args.batch_size // micro)}  num_workers: {num_workers}")

    data_root = Path(config["data_root"])
    tam = args.mode == C.MASK_MODE and args.mask_criterion in TAM_CRITERIA
    tam_cache = tam and args.mask_criterion in TAM_CACHE_CRITERIA
    if tam_cache:   # same images (bit-identical) + crop box and sample index for the attribution-cache lookup
        train_ds = WithIndex(Stage0Dataset(data_root, args.dataset, "train", build_train_transform(return_box=True),
                                           limit=args.limit))
    else:
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
    if args.mode in C.KD_MODES:
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
        mixup_cls = RecordingMixup if tam else Mixup   # tam: records lam / cutmix box (same RNG, same images)
        mixup_fn = mixup_cls(mixup_alpha=config["mixup"], cutmix_alpha=config["cutmix"], prob=C.MIXUP_PROB,
                         switch_prob=C.MIXUP_SWITCH_PROB, mode=C.MIXUP_MODE,
                         label_smoothing=config["smoothing"], num_classes=config["num_classes"])
    assert (mixup_fn is None) == (args.mode == "teacher")
    masked = args.mode == C.MASK_MODE
    if tam:
        attr_cache = AttributionCache(config["output_root"], args.dataset, kinds=(config["tam_kind"],)) \
            if tam_cache else None
        selector = TamSelector(args.mask_criterion, args.keep, model, gap=config["tam_gap"],
                               delta=config["tam_delta"] or TAM_DELTA)
        from stage0.flops import teacher_gflops
        bucket_ks = tam_bucket_ks(num_keep(args.keep), selector.delta) if config["tam_budget"] == "bucket" \
            else (num_keep(args.keep),) * 3
        tflops = {kb: teacher_gflops(kb, config["num_classes"]) for kb in set(bucket_ks)}
        print(f"TAM: kind {config['tam_kind']}, gap {config['tam_gap']}, budget {config['tam_budget']}, "
              f"k per bucket {bucket_ks}, teacher GFLOPs {tflops}")
    else:
        selector = StudentMaskSelector(args.mask_criterion, args.keep, model) if masked else None
    stack = contextlib.ExitStack()
    if selector is not None:
        stack.enter_context(selector)   # maskedkd: hooks on the student's last attention block
    csv_fields = CSV_FIELDS + (["mask_agree"] if masked else []) \
        + (["tam_gap_ratio", "mean_k", "bucket_k", "teacher_gflops"] if tam else []) \
        + (["cache_oracle_overlap"] if tam_cache else [])
    hash_log = os.environ.get("STAGE0_BATCH_HASH_LOG")

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
    write_csv_rows(csv_path, rows, csv_fields)

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
                             generator=torch.Generator().manual_seed(C.epoch_seed(args.seed, epoch, 2)),
                             **({"collate_fn": numpy_collate_box_index} if tam_cache else {}))
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "n": 0}
        mask_agree = None
        if tam:
            selector.set_epoch(epoch, args.epochs)
            tam_tokens, tam_gflops = 0, 0.0
            overlap = None
        for step, batch in enumerate(loader):
            x, y = batch[0], batch[1]
            if tam_cache:   # cached teacher attribution on this view's crop (mixed after mixup below)
                rels = [train_ds.items[i][0] for i in batch[4].tolist()]
                T_attr = crop_maps(attr_cache.lookup(config["tam_kind"], rels).to(device),
                                   batch[3].to(device)).flatten(1)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if tam_cache and step == 0:   # cache vs oracle top-k overlap, first micro-batch of the epoch
                x_pre = x[:micro]
                o = teacher_attribution(teacher, x_pre, config["tam_kind"], lambda: amp_ctx(device, amp))
                overlap = topk_overlap(T_attr[:micro], o, num_keep(args.keep))
            target = y
            if mixup_fn is not None:
                x, target = mixup_fn(x, y)
            if tam_cache:   # cached maps of the two source views -> attribution of the mixed image
                T_attr = mix_attribution(T_attr, mixup_fn.last)
            elif tam:       # tam_oracle: teacher attribution on the mixed image the teacher actually sees
                T_attr = teacher_attribution(teacher, x, config["tam_kind"], lambda: amp_ctx(device, amp), micro)
            if hash_log:
                with open(hash_log, "a") as hf:
                    hf.write(f"{epoch} {step} {hashlib.sha1(x.cpu().numpy().tobytes()).hexdigest()} "
                             f"{hashlib.sha1(target.float().cpu().numpy().tobytes()).hexdigest()}\n")
            n = x.shape[0]
            optimizer.zero_grad(set_to_none=True)
            # random criterion: dedicated generator per step (never the global RNG -> pairing with kd preserved)
            mask_gen = torch.Generator().manual_seed(C.epoch_seed(args.seed, epoch, 4) + step) if masked else None
            t_chunks = T_attr.split(micro) if tam else [None] * len(x.split(micro))
            for c, (xs, ts, ys, t_attr) in enumerate(zip(x.split(micro), target.split(micro), y.split(micro),
                                                          t_chunks)):
                with amp_ctx(device, amp):
                    out = model(xs)
                    if tam:
                        sel = selector.select(xs.shape[0], t_attr)
                        if torch.is_tensor(sel):
                            t_logits = teacher_forward(teacher, xs, sel)
                            parts = [(xs.shape[0], sel.shape[1])]
                        else:
                            t_logits = teacher_forward_buckets(teacher, xs, sel)
                            parts = [(rows.numel(), idx.shape[1]) for rows, idx in sel]
                        tam_tokens += sum(nb * kb for nb, kb in parts)
                        tam_gflops += sum(nb * tflops[kb] for nb, kb in parts)
                        if step == 0 and c == 0:
                            t_full = teacher_forward(teacher, xs)
                            mask_agree = (t_logits.argmax(1) == t_full.argmax(1)).float().mean().item()
                    elif masked:
                        keep_idx = selector.select(xs.shape[0], device, mask_gen)
                        t_logits = teacher_forward(teacher, xs, keep_idx)
                        if step == 0 and c == 0:
                            t_full = teacher_forward(teacher, xs)
                            mask_agree = (t_logits.argmax(1) == t_full.argmax(1)).float().mean().item()
                    else:
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
               "train_kd": f"{sums['kd'] / n:.5f}" if args.mode in C.KD_MODES else "",
               "train_acc": f"{100.0 * sums['correct'] / n:.3f}", "val_acc": f"{val_acc:.3f}",
               "epoch_time_s": f"{time.time() - t0:.1f}"}
        if masked:
            row["mask_agree"] = f"{mask_agree:.4f}"
        if tam:
            row.update({"tam_gap_ratio": f"{selector.g:.4f}", "mean_k": f"{tam_tokens / n:.3f}",
                        "bucket_k": "/".join(map(str, bucket_ks)), "teacher_gflops": f"{tam_gflops / n:.4f}"})
            assert abs(tam_tokens / n - num_keep(args.keep)) <= 1.0, (tam_tokens / n, num_keep(args.keep))
        if tam_cache:
            row["cache_oracle_overlap"] = f"{overlap:.4f}"

        if args.mode == "teacher" and val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            C.atomic_torch_save({"model": model.state_dict(), "config": config, "epoch": epoch,
                                 "val_acc": val_acc}, rd / "best.pt")
        elif args.mode != "teacher" and val_acc > best_val:
            best_val, best_epoch = val_acc, epoch  # logged only; students are evaluated at the last epoch
        rows.append(row)
        write_csv_rows(csv_path, rows, csv_fields)
        if masked and epoch + 1 in config["ckpt_epochs"]:
            C.atomic_torch_save(model.state_dict(), rd / f"ckpt_e{epoch + 1}.pt")
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
    subdir = args.subdir
    if args.mode == C.MASK_MODE and subdir is None:
        subdir = C.mask_dirname(args.mask_criterion, args.keep)
    rd = C.run_dir(out_root, args.dataset, args.mode, args.seed, args.alpha, subdir)
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
