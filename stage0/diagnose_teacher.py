"""Teacher diagnosis (4-1): how much non-target information does the fine-tuned teacher emit?

    uv run python -m stage0.diagnose_teacher --dataset cub

Views (no training):
  train_view : train images with the train augmentation (RRC + flip + RandAugment + random erasing),
               BEFORE mixup/cutmix -- one pass, fixed seed 0
  test_clean : test images with Resize(256) -> CenterCrop(224)
Metrics per view: argmax == y rate; mean p_T(y) at tau=1 and 4; mean entropy of the distribution
renormalised over the wrong classes, divided by log(C-1), at tau=1 and 4.
(For C=2 there is a single wrong class, so the normalised wrong-class entropy is undefined -> null.)

Writes outputs/{dataset}/teacher/seed0/diagnosis.json.
"""
import argparse
import math

import torch

from stage0 import common as C
from stage0.datasets import Stage0Dataset, build_eval_transform, build_train_transform
from stage0.engine import get_device, make_loader, predict
from stage0.models import load_finetuned_teacher

TAUS = (1.0, 4.0)
DIAG_SEED = 0


def view_metrics(logits, labels):
    logits = logits.double()
    n, num_classes = logits.shape
    out = {"n": n, "acc": (logits.argmax(1) == labels).double().mean().item()}
    onehot = torch.nn.functional.one_hot(labels, num_classes).bool()
    for tau in TAUS:
        z = logits / tau
        p_y = torch.softmax(z, 1)[torch.arange(n), labels]
        out[f"p_true_tau{tau:g}"] = p_y.mean().item()
        if num_classes > 2:
            logq = torch.log_softmax(z.masked_fill(onehot, float("-inf")), 1)
            q = logq.exp()
            ent = -(q * logq.masked_fill(onehot, 0.0)).sum(1) / math.log(num_classes - 1)
            out[f"wrong_entropy_norm_tau{tau:g}"] = ent.mean().item()
        else:
            out[f"wrong_entropy_norm_tau{tau:g}"] = None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--data-root")
    ap.add_argument("--output-root")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="[smoke] images per view")
    args = ap.parse_args(argv)

    out_root, data_root = C.output_root(args.output_root), C.data_root(args.data_root)
    tdir = C.teacher_dir(out_root, args.dataset)
    if not (tdir / "DONE").exists():
        raise SystemExit(f"[error] teacher run {tdir} is not finished (no DONE)")
    C.set_cpu_threads()
    device = get_device()
    teacher, ck = load_finetuned_teacher(tdir / "best.pt", device)
    if ck["config"]["dataset"] != args.dataset:
        raise SystemExit(f"[error] {tdir}/best.pt is for {ck['config']['dataset']}")
    nw = args.num_workers if args.num_workers is not None else C.default_num_workers()

    result = {"dataset": args.dataset, "teacher_ckpt": str(tdir / "best.pt"), "teacher_epoch": ck["epoch"],
              "teacher_val_acc": ck["val_acc"], "num_classes": ck["config"]["num_classes"], "seed": DIAG_SEED,
              "train_view_transform": "RRC(0.08-1,bicubic)+hflip+RandAugment(rand-m9-mstd0.5-inc1)+"
                                      "RandomErasing(0.25); no mixup/cutmix"}
    views = {"train_view": ("train", build_train_transform()), "test_clean": ("test", build_eval_transform())}
    for name, (split, tf) in views.items():
        ds = Stage0Dataset(data_root, args.dataset, split, tf, limit=args.limit)
        ds.set_aug_seed(DIAG_SEED)
        C.seed_everything(DIAG_SEED)
        loader = make_loader(ds, args.batch_size, nw, shuffle=False,
                             generator=torch.Generator().manual_seed(DIAG_SEED))
        logits, labels, _ = predict(teacher, loader, device)
        result[name] = view_metrics(logits, labels)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result[name].items()})

    C.save_json(tdir / "diagnosis.json", result)
    tv = result["train_view"]
    if tv["p_true_tau1"] > 0.95 and (tv["wrong_entropy_norm_tau1"] or 0) > 0.95:
        print("NOTE: train-view p_T(y)~1 and wrong-class entropy~1 -> little dark knowledge (KD ~ CE likely)")
    print(f"wrote {tdir / 'diagnosis.json'}")


if __name__ == "__main__":
    main()
