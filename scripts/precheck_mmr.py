"""Pre-check before spending GPU on mk_filt / mk_mmr / mk_fmmr: on real val images, does the selection actually differ
from the MaskedKD top-k, and are there sink tokens to filter? (~1 min, 1 GPU or CPU)

    uv run python scripts/precheck_mmr.py --dataset waterbirds                       # ImageNet-init student (start of training)
    uv run python scripts/precheck_mmr.py --dataset waterbirds \
        --ckpt $OUTPUT_ROOT/waterbirds/maskedkd_k0.15/seed0/ckpt_e10.pt $OUTPUT_ROOT/waterbirds/maskedkd_k0.15/seed0/last.pt
    uv run python scripts/precheck_mmr.py --dataset imagenet100 --no-init \
        --ckpt $OUTPUT_ROOT/imagenet100/maskedkd_k0.15/seed0/ckpt_e10.pt $OUTPUT_ROOT/imagenet100/maskedkd_k0.15/seed0/last.pt

Columns: sink_topk = share of the MaskedKD top-k that are high-norm (> 3x median) tokens; redund = mean pairwise cosine
of the selected tokens' features (top-k vs mk_fmmr); overlap = |selected ∩ MaskedKD top-k| / k per criterion.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import common as C  # noqa: E402
from stage0.datasets import Stage0Dataset, build_eval_transform  # noqa: E402
from stage0.masking import StudentMaskSelector  # noqa: E402
from stage0.models import create_student, load_student_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds", choices=C.ALL_DATASETS)
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--no-init", action="store_true", help="skip the ImageNet-init student")
    ap.add_argument("--data-root")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Stage0Dataset(C.data_root(a.data_root), a.dataset, "val", build_eval_transform(), limit=a.n)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    models = [] if a.no_init else [("imagenet-init", create_student(C.NUM_CLASSES[a.dataset], pretrained=True).to(dev).eval())]
    models += [(str(Path(p).relative_to(Path(p).parents[2])), load_student_weights(p, dev)) for p in a.ckpt]
    print(f"{a.dataset} val, {len(ds)} images, keep {a.keep}\n")
    print(f"{'student':<40}{'sink_topk':>10}{'redund topk':>12}{'redund fmmr':>12}"
          f"{'ovl filt':>10}{'ovl mmr':>9}{'ovl fmmr':>10}")
    for name, m in models:
        res = {}
        for crit in ("mk_filt", "mk_mmr", "mk_fmmr"):
            with StudentMaskSelector(crit, a.keep, m) as sel, torch.no_grad():
                for x in xs.split(64):
                    m(x.to(dev))
                    sel.select(x.shape[0], dev)
                res[crit] = sel.pop_diag()
        f = res["mk_fmmr"]
        print(f"{name:<40}{f['sink_frac_topk']:10.3f}{f['redund_topk']:12.3f}{f['redund_sel']:12.3f}"
              f"{res['mk_filt']['overlap_topk']:10.3f}{res['mk_mmr']['overlap_topk']:9.3f}{f['overlap_topk']:10.3f}")
    print("\nread: ovl ~1.0 -> that criterion is ~MaskedKD (cannot test the hypothesis); "
          "sink_topk ~0 -> nothing to filter (mk_filt = maskedkd).")


if __name__ == "__main__":
    main()
