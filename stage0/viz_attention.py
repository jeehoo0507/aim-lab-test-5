"""Visual check of token selection on test images (no training, read-only on run folders).

For each picked image, one row:
  image | student CLS attention (selection signal) | tokens kept (top-k) | teacher CLS attention (full image) |
  teacher top-k tokens
and the predictions of the full teacher vs the masked teacher (teacher fed only the kept tokens).

    uv run python -m stage0.viz_attention --dataset cub --run maskedkd_k0.15 --seed 0 --keep 0.15
    -> results/viz_maskedkd_k0.15_s0.png  (+ a summary line over all scanned images)

--pick mixed (default): scans --scan test images and shows half where the masked teacher keeps the full
teacher's prediction and half where it changes it. --pick random: random images.
--criterion maskedkd (student last-block CLS attention, default) or rollout (student attention rollout).
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stage0 import common as C
from stage0.attention import AttentionRecorder
from stage0.datasets import Stage0Dataset, build_eval_transform, eval_crop
from stage0.engine import get_device
from stage0.masking import num_keep, topk_idx
from stage0.models import load_finetuned_teacher, load_student_weights, teacher_forward

G = 14  # patch grid


def class_names(ds):
    names = {}
    for rel, label, _g in ds.items:
        names.setdefault(label, Path(rel).parent.name or str(label))   # e.g. images/001.Black_footed_Albatross/x.jpg
    return {k: v.split(".", 1)[-1].replace("_", " ") for k, v in names.items()}


@torch.no_grad()
def analyse(teacher, student, x, k, criterion):
    need = ("cls_last", "rollout")
    with AttentionRecorder(student, need=need) as rs:
        s_logits = student(x)
    S = rs.cls_last if criterion == "maskedkd" else rs.rollout
    with AttentionRecorder(teacher, need=("cls_last",)) as rt:
        t_full = teacher_forward(teacher, x)
    T = rt.cls_last
    idx = topk_idx(S, k)
    t_mask = teacher_forward(teacher, x, idx)
    p_full, p_mask = t_full.float().softmax(1), t_mask.float().softmax(1)
    kl = (p_full * (p_full.clamp_min(1e-12).log() - p_mask.clamp_min(1e-12).log())).sum(1)
    t_idx = topk_idx(T, k)
    overlap = (idx.unsqueeze(2) == t_idx.unsqueeze(1)).any(2).float().mean(1)
    return dict(S=S.float().cpu(), T=T.float().cpu(), idx=idx.cpu(), t_idx=t_idx.cpu(),
                full=t_full.argmax(1).cpu(), mask=t_mask.argmax(1).cpu(), stu=s_logits.argmax(1).cpu(),
                p_true_full=p_full.cpu(), p_true_mask=p_mask.cpu(), kl=kl.cpu(), overlap=overlap.cpu())


def heat(m):
    m = m.view(1, 1, G, G)
    m = F.interpolate(m, size=(C.IMG_SIZE, C.IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    return ((m - m.min()) / (m.max() - m.min() + 1e-12)).numpy()


def keep_view(img, idx):
    mask = torch.zeros(G * G)
    mask[idx] = 1
    mask = mask.view(G, G).repeat_interleave(C.IMG_SIZE // G, 0).repeat_interleave(C.IMG_SIZE // G, 1).numpy()
    return (img * (0.2 + 0.8 * mask[..., None])).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cub")
    ap.add_argument("--run", default="maskedkd_k0.15", help="run folder under {dataset}/ whose student is used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--criterion", choices=("maskedkd", "rollout"), default="maskedkd")
    ap.add_argument("--student-ckpt", default=None, help="default: {run}/seed{seed}/last.pt")
    ap.add_argument("--teacher-ckpt", default=None, help="default: {dataset}/teacher/seed0/best.pt")
    ap.add_argument("--n", type=int, default=6, help="rows in the figure")
    ap.add_argument("--scan", type=int, default=256, help="test images scanned (summary + picking)")
    ap.add_argument("--pick", choices=("mixed", "random"), default="mixed")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--output-root")
    ap.add_argument("--data-root")
    a = ap.parse_args(argv)

    out_root = Path(a.output_root or os.environ.get("OUTPUT_ROOT", "outputs"))
    data_root = Path(a.data_root or os.environ.get("DATA_ROOT", "data"))
    sck = Path(a.student_ckpt) if a.student_ckpt else out_root / a.dataset / a.run / f"seed{a.seed}" / "last.pt"
    tck = Path(a.teacher_ckpt) if a.teacher_ckpt else C.teacher_dir(out_root, a.dataset) / "best.pt"
    out = Path(a.out or Path(os.environ.get("RESULTS_DIR", "results")) / f"viz_{a.run}_s{a.seed}.png")
    device = get_device()
    teacher, _ = load_finetuned_teacher(tck, device)
    student = load_student_weights(sck, device)
    k = num_keep(a.keep)
    print(f"teacher {tck}\nstudent {sck}\nselection {a.criterion}, keep {a.keep:g} (k={k})")

    ds = Stage0Dataset(data_root, a.dataset, "test", build_eval_transform())
    names = class_names(ds)
    order = list(range(len(ds)))
    random.Random(a.sample_seed).shuffle(order)
    order = order[:a.scan]
    res, labels = [], []
    for b in range(0, len(order), a.batch_size):
        ids = order[b:b + a.batch_size]
        x = torch.stack([ds[i][0] for i in ids]).to(device)
        labels += [ds.items[i][1] for i in ids]
        res.append(analyse(teacher, student, x, k, a.criterion))
    R = {key: torch.cat([r[key] for r in res]) for key in res[0]}
    y = torch.tensor(labels)
    agree = (R["mask"] == R["full"]).float()
    print(f"scanned {len(order)} test images: masked-teacher agree {agree.mean() * 100:.1f}%, "
          f"full-teacher acc {(R['full'] == y).float().mean() * 100:.1f}%, "
          f"masked-teacher acc {(R['mask'] == y).float().mean() * 100:.1f}%, "
          f"KL {R['kl'].mean():.3f}, overlap(selected, teacher top-k) {R['overlap'].mean() * 100:.1f}%")

    rng = random.Random(a.sample_seed + 1)
    pos = list(range(len(order)))
    if a.pick == "mixed":
        good = [p for p in pos if agree[p] == 1]
        bad = [p for p in pos if agree[p] == 0]
        nb = min(len(bad), a.n // 2)
        rows = rng.sample(bad, nb) + rng.sample(good, min(len(good), a.n - nb))
    else:
        rows = rng.sample(pos, min(a.n, len(pos)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    crop = eval_crop()
    titles = [f"student attn ({a.criterion})", f"kept {k} tokens ({a.keep:g})", "teacher attn (full)",
              f"teacher top-{k}"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(rows):
        i = order[p]
        with Image.open(ds.root / ds.items[i][0]) as im:
            img = np.asarray(crop(im.convert("RGB")))
        yt, full, mask = labels[p], int(R["full"][p]), int(R["mask"][p])
        pf, pm = R["p_true_full"][p], R["p_true_mask"][p]
        ax = axes[r]
        ax[0].imshow(img)
        ax[0].set_title(f"#{i} true: {names.get(yt, yt)}"[:42], fontsize=8)
        ax[1].imshow(img)
        ax[1].imshow(heat(R["S"][p]), cmap="jet", alpha=0.5)
        ax[2].imshow(keep_view(img, R["idx"][p]))
        ax[2].set_title(f"masked T: {names.get(mask, mask)}"[:40] + f"\np(full pred)={pm[full]:.2f}  KL={R['kl'][p]:.2f}",
                        fontsize=8, color="green" if mask == full else "red")
        ax[3].imshow(img)
        ax[3].imshow(heat(R["T"][p]), cmap="jet", alpha=0.5)
        ax[3].set_title(f"full T: {names.get(full, full)}"[:40] + f"  p={pf[full]:.2f}", fontsize=8,
                        color="black" if full == yt else "red")
        ax[4].imshow(keep_view(img, R["t_idx"][p]))
        ax[4].set_title(f"overlap with kept: {R['overlap'][p] * 100:.0f}%", fontsize=8)
        if r == 0:   # column headers above the first row's titles
            for c, t in enumerate(titles, 1):
                ax[c].set_title(f"[{t}]\n" + ax[c].get_title(), fontsize=8, color=ax[c].title.get_color())
        for c in range(5):
            ax[c].set_xticks([])
            ax[c].set_yticks([])
    fig.suptitle(f"{a.dataset} {a.run}/seed{a.seed}: selection {a.criterion}, keep {a.keep:g} (k={k}). "
                 f"Green = masked teacher keeps the full teacher's prediction, red = changes it.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
