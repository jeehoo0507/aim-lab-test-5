"""Why does MaskedKD's WGA collapse when keep shrinks, and why does rollout not? (Waterbirds, inference only)

The student always sees the full image; token selection only changes the teacher's target. So the chain to test is
    selection (which patches)  ->  masked-teacher target (per group)  ->  student (per group)
For every finished run (criterion x keep x seed) and checkpoint (e10 / e30 / e60 / last) this script re-selects the
k patches with THAT run's student on clean eval views (Resize 256 -> CenterCrop 224) and records, per image:
  p_t_full  full teacher P(true class)          p_t_mask  teacher P(true class) on the kept patches only
  bird_sel  share of kept patches on the bird   bird_rec  share of the bird's patches that were kept
  p_s       the student's own P(true class) on the full image
Bird masks: CUB-200-2011 segmentations (Waterbirds pastes CUB birds with these masks), pooled to the 14x14 grid.
Control: the Full-KD student selecting with both criteria (same student -> isolates the selection signal from
the training feedback loop). Optional --cka: linear CKA per block (CLS token) student vs teacher and vs KD student.

    uv run python -m stage0.why_collapse --cka           # -> $RESULTS_DIR/why/{why.md, records.csv.gz, *.png}

Caveat (stated in the report): training-time selection runs on augmented, mixed images in train mode (drop path);
here it runs on clean eval views in eval mode.
"""
import argparse
import csv
import gzip
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from stage0 import common as C
from stage0.attention import AttentionRecorder
from stage0.datasets import build_eval_transform, eval_crop, load_split
from stage0.engine import amp_ctx
from stage0.masking import num_keep
from stage0.models import load_finetuned_teacher, load_student_weights, teacher_forward

GROUPS = ("landbird/land", "landbird/water", "waterbird/land", "waterbird/water")
SEG_URL = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
SEG_MD5 = "4d47ba1228eae64f2fa547c47bc65255"
H1_TEACHER_GAP, H2_BIRD_GAP = 10.0, 0.10      # pre-registered thresholds (see verdicts)


# ------------------------------------------------------------------------------------------- data ---
def pick_items(data_root, ds, split, per_group, seed=0):
    root, items = load_split(data_root, ds, split)
    by_g = defaultdict(list)
    for it in items:
        by_g[it[2]].append(it)
    rng = random.Random(seed)
    out = []
    for g in sorted(by_g):
        xs = by_g[g]
        out += sorted(rng.sample(xs, per_group)) if per_group and len(xs) > per_group else xs
    return root, out


def ensure_masks(seg_root):
    """CUB segmentations -> seg_root/segmentations/<class>/<name>.png (downloaded once)."""
    seg_root = Path(seg_root)
    if (seg_root / "segmentations").is_dir():
        return seg_root / "segmentations"
    from stage0.prepare_data import download
    arc = seg_root / "segmentations.tgz"
    if not arc.exists():
        download([SEG_URL], arc, SEG_MD5)
    with tarfile.open(arc) as t:
        t.extractall(seg_root)
    return seg_root / "segmentations"


def bird_maps(items, img_root, seg_dir):
    """(N, 196) bird fraction per patch on the eval view, or None if a mask is missing."""
    crop = eval_crop()
    out, missing, resized = [], 0, 0
    for rel, _, _ in items:
        p = Path(seg_dir) / Path(rel).with_suffix(".png")
        if not p.exists():
            missing += 1
            out.append(None)
            continue
        with Image.open(p) as m, Image.open(img_root / rel) as im:
            m = m.convert("L")
            if m.size != im.size:
                resized += 1
                m = m.resize(im.size, Image.BILINEAR)
        t = transforms.functional.to_tensor(crop(m))[0]                 # (224, 224) in [0, 1]
        out.append(F.avg_pool2d((t > 0.5).float()[None, None], 16).flatten())
    if missing:
        print(f"[warn] {missing}/{len(items)} masks missing -> those images get no bird metrics")
    if resized:
        print(f"[info] {resized} masks resized to their image size")
    return out


def load_images(items, img_root):
    tf = build_eval_transform()
    xs = []
    for rel, _, _ in items:
        with Image.open(img_root / rel) as im:
            xs.append(tf(im.convert("RGB")).half())
    return torch.stack(xs)


# ------------------------------------------------------------------------------------------ models ---
def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


@torch.no_grad()
def run_student(student, X, crits, k_by_crit, dev, bs, amp=True):
    """Student logits on full images + top-k indices per criterion from its own attention (eval mode)."""
    need = tuple(sorted({"cls_last" if c == "maskedkd" else "rollout" for c in crits}))
    logits, idx = [], {c: [] for c in crits}
    with AttentionRecorder(student, need=need) as rec:
        for sl in batches(len(X), bs):
            x = X[sl].to(dev).float()
            with amp_ctx(dev, amp):
                logits.append(student(x).float().cpu())
            for c in crits:
                s = rec.cls_last if c == "maskedkd" else rec.rollout
                idx[c].append(torch.topk(s.float(), k_by_crit[c], dim=1).indices.cpu())
            rec.cls_last = rec.rollout = None
    return torch.cat(logits), {c: torch.cat(v) for c, v in idx.items()}


@torch.no_grad()
def run_teacher(teacher, X, idx, dev, bs, amp=True):
    out = []
    for sl in batches(len(X), bs):
        with amp_ctx(dev, amp):
            out.append(teacher_forward(teacher, X[sl].to(dev).float(), None if idx is None else idx[sl].to(dev))
                       .float().cpu())
    return torch.cat(out)


def p_true(logits, labels):
    return logits.softmax(1).gather(1, labels[:, None])[:, 0]


# ------------------------------------------------------------------------------------------- CKA ---
@torch.no_grad()
def block_cls(model, X, dev, bs, teacher=False, amp=True):
    """(L, N, D) CLS token after every block on full images."""
    feats = [[] for _ in model.blocks]
    hs = [b.register_forward_hook(lambda m, i, o, l=l: feats[l].append(o[:, 0].float().cpu()))
          for l, b in enumerate(model.blocks)]
    try:
        for sl in batches(len(X), bs):
            with amp_ctx(dev, amp):
                (teacher_forward(model, X[sl].to(dev).float()) if teacher else model(X[sl].to(dev).float()))
    finally:
        for h in hs:
            h.remove()
    return [torch.cat(f) for f in feats]


def linear_cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return ((b.T @ a).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm())).item()


# ------------------------------------------------------------------------------------------- main ---
def ckpt_path(rd, ck):
    return rd / ("last.pt" if ck == "last" else f"ckpt_{ck}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--per-group", type=int, default=500, help="images per group per split (0 = all)")
    ap.add_argument("--mkd-keeps", type=float, nargs="+", default=[0.5, 0.3, 0.25, 0.2, 0.15])
    ap.add_argument("--rollout-keeps", type=float, nargs="+", default=[0.3, 0.2, 0.15])
    ap.add_argument("--control-keeps", type=float, nargs="+", default=[0.3, 0.15], help="KD student, both criteria")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpts", nargs="+", default=["e10", "e30", "e60", "last"])
    ap.add_argument("--seg-root", default=None, help="default $DATA_ROOT/cub_segmentations (downloaded if missing)")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--cka", action="store_true")
    ap.add_argument("--cka-n", type=int, default=1000)
    ap.add_argument("--fig-n", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--output-root")
    ap.add_argument("--data-root")
    a = ap.parse_args()
    ds, root, data_root = a.dataset, C.output_root(a.output_root), C.data_root(a.data_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(os.environ.get("RESULTS_DIR", "results")) / "why"
    out.mkdir(parents=True, exist_ok=True)
    seg_dir = None if a.no_masks else ensure_masks(a.seg_root or Path(data_root) / "cub_segmentations")
    teacher, _ = load_finetuned_teacher(C.teacher_dir(root, ds) / "best.pt", dev)

    configs = ([("maskedkd", k, C.mask_dirname("maskedkd", k), "maskedkd") for k in a.mkd_keeps]
               + [("rollout", k, C.mask_dirname("rollout", k), "rollout") for k in a.rollout_keeps]
               + [(c, k, "kd", f"KD student→{c}") for k in a.control_keeps for c in ("maskedkd", "rollout")])
    rows, data = [], {}
    for split in a.splits:
        img_root, items = pick_items(data_root, ds, split, a.per_group)
        X = load_images(items, img_root)
        y = torch.tensor([it[1] for it in items])
        g = np.array([it[2] for it in items])
        bm = bird_maps(items, img_root, seg_dir) if seg_dir else [None] * len(items)
        has = np.array([m is not None for m in bm])
        B = torch.stack([m if m is not None else torch.full((196,), float("nan")) for m in bm])
        pt_full = p_true(run_teacher(teacher, X, None, dev, a.batch_size), y)
        data[split] = dict(items=items, X=X, y=y, g=g, B=B, img_root=img_root)
        print(f"{split}: {len(items)} images, groups {np.bincount(g, minlength=4).tolist()}, masks {has.sum()}")
        for crit, keep, rdir, label in configs:
            k = num_keep(keep)
            for seed in a.seeds:
                rd = Path(root) / ds / rdir / f"seed{seed}"
                cks = ["last"] if rdir == "kd" else a.ckpts
                for ck in cks:
                    p = ckpt_path(rd, ck)
                    if not p.exists():
                        continue
                    st = load_student_weights(p, dev)
                    s_logits, idx = run_student(st, X, [crit], {crit: k}, dev, a.batch_size)
                    sel = idx[crit]
                    pt_mask = p_true(run_teacher(teacher, X, sel, dev, a.batch_size), y)
                    ps = p_true(s_logits, y)
                    bsel = torch.gather(B, 1, sel).mean(1)
                    bird = B > 0.5
                    kept = torch.zeros_like(bird).scatter_(1, sel, True)
                    brec = (bird & kept).sum(1) / bird.sum(1).clamp_min(1)
                    for i in range(len(items)):
                        rows.append({"split": split, "label": label, "criterion": crit, "keep": keep, "seed": seed,
                                     "ckpt": ck, "img": i, "group": int(g[i]), "y": int(y[i]),
                                     "p_t_full": float(pt_full[i]), "p_t_mask": float(pt_mask[i]),
                                     "p_s": float(ps[i]),
                                     "bird_img": float(B[i].nanmean()) if has[i] else float("nan"),
                                     "bird_sel": float(bsel[i]) if has[i] else float("nan"),
                                     "bird_rec": float(brec[i]) if has[i] else float("nan")})
                    if split == "test" and seed == a.seeds[0] and ck == "last":
                        data[split].setdefault("sel", {})[(label, keep)] = sel
                        data[split].setdefault("ptm", {})[(label, keep)] = pt_mask
                    del st
            print(f"  {split} {label} k{keep:g}: done")
    with gzip.open(out / "records.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = report(rows, a)
    if "test" in data and a.fig_n:
        figure(data["test"], out / "patches_g2.png", a.fig_n, seed=0)
        md += ["", "## 그림", "", "`patches_g2.png`: 땅 위의 물새(g2) test 이미지 (무작위 고정 seed 0), "
               "열 = 방법·keep, 밝은 칸 = teacher에 남긴 패치, 초록 윤곽 = 새 마스크, 제목 = masked teacher P(정답)", ""]
    if a.cka and "test" in data:
        md += cka_section(data["test"], root, ds, a, teacher, dev)
    (out / "why.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out / 'why.md'}")


# ----------------------------------------------------------------------------------------- report ---
def agg(rows, **flt):
    sel = [r for r in rows if all(r[k] == v for k, v in flt.items())]
    return sel


def m(rows, key, scale=1.0):
    xs = [r[key] for r in rows if r[key] == r[key]]     # drop NaN
    return scale * float(np.mean(xs)) if xs else float("nan")


def f(x, nd=1):
    return "–" if x != x else f"{x:.{nd}f}"


def wb(rows):
    return [r for r in rows if r["group"] in (2, 3)]


def report(rows, a):
    md = ["# 왜 MaskedKD의 WGA가 무너지는가 (Waterbirds)", "",
          "student는 항상 전체 이미지를 보므로 마스킹이 바꾸는 것은 teacher target뿐이다. 각 run의 student로 패치를 "
          "다시 골라 (clean eval view) teacher target과 새 커버리지를 잰다. 값은 seed 평균.", "",
          "- `T(mask)` = 남긴 패치만 본 teacher의 정답 확률 평균 (%) · `T(full)` = 전체 이미지 teacher",
          "- `새 비율` = 남긴 패치 중 새 위 비율 · `새 회수율` = 새 패치 중 남긴 비율 · `이미지` = 이미지 전체의 새 비율 "
          "(무작위로 고르면 기대되는 값)", ""]
    labels = [("maskedkd", k) for k in a.mkd_keeps] + [("rollout", k) for k in a.rollout_keeps]
    for split in a.splits:
        R = agg(rows, split=split, ckpt="last")
        md += [f"## 1. teacher target ({split}, 최종 체크포인트): 남긴 패치만 본 teacher가 그룹별로 맞히는가", "",
               "| 방법 | keep | " + " | ".join(f"g{i} {GROUPS[i]}" for i in range(4)) + " | 물새 평균 |",
               "|---|---|" + "---|" * 5]
        base = agg(R, label="maskedkd", keep=a.mkd_keeps[0])
        md.append("| 전체 teacher | – | " + " | ".join(f(m(agg(base, group=i), "p_t_full", 100)) for i in range(4))
                  + f" | {f(m(wb(base), 'p_t_full', 100))} |")
        for lab, k in labels:
            S = agg(R, label=lab, keep=k)
            if S:
                md.append(f"| {lab} | {k:g} | " + " | ".join(f(m(agg(S, group=i), "p_t_mask", 100)) for i in range(4))
                          + f" | {f(m(wb(S), 'p_t_mask', 100))} |")
        md += ["", f"## 2. 새 커버리지 ({split}, 최종 체크포인트): 남긴 패치가 새 위에 있는가", "",
               "| 방법 | keep | 새 비율 (물새) | 새 비율 (육지새) | 새 회수율 (물새) | 이미지 기준 (물새) |",
               "|---|---|---|---|---|---|"]
        for lab, k in labels:
            S = agg(R, label=lab, keep=k)
            if S:
                lb = [r for r in S if r["group"] in (0, 1)]
                md.append(f"| {lab} | {k:g} | {f(m(wb(S), 'bird_sel'), 2)} | {f(m(lb, 'bird_sel'), 2)} | "
                          f"{f(m(wb(S), 'bird_rec'), 2)} | {f(m(wb(S), 'bird_img'), 2)} |")
        md.append("")
    # 3. dose-response
    R = [r for r in agg(rows, split=a.splits[0], ckpt="last") if not r["label"].startswith("KD student")]
    md += [f"## 3. 새를 덜 남길수록 teacher가 틀리는가 ({a.splits[0]}, 물새 이미지, 모든 방법·keep 합침)", "",
           "| 새 비율 구간 | 이미지 수 | T(mask) |", "|---|---|---|"]
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        S = [r for r in wb(R) if r["bird_sel"] == r["bird_sel"] and lo <= r["bird_sel"] < hi]
        md.append(f"| {lo:.1f}–{min(hi, 1):.1f} | {len(S)} | {f(m(S, 'p_t_mask', 100))} |")
    # 4. trajectory
    md += ["", f"## 4. 학습 중 변화 ({a.splits[0]}, keep {min(a.mkd_keeps):g}, 물새 이미지): 되먹임이 있는가", "",
           "| 방법 | 체크포인트 | 새 비율 | T(mask) | student P(정답) |", "|---|---|---|---|---|"]
    kmin = min(a.mkd_keeps)
    for lab in ("maskedkd", "rollout"):
        for ck in a.ckpts:
            S = wb(agg(rows, split=a.splits[0], label=lab, keep=kmin, ckpt=ck))
            if S:
                md.append(f"| {lab} | {ck} | {f(m(S, 'bird_sel'), 2)} | {f(m(S, 'p_t_mask', 100))} | {f(m(S, 'p_s', 100))} |")
    # 5. control
    md += ["", f"## 5. 같은 student로 기준만 바꾸면 ({a.splits[0]}, Full KD student, 물새 이미지): 신호 자체의 차이", "",
           "| 기준 | keep | 새 비율 | T(mask) |", "|---|---|---|---|"]
    for k in a.control_keeps:
        for c in ("maskedkd", "rollout"):
            S = wb(agg(rows, split=a.splits[0], label=f"KD student→{c}", keep=k))
            if S:
                md.append(f"| {c} | {k:g} | {f(m(S, 'bird_sel'), 2)} | {f(m(S, 'p_t_mask', 100))} |")
    # 6. verdicts
    md += ["", "## 6. 판정 (사전 고정)", ""]
    sp = a.splits[0]
    mk = wb(agg(rows, split=sp, label="maskedkd", keep=kmin, ckpt="last"))
    ro = wb(agg(rows, split=sp, label="rollout", keep=kmin, ckpt="last"))
    if mk and ro:
        dt = m(ro, "p_t_mask", 100) - m(mk, "p_t_mask", 100)
        db = m(ro, "bird_sel") - m(mk, "bird_sel")
        md.append(f"- **H1 teacher target 손상** (물새 {sp} 이미지, keep {kmin:g}): rollout − MaskedKD T(mask) = "
                  f"{dt:+.1f}%p → " + ("**지지**" if dt >= H1_TEACHER_GAP else "지지 안 됨")
                  + f" (기준 ≥ {H1_TEACHER_GAP:g}%p)")
        md.append(f"- **H2 새 증거 손실**: rollout − MaskedKD 새 비율 = {f(db, 2)} → "
                  + ("**지지**" if db == db and db >= H2_BIRD_GAP else ("마스크 없음" if db != db else "지지 안 됨"))
                  + f" (기준 ≥ {H2_BIRD_GAP:g})")
    cm = wb(agg(rows, split=sp, label="KD student→maskedkd", keep=kmin))
    cr = wb(agg(rows, split=sp, label="KD student→rollout", keep=kmin))
    if cm and cr and mk:
        dsig = m(cr, "bird_sel") - m(cm, "bird_sel")
        dloop = m(cm, "bird_sel") - m(mk, "bird_sel")
        md.append(f"- 원인 분리 (새 비율): 같은 KD student에서 rollout − maskedkd = {f(dsig, 2)} (선택 신호 차이), "
                  f"KD student→maskedkd − MaskedKD 자기 student = {f(dloop, 2)} (학습 되먹임)")
    return md


def figure(d, path, n, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    sels = d.get("sel", {})
    if not sels:
        return
    cols = [c for c in (("maskedkd", 0.3), ("maskedkd", 0.15), ("rollout", 0.15), ("KD student→maskedkd", 0.15),
                        ("KD student→rollout", 0.15)) if c in sels]
    g2 = [i for i in range(len(d["items"])) if d["g"][i] == 2]
    pick = sorted(random.Random(seed).sample(g2, min(n, len(g2))))
    mean, std = torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None], torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None]
    fig, axes = plt.subplots(len(pick), 1 + len(cols), figsize=(2.2 * (1 + len(cols)), 2.3 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        img = (d["X"][i].float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        bird = d["B"][i].reshape(14, 14).numpy()
        for c in range(1 + len(cols)):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                ax.imshow(img)
                ax.set_title("image (waterbird/land)", fontsize=7)
            else:
                lab, k = cols[c - 1]
                keep = np.zeros(196)
                keep[sels[(lab, k)][i].numpy()] = 1
                shade = np.kron(keep.reshape(14, 14), np.ones((16, 16)))[..., None]
                ax.imshow(img * (0.25 + 0.75 * shade))
                ax.set_title(f"{lab} k{k:g}\nT(mask) {100 * float(d['ptm'][(lab, k)][i]):.0f}%", fontsize=7)
            if not np.isnan(bird).all():
                ax.contour(np.kron(bird, np.ones((16, 16))), levels=[0.5], colors="lime", linewidths=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cka_section(d, root, ds, a, teacher, dev):
    idx = list(range(len(d["items"])))
    random.Random(0).shuffle(idx)
    idx = sorted(idx[: a.cka_n])
    X, g = d["X"][idx], d["g"][idx]
    T = block_cls(teacher, X, dev, a.batch_size, teacher=True)
    runs = [("kd", "kd")] + [(C.mask_dirname("maskedkd", min(a.mkd_keeps)), "maskedkd"),
                             (C.mask_dirname("rollout", min(a.rollout_keeps)), "rollout")]
    feats = {}
    for rdir, lab in runs:
        for s in a.seeds:
            p = Path(root) / ds / rdir / f"seed{s}" / "last.pt"
            if p.exists():
                feats[(lab, s)] = block_cls(load_student_weights(p, dev), X, dev, a.batch_size)
    L = len(T)
    wbm = np.isin(g, (2, 3))
    md = ["", f"## 7. 층별 표현 비교 (linear CKA, CLS 토큰, test {len(idx)}장 / 물새 {wbm.sum()}장, seed 평균)", "",
          "student 블록 l vs teacher 블록 l, 그리고 vs Full KD student 블록 l. 낮을수록 표현이 다르다.", "",
          "| 블록 | KD↔T | MaskedKD↔T | rollout↔T | MaskedKD↔KD | rollout↔KD | MaskedKD↔KD (물새) | rollout↔KD (물새) |",
          "|---|---|---|---|---|---|---|---|"]
    for l in range(L):
        def mean_cka(lab, other, mask=None):
            v = []
            for s in a.seeds:
                if (lab, s) not in feats or (other != "T" and (other, s) not in feats):
                    continue
                x = feats[(lab, s)][l]
                y = T[l] if other == "T" else feats[(other, s)][l]
                if mask is not None:
                    x, y = x[mask], y[mask]
                v.append(linear_cka(x, y))
            return f"{np.mean(v):.3f}" if v else "–"
        md.append(f"| {l} | {mean_cka('kd', 'T')} | {mean_cka('maskedkd', 'T')} | {mean_cka('rollout', 'T')} | "
                  f"{mean_cka('maskedkd', 'kd')} | {mean_cka('rollout', 'kd')} | "
                  f"{mean_cka('maskedkd', 'kd', wbm)} | {mean_cka('rollout', 'kd', wbm)} |")
    return md


if __name__ == "__main__":
    main()
