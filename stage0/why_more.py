"""Follow-up to why_collapse.py (Waterbirds, inference only): WHY does MaskedKD's selection pick background, is the
masked teacher's error background-driven, and what do the new criteria (random / sink filter / TrustMask / oracle)
select?  -> $RESULTS_DIR/why_more/{why_more.md, records.csv.gz}

  A. Selection-signal decomposition on the SAME students (Full KD and MaskedKD k0.15 students, final weights), keep k:
       attn      last-block CLS->patch attention, head mean            (= MaskedKD)
       attn_v    last block, sum_h a_h,j * ||v_h,j|| / H                (what the CLS actually reads: weight x value size)
       attn_filt attn with sink tokens (last-block output norm > 3 x image median) taken last   (= mk_filt)
       L0..L11   CLS attention of each layer (head mean)                (where along depth does it drift to background?)
       h0..h2    last-block single heads
       rollout   attention rollout                                      (= rollout criterion)
       random    uniform k
     per score: bird share of the kept patches, masked-teacher P(true) per group, share of kept patches that are sinks,
     bird share of the sink tokens themselves.
  B. Is the masked teacher's error background-driven?  Conflict groups (g1 landbird/water, g2 waterbird/land) vs aligned
     groups (g0, g3), stratified by the bird share of the kept patches. If background drives the target, conflict
     images are wrong far more often than aligned images WITH THE SAME bird share.
  C. Every finished run of the follow-up sweep, re-selected with its own final student and its own criterion
     (maskedkd, rollout, random, mk_filt, tam_r_g0 = TrustMask, oracle_bird, oracle_bg): bird share + teacher target.

Selection here runs on clean eval views (Resize 256 -> CenterCrop 224) in eval mode; training selects on augmented,
mixed views (stated caveat, as in why_collapse). TrustMask's teacher map is the teacher's last-block attention on
the eval view (training reads the cached whole-image map, crop-mapped): same quantity, different view.

    uv run python -m stage0.why_more                       # all parts
    uv run python -m stage0.why_more --parts A B           # before the new runs exist
"""
import argparse
import csv
import gzip
import os
from pathlib import Path

import numpy as np
import torch

from stage0 import common as C
from stage0.attention import attention_probs
from stage0.engine import amp_ctx
from stage0.masking import SINK_RATIO, num_keep
from stage0.models import load_finetuned_teacher, load_student_weights
from stage0.why_collapse import GROUPS, batches, bird_maps, ensure_masks, load_images, p_true, pick_items, run_teacher

CONFLICT, ALIGNED = (1, 2), (0, 3)
BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))
H_BG_GAP = 20.0     # pre-registered: conflict - aligned masked-teacher error >= 20 %p at matched bird share


# --------------------------------------------------------------------------------------- signals ---
@torch.no_grad()
def student_signals(student, X, dev, bs, amp=True):
    """Per image: CLS attention of every layer (L, N, 196) head-mean, last-block per-head (H, N, 196),
    last-block attn x ||v|| (N, 196), last-block output norms (N, 196), rollout (N, 196)."""
    blocks = list(student.blocks)
    L = len(blocks)
    store = {}

    def attn_hook(i):
        def hook(module, inputs, _o):
            x = inputs[0].detach()
            a = attention_probs(module, x, rows=slice(0, 1))[:, :, 0, 1:]            # (B, H, 196)
            store.setdefault("cls", [None] * L)[i] = a.mean(1)
            if i == L - 1:
                B_, N, _ = x.shape
                qkv = module.qkv(x).reshape(B_, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
                vn = qkv[2].float().norm(dim=-1)[:, :, 1:]                                # (B, H, 196)
                store["heads"] = a
                store["attn_v"] = (a * vn).mean(1)
            full = attention_probs(module, x).mean(1)                                   # rollout factor
            a_hat = 0.5 * full + 0.5 * torch.eye(full.shape[-1], device=full.device)
            store["R"] = a_hat if i == 0 else a_hat @ store["R"]
        return hook

    hs = [b.attn.register_forward_hook(attn_hook(i)) for i, b in enumerate(blocks)]
    hs.append(blocks[-1].register_forward_hook(lambda m, i, o: store.__setitem__("norm", o[:, 1:].float().norm(dim=-1))))
    out = {"cls": [], "heads": [], "attn_v": [], "norm": [], "rollout": []}
    try:
        for sl in batches(len(X), bs):
            with amp_ctx(dev, amp):
                student(X[sl].to(dev).float())
            out["cls"].append(torch.stack(store["cls"]).float().cpu())
            out["heads"].append(store["heads"].float().cpu())
            out["attn_v"].append(store["attn_v"].float().cpu())
            out["norm"].append(store["norm"].cpu())
            out["rollout"].append(store["R"][:, 0, 1:].float().cpu())
    finally:
        for h in hs:
            h.remove()
    return {"cls": torch.cat(out["cls"], 1), "heads": torch.cat(out["heads"]), "attn_v": torch.cat(out["attn_v"]),
            "norm": torch.cat(out["norm"]), "rollout": torch.cat(out["rollout"])}


def sinks_of(norm):
    return norm > SINK_RATIO * norm.median(dim=1, keepdim=True).values


def scores_from(sig, k, seed=0):
    """name -> (N, 196) score; top-k of each is the selection."""
    attn = sig["cls"][-1]
    sc = {"attn": attn, "attn_v": sig["attn_v"], "attn_filt": attn - 1e4 * sinks_of(sig["norm"]).float(),
          "rollout": sig["rollout"],
          "random": torch.rand(attn.shape, generator=torch.Generator().manual_seed(seed))}
    sc.update({f"L{l}": sig["cls"][l] for l in range(sig["cls"].shape[0])})
    sc.update({f"h{h}": sig["heads"][:, h] for h in range(sig["heads"].shape[1])})
    return sc


@torch.no_grad()
def teacher_attn(teacher, X, dev, bs, amp=True):
    from stage0.attribution import teacher_attribution
    return torch.cat([teacher_attribution(teacher, X[sl].to(dev).float(), "attn_last", lambda: amp_ctx(dev, amp)).float().cpu()
                      for sl in batches(len(X), bs)])


def run_selection(crit, k, sig, B, T, seed):
    """Re-select like the training criterion `crit` (eval view). B: bird map, T: teacher attention map."""
    if crit == "maskedkd":
        s = sig["cls"][-1]
    elif crit == "mk_filt":
        s = sig["cls"][-1] - 1e4 * sinks_of(sig["norm"]).float()
    elif crit == "rollout":
        s = sig["rollout"]
    elif crit == "random":
        s = torch.rand(B.shape, generator=torch.Generator().manual_seed(seed))
    elif crit in ("oracle_bird", "oracle_bg"):
        b = torch.nan_to_num(B, nan=0.0) + 1e-6 * torch.rand(B.shape, generator=torch.Generator().manual_seed(seed))
        s = b if crit == "oracle_bird" else -b
    elif crit == "tam_r_g0":   # TrustMask: pool = teacher top-2k, then the student's rollout picks k inside it
        pool = torch.topk(T, min(196, 2 * k), dim=1).indices
        pos = torch.topk(torch.gather(sig["rollout"], 1, pool), k, dim=1).indices
        return torch.gather(pool, 1, pos)
    else:
        raise ValueError(crit)
    return torch.topk(s.float(), k, dim=1).indices


# ------------------------------------------------------------------------------------------ main ---
def record(rows, base, sel, B, sinks, pt_mask, pred_mask, y, g, has):
    bsel = torch.gather(torch.nan_to_num(B, nan=0.0), 1, sel).mean(1)
    ssel = torch.gather(sinks.float(), 1, sel).mean(1) if sinks is not None else None
    for i in range(len(y)):
        rows.append({**base, "img": i, "group": int(g[i]), "y": int(y[i]),
                     "p_t_mask": float(pt_mask[i]), "t_correct": int(pred_mask[i] == y[i]),
                     "bird_sel": float(bsel[i]) if has[i] else float("nan"),
                     "sink_sel": float(ssel[i]) if ssel is not None else float("nan")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--per-group", type=int, default=300)
    ap.add_argument("--keep", type=float, default=0.15, help="parts A/B")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", nargs="+",
                    default=["maskedkd:0.15", "rollout:0.15", "rollout:0.1", "rollout:0.05", "random:0.25",
                             "random:0.15", "mk_filt:0.15", "tam_r_g0:0.15", "oracle_bird:0.15", "oracle_bg:0.15"],
                    help="part C: criterion:keep (run dir {criterion}_k{keep})")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seg-root", default=None)
    ap.add_argument("--output-root")
    ap.add_argument("--data-root")
    a = ap.parse_args()
    ds, root, data_root = a.dataset, C.output_root(a.output_root), C.data_root(a.data_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(os.environ.get("RESULTS_DIR", "results")) / "why_more"
    out.mkdir(parents=True, exist_ok=True)
    seg_dir = ensure_masks(a.seg_root or Path(data_root) / "cub_segmentations")
    teacher, _ = load_finetuned_teacher(C.teacher_dir(root, ds) / "best.pt", dev)
    k = num_keep(a.keep)
    rows, sink_rows = [], []
    for split in a.splits:
        img_root, items = pick_items(data_root, ds, split, a.per_group)
        X = load_images(items, img_root)
        y = torch.tensor([it[1] for it in items])
        g = np.array([it[2] for it in items])
        bm = bird_maps(items, img_root, seg_dir)
        has = np.array([m_ is not None for m_ in bm])
        B = torch.stack([m_ if m_ is not None else torch.full((196,), float("nan")) for m_ in bm])
        T = teacher_attn(teacher, X, dev, a.batch_size) if "C" in a.parts else None
        print(f"{split}: {len(items)} images, groups {np.bincount(g, minlength=4).tolist()}, masks {has.sum()}")

        def masked(sel):
            lg = run_teacher(teacher, X, sel, dev, a.batch_size)
            return p_true(lg, y), lg.argmax(1)

        if "A" in a.parts or "B" in a.parts:      # same students, many signals
            for who, rdir in (("KD student", "kd"), ("MaskedKD student", C.mask_dirname("maskedkd", a.keep))):
                for seed in a.seeds:
                    p = Path(root) / ds / rdir / f"seed{seed}" / "last.pt"
                    if not p.exists():
                        continue
                    sig = student_signals(load_student_weights(p, dev), X, dev, a.batch_size)
                    sinks = sinks_of(sig["norm"])
                    bird = torch.nan_to_num(B, nan=0.0)
                    sink_rows.append({"split": split, "student": who, "seed": seed,
                                      "sinks_per_img": float(sinks.float().sum(1).mean()),
                                      "sink_bird": float((bird * sinks).sum() / sinks.sum().clamp_min(1)),
                                      "nonsink_bird": float((bird * ~sinks).sum() / (~sinks).sum().clamp_min(1))})
                    for name, s in scores_from(sig, k, seed).items():
                        sel = torch.topk(s.float(), k, dim=1).indices
                        pt, pr = masked(sel)
                        record(rows, {"part": "A", "split": split, "student": who, "score": name, "keep": a.keep,
                                      "seed": seed}, sel, B, sinks, pt, pr, y, g, has)
                    print(f"  A {split} {who} seed{seed}: done", flush=True)
        if "C" in a.parts:
            for spec in a.runs:
                crit, keep = spec.split(":")
                keep = float(keep)
                kk = num_keep(keep)
                for seed in a.seeds:
                    p = Path(root) / ds / C.mask_dirname(crit, keep) / f"seed{seed}" / "last.pt"
                    if not p.exists():
                        continue
                    sig = student_signals(load_student_weights(p, dev), X, dev, a.batch_size)
                    sel = run_selection(crit, kk, sig, B, T, seed)
                    pt, pr = masked(sel)
                    record(rows, {"part": "C", "split": split, "student": "own", "score": crit, "keep": keep,
                                  "seed": seed}, sel, B, sinks_of(sig["norm"]), pt, pr, y, g, has)
                print(f"  C {split} {spec}: done", flush=True)
    if not rows:
        print("[warn] no runs found; nothing to report")
        return
    with gzip.open(out / "records.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = report(rows, sink_rows, a)
    (out / "why_more.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out / 'why_more.md'}")


# ---------------------------------------------------------------------------------------- report ---
def sel_(rows, **flt):
    return [r for r in rows if all(r[k_] == v for k_, v in flt.items())]


def mean_(rows, key, scale=1.0):
    xs = [r[key] for r in rows if r[key] == r[key]]
    return scale * float(np.mean(xs)) if xs else float("nan")


def f_(x, nd=1):
    return "–" if x != x else f"{x:.{nd}f}"


def wb(rows):
    return [r for r in rows if r["group"] in (2, 3)]


def err(rows):
    return 100.0 * (1 - mean_(rows, "t_correct")) if rows else float("nan")


def report(rows, sink_rows, a):
    sp = a.splits[0]
    md = ["# 후속 분석: 왜 배경을 고르나, teacher 오류는 배경 때문인가, 새 기준은 무엇을 고르나", "",
          f"값은 seed 평균, `{sp}` split (그룹당 최대 {a.per_group}장), clean eval view. "
          "`새 비율` = 남긴 패치 중 새 위 비율 (물새 이미지), `T(gK)` = 남긴 패치만 본 teacher의 그룹 K 정답 확률 (%), "
          "`sink` = 남긴 패치 중 sink 토큰(마지막 블록 출력 norm > 3 × 이미지 중앙값) 비율.", ""]
    names = ["attn", "attn_v", "attn_filt", "rollout", "random"] + [f"h{h}" for h in range(3)] + [f"L{l}" for l in range(12)]
    if any(r["part"] == "A" for r in rows):
        for who in ("KD student", "MaskedKD student"):
            R = sel_(rows, part="A", split=sp, student=who)
            if not R:
                continue
            md += [f"## A. 같은 student, 신호만 바꾸기 ({who}, keep {a.keep:g})", "",
                   "| 점수 | 새 비율 | sink | T(g0) | T(g1) | T(g2) | T(g3) |", "|---|---|---|---|---|---|---|"]
            for n in names:
                S = sel_(R, score=n)
                if S:
                    md.append(f"| {n} | {f_(mean_(wb(S), 'bird_sel'), 2)} | {f_(mean_(S, 'sink_sel'), 2)} | "
                              + " | ".join(f_(mean_(sel_(S, group=i), 'p_t_mask', 100)) for i in range(4)) + " |")
            md.append("")
        if sink_rows:
            md += ["### sink 토큰은 어디에 있나", "", "| student | 이미지당 sink 수 | sink의 새 비율 | 나머지 토큰의 새 비율 |",
                   "|---|---|---|---|"]
            for who in ("KD student", "MaskedKD student"):
                S = [r for r in sink_rows if r["student"] == who and r["split"] == sp]
                if S:
                    md.append(f"| {who} | {np.mean([r['sinks_per_img'] for r in S]):.1f} | "
                              f"{np.mean([r['sink_bird'] for r in S]):.2f} | {np.mean([r['nonsink_bird'] for r in S]):.2f} |")
            md.append("")
    # B. background-driven errors
    md += [f"## B. teacher 오류는 배경 때문인가 ({sp}, keep {a.keep:g}, 마스크된 teacher의 오답률 %)", "",
           "충돌 그룹(g1 육지새/물, g2 물새/땅) vs 일치 그룹(g0, g3), 남긴 패치의 새 비율 구간별. "
           "배경이 target을 정하면 같은 새 비율에서도 충돌 그룹 오답률이 훨씬 높다.", "",
           "| 선택 | 새 비율 구간 | 충돌 n | 충돌 오답률 | 일치 n | 일치 오답률 | 차이 |", "|---|---|---|---|---|---|---|"]
    gaps = {}
    srcs = [("A", "KD student", "attn", "MaskedKD 신호 (KD student)"), ("A", "MaskedKD student", "attn", "MaskedKD (자기 student)"),
            ("A", "KD student", "rollout", "rollout 신호 (KD student)"), ("A", "KD student", "random", "random")]
    for part, who, score, lab in srcs:
        R = sel_(rows, part=part, split=sp, student=who, score=score)
        if not R:
            continue
        for lo, hi in BINS:
            S = [r for r in R if r["bird_sel"] == r["bird_sel"] and lo <= r["bird_sel"] < hi]
            c = [r for r in S if r["group"] in CONFLICT]
            al = [r for r in S if r["group"] in ALIGNED]
            if len(c) >= 10 and len(al) >= 10:
                d = err(c) - err(al)
                gaps.setdefault(lab, []).append(d)
                md.append(f"| {lab} | {lo:.1f}–{min(hi, 1):.1f} | {len(c)} | {f_(err(c))} | {len(al)} | {f_(err(al))} | {d:+.1f} |")
    md += ["", "### 판정 (사전 고정)", ""]
    for lab, ds_ in gaps.items():
        avg = float(np.mean(ds_))
        md.append(f"- **배경 주도 오류** ({lab}): 새 비율 구간 평균 (충돌 − 일치) 오답률 = {avg:+.1f}%p → "
                  + ("**지지**" if avg >= H_BG_GAP else "지지 안 됨") + f" (기준 ≥ {H_BG_GAP:g}%p)")
    if not gaps:
        md.append("- 구간별 표본(≥ 10) 부족 → 판정 없음")
    # C. per-run criteria
    R = sel_(rows, part="C", split=sp)
    if R:
        md += ["", f"## C. 학습된 각 run이 고르는 패치 ({sp}, 자기 student, 자기 기준)", "",
               "| 기준 | keep | n seed | 새 비율 | sink | T(g0) | T(g1) | T(g2) | T(g3) |", "|---|---|---|---|---|---|---|---|---|"]
        for spec in a.runs:
            crit, keep = spec.split(":")
            S = sel_(R, score=crit, keep=float(keep))
            if S:
                md.append(f"| {crit} | {keep} | {len({r['seed'] for r in S})} | {f_(mean_(wb(S), 'bird_sel'), 2)} | "
                          f"{f_(mean_(S, 'sink_sel'), 2)} | "
                          + " | ".join(f_(mean_(sel_(S, group=i), 'p_t_mask', 100)) for i in range(4)) + " |")
    md += ["", "주의: 재선택은 clean eval view·eval mode (학습은 증강·mixup view). TrustMask의 teacher 지도는 eval view에서 "
           "직접 계산 (학습은 캐시를 crop 매핑). oracle은 분할 마스크로 고른 통제 조건이지 방법이 아니다.", ""]
    return md


if __name__ == "__main__":
    main()
