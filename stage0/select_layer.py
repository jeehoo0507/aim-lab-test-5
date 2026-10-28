"""Pick the student layer L* for the mk_L criterion on the VALIDATION split (never test), before any mk_L training.

Rule (fixed before the runs): with the Full-KD students (seeds given), keep k, clean val views, select the top-k
patches by the CLS attention of each layer l; score(l) = worst-group mean masked-teacher P(true) (the val analogue of
WGA), averaged over seeds. L* = argmax score(l); ties -> the deeper layer. Also reported: bird share per layer.
  uv run python -m stage0.select_layer          # -> $RESULTS_DIR/why_more/layer_select.{json,md}; prints L*
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from stage0 import common as C
from stage0.masking import num_keep
from stage0.models import load_finetuned_teacher, load_student_weights
from stage0.why_collapse import bird_maps, ensure_masks, load_images, p_true, pick_items, run_teacher
from stage0.why_more import student_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="waterbirds")
    ap.add_argument("--split", default="val")
    ap.add_argument("--keep", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-group", type=int, default=0, help="0 = all val images")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seg-root", default=None)
    ap.add_argument("--output-root")
    ap.add_argument("--data-root")
    a = ap.parse_args()
    assert a.split != "test", "layer selection must not look at test"
    ds, root, data_root = a.dataset, C.output_root(a.output_root), C.data_root(a.data_root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(os.environ.get("RESULTS_DIR", "results")) / "why_more"
    out.mkdir(parents=True, exist_ok=True)
    teacher, _ = load_finetuned_teacher(C.teacher_dir(root, ds) / "best.pt", dev)
    img_root, items = pick_items(data_root, ds, a.split, a.per_group)
    X = load_images(items, img_root)
    y = torch.tensor([it[1] for it in items])
    g = np.array([it[2] for it in items])
    seg = ensure_masks(a.seg_root or Path(data_root) / "cub_segmentations")
    bm = bird_maps(items, img_root, seg)
    B = torch.stack([m if m is not None else torch.full((196,), float("nan")) for m in bm])
    k = num_keep(a.keep)
    per = {}                                  # layer -> list over seeds of (worst-group T, group Ts, bird share)
    for s in a.seeds:
        p = Path(root) / ds / "kd" / f"seed{s}" / "last.pt"
        if not p.exists():
            continue
        sig = student_signals(load_student_weights(p, dev), X, dev, a.batch_size)
        for l in range(sig["cls"].shape[0]):
            sel = torch.topk(sig["cls"][l].float(), k, dim=1).indices
            pt = p_true(run_teacher(teacher, X, sel, dev, a.batch_size), y).numpy()
            gt = [float(pt[g == i].mean()) * 100 for i in range(4)]
            wbm = np.isin(g, (2, 3))
            bird = float(np.nanmean(torch.gather(torch.nan_to_num(B, nan=0.0), 1, sel).mean(1).numpy()[wbm]))
            per.setdefault(l, []).append((min(gt), gt, bird))
        print(f"  seed {s}: done", flush=True)
    if not per:
        raise SystemExit("[error] no Full-KD student found")
    rows = {l: {"worst": float(np.mean([v[0] for v in vs])), "groups": np.mean([v[1] for v in vs], 0).tolist(),
                "bird": float(np.mean([v[2] for v in vs]))} for l, vs in per.items()}
    best = max(rows, key=lambda l: (round(rows[l]["worst"], 6), l))
    md = [f"# 층 선택 (val, Full KD student, keep {a.keep:g}, seed {a.seeds})", "",
          "규칙(사전 고정): 각 층 CLS attention으로 top-k → 마스크된 teacher의 그룹별 정답 확률 중 최저(val WGA 대응)가 "
          "가장 높은 층. test는 보지 않음.", "",
          "| 층 | 최저 그룹 T | T(g0) | T(g1) | T(g2) | T(g3) | 새 비율 (물새) |", "|---|---|---|---|---|---|---|"]
    for l in sorted(rows):
        r = rows[l]
        md.append(f"| {'**L%d**' % l if l == best else 'L%d' % l} | {r['worst']:.1f} | "
                  + " | ".join(f"{x:.1f}" for x in r["groups"]) + f" | {r['bird']:.2f} |")
    md += ["", f"**L\\* = {best}** (마지막 층 L{max(rows)} = MaskedKD)", ""]
    (out / "layer_select.md").write_text("\n".join(md) + "\n")
    C.save_json(out / "layer_select.json", {"layer": best, "keep": a.keep, "split": a.split, "seeds": a.seeds, "rows": rows})
    print("\n".join(md))
    print(f"LAYER={best}")


if __name__ == "__main__":
    main()
