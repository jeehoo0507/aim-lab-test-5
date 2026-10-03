"""Write synthetic CUB-200-2011 / Waterbirds archives in the official on-disk format.

Only for testing the pipeline on machines without dataset access (no internet). Split sizes match
the official ones so prepare_data.py's checks run unchanged; images are small, class-coloured noise
so models can learn something. Output goes to $DATA_ROOT/{cub/CUB_200_2011, waterbirds/...}.

    DATA_ROOT=smoke/data uv run python tests/make_fake_data.py
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 48


def img(rng, color, place_color=None):
    a = rng.integers(0, 60, (SIZE, SIZE, 3)) + np.array(color)[None, None]
    if place_color is not None:
        a[: SIZE // 3] = rng.integers(0, 60, (SIZE // 3, SIZE, 3)) + np.array(place_color)[None, None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_cub(root, rng):
    base = root / "cub" / "CUB_200_2011"
    if (base / "images.txt").exists():
        return
    (base / "images").mkdir(parents=True, exist_ok=True)
    colors = rng.integers(0, 195, (200, 3))
    n_train = [5994 // 200 + (1 if c < 5994 % 200 else 0) for c in range(200)]
    n_test = [5794 // 200 + (1 if c < 5794 % 200 else 0) for c in range(200)]
    images, labels, split, idx = [], [], [], 1
    for c in range(200):
        d = base / "images" / f"{c + 1:03d}.Fake_{c}"
        d.mkdir(exist_ok=True)
        for k in range(n_train[c] + n_test[c]):
            name = f"Fake_{c}_{k}.jpg"
            img(rng, colors[c]).save(d / name, quality=85)
            images.append(f"{idx} {d.name}/{name}")
            labels.append(f"{idx} {c + 1}")
            split.append(f"{idx} {1 if k < n_train[c] else 0}")
            idx += 1
    for fname, lines in (("images.txt", images), ("image_class_labels.txt", labels),
                         ("train_test_split.txt", split)):
        (base / fname).write_text("\n".join(lines) + "\n")
    (base / "classes.txt").write_text("".join(f"{c + 1} {c + 1:03d}.Fake_{c}\n" for c in range(200)))


def make_waterbirds(root, rng):
    base = root / "waterbirds" / "waterbird_complete95_forest2water2"
    if (base / "metadata.csv").exists():
        return
    # official group counts (y, place): train / val / test
    counts = {0: {(0, 0): 3498, (0, 1): 184, (1, 0): 56, (1, 1): 1057},
              1: {(0, 0): 467, (0, 1): 466, (1, 0): 133, (1, 1): 133},
              2: {(0, 0): 2255, (0, 1): 2255, (1, 0): 642, (1, 1): 642}}
    rows, idx = [], 1
    for split, groups in counts.items():
        for (y, place), n in groups.items():
            d = base / f"{y:03d}.Fake_bird_{y}"
            d.mkdir(parents=True, exist_ok=True)
            for _ in range(n):
                name = f"{d.name}/Fake_{idx}.jpg"
                img(rng, (40 + 120 * y, 80, 90), (60, 150 * place, 200 - 150 * place)).save(base / name, quality=85)
                rows.append({"img_id": idx, "img_filename": name, "y": y, "split": split, "place": place,
                             "place_filename": "/fake.jpg"})
                idx += 1
    with open(base / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    root = Path(os.environ.get("DATA_ROOT") or sys.exit("set DATA_ROOT"))
    rng = np.random.default_rng(0)
    make_cub(root, rng)
    make_waterbirds(root, rng)
    print(f"fake datasets written under {root}")


if __name__ == "__main__":
    main()
