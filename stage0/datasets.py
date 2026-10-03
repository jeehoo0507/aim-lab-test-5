"""CUB-200-2011 / Waterbirds datasets and Stage 0 transforms.

Splits are produced once by prepare_data.py (stage0_split.json per dataset) and only read here.
Waterbirds group labels (y x place) are returned for evaluation only; train.py never uses them.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import RandomResizedCropAndInterpolation, str_to_pil_interp
from torchvision import transforms

from stage0 import common as C

EXPECTED_SPLITS = {
    "cub": {"train_full": 5994, "test": 5794},  # train_full = train + val (val = 10% per class)
    "waterbirds": {"train": 4795, "val": 1199, "test": 5794},
}
SPLIT_FILE = "stage0_split.json"


def dataset_dir(data_root, dataset):
    return Path(data_root) / dataset


def load_split(data_root, dataset, split):
    """Returns (image_root, items) with items = list of (relative_path, label, group)."""
    split_path = dataset_dir(data_root, dataset) / SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} missing. Run: uv run python -m stage0.prepare_data --datasets {dataset}")
    meta = C.load_json(split_path)
    if split not in meta["splits"]:
        raise KeyError(f"unknown split {split!r} for {dataset}; have {list(meta['splits'])}")
    root = dataset_dir(data_root, dataset) / meta["image_root"]
    return root, [tuple(x) for x in meta["splits"][split]]


class Stage0Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, dataset, split, transform, limit=None, limit_seed=0):
        self.root, items = load_split(data_root, dataset, split)
        if limit is not None and limit < len(items):
            rng = random.Random(limit_seed)
            items = sorted(rng.sample(items, limit))
        self.items = items
        self.transform = transform
        self.dataset, self.split = dataset, split
        self.aug_seed = None

    def set_aug_seed(self, seed):
        """Seed augmentation per sample from (seed, index): the augmented view of an image depends only
        on the seed (train.py: run seed + epoch), not on num_workers or which worker loads it."""
        self.aug_seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label, group = self.items[i]
        with Image.open(self.root / rel) as im:
            img = im.convert("RGB")
        if self.aug_seed is None:
            return self.transform(img), label, group
        # scoped: the caller's RNG streams (mixup, drop path when num_workers=0) are left untouched
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        s = (self.aug_seed * 1_000_003 + i * 7_919) % (2**31 - 1)
        random.seed(s)
        np.random.seed(s)
        torch.default_generator.manual_seed(s)  # CPU only (RandomErasing runs on CPU)
        try:
            x = self.transform(img)
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
        return x, label, group


def build_train_transform():
    """MaskedKD transforms_imagenet_train with its main.py defaults (transforms_factory.py:56-139).

    RRC(0.08-1, bicubic) -> hflip -> RandAugment(rand-m9-mstd0.5-inc1) -> ToTensor -> Normalize
    -> RandomErasing(0.25, pixel). Mixup/cutmix are batch-level and live in train.py.
    Color jitter is not applied because RandAugment is on (transforms_factory.py:110).
    """
    interp = str_to_pil_interp(C.INTERPOLATION)
    aa_params = dict(
        translate_const=int(C.IMG_SIZE * 0.45),
        img_mean=tuple(min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN),
        interpolation=interp,
    )
    return transforms.Compose([
        RandomResizedCropAndInterpolation(C.IMG_SIZE, scale=C.RRC_SCALE, ratio=C.RRC_RATIO,
                                          interpolation=C.INTERPOLATION),
        transforms.RandomHorizontalFlip(p=C.HFLIP),
        rand_augment_transform(C.RAND_AUGMENT, aa_params),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(C.REPROB, mode=C.REMODE, max_count=C.RECOUNT, num_splits=0, device="cpu"),
    ])


def eval_crop():
    """Resize(256, bicubic) -> CenterCrop(224); shared by clean eval and the corruption cache."""
    return transforms.Compose([
        transforms.Resize(C.RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(C.IMG_SIZE),
    ])


def normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def build_eval_transform():
    return transforms.Compose([eval_crop(), normalize()])
