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
    "imagenet100": {"train": 117000, "val": 13000, "test": 5000},  # HF ilee0022/ImageNet100
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


class _RecordingRRC(RandomResizedCropAndInterpolation):
    """timm RRC that also records the crop box. Same RNG calls and same output as the parent."""

    last = None

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        interpolation = self.interpolation
        if isinstance(interpolation, (tuple, list)):
            interpolation = random.choice(interpolation)
        W, H = img.size
        self.last = (i, j, h, w, W, H)
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, interpolation)


class _RecordingFlip(transforms.RandomHorizontalFlip):
    """torchvision RandomHorizontalFlip that records whether it flipped. Same RNG call and output."""

    last = None

    def forward(self, img):
        self.last = bool(torch.rand(1) < self.p)
        return transforms.functional.hflip(img) if self.last else img


class TrainTransformWithBox:
    """build_train_transform() that returns (image, box). box = float32 [i, j, h, w, flip, W, H]:
    the RandomResizedCrop box in original-image pixels (top, left, height, width), whether the crop
    was flipped, and the original size. The image is bit-identical to build_train_transform()'s."""

    def __init__(self, ops):
        self.ops = ops
        self.rrc, self.flip = ops[0], ops[1]
        assert isinstance(self.rrc, _RecordingRRC) and isinstance(self.flip, _RecordingFlip)

    def __call__(self, img):
        for op in self.ops:
            img = op(img)
        i, j, h, w, W, H = self.rrc.last
        return img, np.array([i, j, h, w, float(self.flip.last), W, H], dtype=np.float32)


class WithIndex(torch.utils.data.Dataset):
    """Wraps a Stage0Dataset whose transform returns (image, box); yields ((image, box, index), label,
    group) so a training batch can look up per-image data (e.g. the attribution cache) by index.
    Augmentation (and therefore the image) is exactly the wrapped dataset's."""

    def __init__(self, ds):
        self.ds, self.items = ds, ds.items

    def set_aug_seed(self, seed):
        self.ds.set_aug_seed(seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        (img, box), label, group = self.ds[i]
        return (img, box, i), label, group


def build_train_transform(return_box=False):
    """MaskedKD transforms_imagenet_train with its main.py defaults (transforms_factory.py:56-139).

    RRC(0.08-1, bicubic) -> hflip -> RandAugment(rand-m9-mstd0.5-inc1) -> ToTensor -> Normalize
    -> RandomErasing(0.25, pixel). Mixup/cutmix are batch-level and live in train.py.
    Color jitter is not applied because RandAugment is on (transforms_factory.py:110).
    return_box=True: TrainTransformWithBox (same image + crop box / flip), used by Stage 1 / attribution.
    """
    interp = str_to_pil_interp(C.INTERPOLATION)
    aa_params = dict(
        translate_const=int(C.IMG_SIZE * 0.45),
        img_mean=tuple(min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN),
        interpolation=interp,
    )
    rrc_cls, flip_cls = (_RecordingRRC, _RecordingFlip) if return_box else \
        (RandomResizedCropAndInterpolation, transforms.RandomHorizontalFlip)
    ops = [
        rrc_cls(C.IMG_SIZE, scale=C.RRC_SCALE, ratio=C.RRC_RATIO, interpolation=C.INTERPOLATION),
        flip_cls(p=C.HFLIP),
        rand_augment_transform(C.RAND_AUGMENT, aa_params),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        RandomErasing(C.REPROB, mode=C.REMODE, max_count=C.RECOUNT, num_splits=0, device="cpu"),
    ]
    return TrainTransformWithBox(ops) if return_box else transforms.Compose(ops)


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
