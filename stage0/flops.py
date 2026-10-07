"""Measured GFLOPs (torch.utils.flop_counter.FlopCounterMode, 1 image, CPU, fp32).

Reported in the convention of the DeiT / MaskedKD papers (1 multiply-add = 1 FLOP, DeiT-B = 17.6 G):
FlopCounterMode counts 2 FLOPs per multiply-add, so its total is divided by 2.

FlopCounterMode counts matmul/conv/SDPA ops; elementwise ops (softmax, layernorm, gelu, grid_sample)
are not counted, as is conventional. The non-fused attention path is used for counting so that the
q k^T and attn v products are always counted explicitly (same math as SDPA).
"""
import copy
import functools

import torch
from torch.utils.flop_counter import FlopCounterMode

from stage0 import common as C
from stage0.attention import AttentionRecorder
from stage0.models import create_student, create_teacher, teacher_forward


def _unfused(model):
    m = copy.deepcopy(model).eval()
    for blk in m.blocks:
        blk.attn.fused_attn = False
    return m


def count(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 2 / 1e9


@functools.lru_cache(maxsize=None)
def _models(num_classes):
    return _unfused(create_teacher(num_classes, pretrained=False)), _unfused(create_student(num_classes, pretrained=False))


@torch.no_grad()
def teacher_gflops(k, num_classes=200):
    """Teacher forward with k patch tokens (+ cls)."""
    teacher, _ = _models(num_classes)
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    idx = None if k == 196 else torch.arange(k)[None]
    return count(lambda: teacher_forward(teacher, x, idx))


@torch.no_grad()
def criterion_gflops(criterion, num_classes=200):
    """Extra compute a criterion needs per image, beyond the student forward the training already does."""
    teacher, student = _models(num_classes)
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    if criterion in ("random", "teacher_cache"):
        return 0.0          # random draw / 14x14 map resample + top-k: no counted FLOPs
    if criterion in ("maskedkd", "rollout"):
        need = ("cls_last",) if criterion == "maskedkd" else ("rollout",)
        base = count(lambda: student(x))

        def with_hooks():
            with AttentionRecorder(student, need=need):
                student(x)
        return count(with_hooks) - base
    if criterion == "teacher_oracle":
        def oracle():
            with AttentionRecorder(teacher, need=("cls_last", "rollout")):
                teacher_forward(teacher, x)
        return count(oracle)
    raise ValueError(criterion)


@torch.no_grad()
def selection_gflops(criterion, kind="attn_last", num_classes=200):
    """Per-image selection cost of a training criterion beyond the student / masked-teacher forwards."""
    if criterion in ("maskedkd", "rollout", "random"):
        return criterion_gflops(criterion, num_classes)
    if criterion in ("mk_filt", "mk_mmr", "mk_fmmr"):          # maskedkd hook + k-step greedy on 196 tokens (negligible)
        return criterion_gflops("maskedkd", num_classes)
    if criterion in ("tam_r", "tam_r_var"):                     # S = student rollout (all blocks hooked)
        return criterion_gflops("rollout", num_classes)
    s_cost = criterion_gflops("maskedkd", num_classes)          # S = student last-block CLS attention (hook)
    if criterion in ("tam", "tam_var"):
        return s_cost                                           # + cache lookup / crop resample (no counted FLOPs)
    if criterion == "tam_oracle":
        teacher, _ = _models(num_classes)
        x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
        need = ("cls_last",) if kind == "attn_last" else ("rollout",)

        def oracle():
            with AttentionRecorder(teacher, need=need):
                teacher_forward(teacher, x)
        return s_cost + count(oracle)
    raise ValueError(criterion)
