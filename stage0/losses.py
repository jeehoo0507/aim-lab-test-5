"""Losses. Soft KD is MaskedKD losses.py:40-49 verbatim:

    KD   = KLDivLoss(reduction='batchmean')(log_softmax(s/T), softmax(t/T)) * T*T
    loss = (1 - alpha) * base + alpha * KD

base is the student's hard-label criterion: CrossEntropy(label_smoothing=0) for the teacher
fine-tune, SoftTargetCrossEntropy on mixup/cutmix targets (label smoothing 0.1 folded in by timm
Mixup) for students -- the same choice MaskedKD main.py:260-268 makes.
"""
import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from stage0 import common as C


def soft_kd_loss(student_logits, teacher_logits, tau):
    s = student_logits.float()
    t = teacher_logits.float()
    return torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(s / tau, dim=1),
                                                     F.softmax(t / tau, dim=1)) * (tau * tau)


class HardCE(torch.nn.Module):
    """Teacher fine-tune criterion. label_smoothing is pinned to 0.0."""

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        assert label_smoothing == 0.0, "label smoothing must be 0.0 for the teacher fine-tune"
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        assert target.dim() == 1, "teacher fine-tune expects hard labels (mixup must be off)"
        return F.cross_entropy(logits.float(), target, label_smoothing=self.label_smoothing)


class SoftCE(torch.nn.Module):
    """Student criterion on mixup targets (smoothing is applied inside timm Mixup)."""

    def __init__(self):
        super().__init__()
        self.fn = SoftTargetCrossEntropy()

    def forward(self, logits, target):
        assert target.dim() == 2, "student criterion expects mixup soft targets"
        return self.fn(logits.float(), target)


class Stage0Loss(torch.nn.Module):
    """Returns (total, base_term, kd_term or None)."""

    def __init__(self, mode, alpha=C.KD_ALPHA, tau=C.KD_TAU):
        super().__init__()
        self.mode, self.alpha, self.tau = mode, alpha, tau
        self.base = HardCE(C.RECIPE["teacher"]["smoothing"]) if mode == "teacher" else SoftCE()

    def forward(self, student_logits, target, teacher_logits=None):
        base = self.base(student_logits, target)
        if self.mode not in C.KD_MODES:
            assert teacher_logits is None
            return base, base, None
        kd = soft_kd_loss(student_logits, teacher_logits, self.tau)
        return (1 - self.alpha) * base + self.alpha * kd, base, kd
