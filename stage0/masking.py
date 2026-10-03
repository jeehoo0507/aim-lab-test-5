"""Teacher input-token selection criteria (Stage 1 / 2). Every criterion returns (B, k) patch indices.

  maskedkd       student last-block CLS->patch attention (head mean), top-k        (MaskedKD losses.py:32)
  random         uniform k of 196, from a dedicated torch.Generator (never the global RNG, so mixup /
                 drop-path streams -- and therefore the pairing with the same-seed KD run -- are unchanged)
  rollout        student attention-rollout CLS row, top-k
  teacher_cache  cached whole-image teacher attribution mapped to the current crop (attribution.py), top-k
  teacher_oracle teacher attribution computed on the current view (Stage 1 upper reference only)
"""
import torch

from stage0.attention import AttentionRecorder

NUM_PATCHES = 196
CRITERIA = ("maskedkd", "random", "rollout", "teacher_cache", "teacher_oracle")
TRAIN_CRITERIA = ("maskedkd", "random")   # Stage 2


def num_keep(keep, n=NUM_PATCHES):
    k = int(round(keep * n))
    assert 1 <= k <= n, (keep, k)
    return k


def topk_idx(scores, k):
    return torch.topk(scores.float(), k, dim=1).indices


def random_idx(batch, k, generator, device, n=NUM_PATCHES):
    """k distinct indices per row, drawn from `generator` (CPU) only."""
    return torch.rand(batch, n, generator=generator).argsort(dim=1)[:, :k].to(device)


class StudentMaskSelector:
    """Training-time selection for Stage 2 (criteria maskedkd / random).

    maskedkd: attach() hooks the student's last block; the attention of the student's own training forward
    (train mode, drop path on -- as in MaskedKD) is then read after each `model(x)` call.
    random:   no hooks; per step generator seeded from (run seed, epoch, step).
    """

    def __init__(self, criterion, keep, student):
        assert criterion in TRAIN_CRITERIA, criterion
        self.criterion, self.k = criterion, num_keep(keep)
        self.recorder = AttentionRecorder(student, need=("cls_last",)) if criterion == "maskedkd" else None

    def __enter__(self):
        if self.recorder is not None:
            self.recorder.__enter__()
        return self

    def __exit__(self, *exc):
        if self.recorder is not None:
            self.recorder.__exit__(*exc)

    def select(self, batch, device, generator=None):
        """Call right after the student forward on the same micro-batch."""
        if self.criterion == "maskedkd":
            scores = self.recorder.cls_last
            assert scores is not None and scores.shape[0] == batch, "student forward must run before select()"
            self.recorder.cls_last = None
            return topk_idx(scores, self.k)
        assert generator is not None
        return random_idx(batch, self.k, generator, device)
