"""Teacher input-token selection criteria (Stage 1 / 2). Every criterion returns (B, k) patch indices.

  maskedkd       student last-block CLS->patch attention (head mean), top-k        (MaskedKD losses.py:32)
  random         uniform k of 196, from a dedicated torch.Generator (never the global RNG, so mixup /
                 drop-path streams -- and therefore the pairing with the same-seed KD run -- are unchanged)
  rollout        student attention-rollout CLS row, top-k
  teacher_cache  cached whole-image teacher attribution mapped to the current crop (attribution.py), top-k
  teacher_oracle teacher attribution computed on the current view (Stage 1 upper reference only)
  tam            Teachability-Aware Masking (Stage 3 pilot): teacher candidate pool from the cached teacher
                 attribution T (crop-mapped), split into "shared" tokens (high student attention S) and
                 "gap" tokens (high T, lower S); gap share g follows a curriculum. k tokens per image.
  tam_var        tam + per-image budget: 3 buckets (k(1-delta), k, k(1+delta)) by teacher concentration,
                 mean k per image; the teacher runs once per bucket (no padding / attention masks).
  tam_oracle     tam with T from a full teacher forward on the current (pre-mixup) view instead of the cache
                 (diagnostic upper bound; costs more than full KD).
Training criteria (Stage 2 / 3 pilot): maskedkd, random, rollout (student rollout, Stage 1 criterion), tam,
tam_var, tam_oracle.
"""
import torch

from stage0.attention import AttentionRecorder

NUM_PATCHES = 196
CRITERIA = ("maskedkd", "random", "rollout", "teacher_cache", "teacher_oracle")
TRAIN_CRITERIA = ("maskedkd", "random", "rollout", "tam", "tam_var", "tam_oracle")   # Stage 2 + Stage 3 pilot
STAGE2_CRITERIA = ("maskedkd", "random")            # default of `scheduler --stage 2` (unchanged)
STUDENT_CRITERIA = ("maskedkd", "random", "rollout")  # StudentMaskSelector
TAM_CRITERIA = ("tam", "tam_var", "tam_oracle")
TAM_CACHE_CRITERIA = ("tam", "tam_var")             # T from the attribution cache (tam_oracle: from the teacher)
TAM_GAP = (0.1, 0.5)      # g0 -> g1 linearly over training
TAM_DELTA = 0.33


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
        assert criterion in STUDENT_CRITERIA, criterion
        self.criterion, self.k = criterion, num_keep(keep)
        self.recorder = AttentionRecorder(student, need=("cls_last",)) if criterion == "maskedkd" else None
        if criterion == "rollout":   # all blocks hooked: rollout of the student's training forward
            self.recorder = AttentionRecorder(student, need=("rollout",))

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
        if self.criterion == "rollout":
            scores = self.recorder.rollout
            assert scores is not None and scores.shape[0] == batch, "student forward must run before select()"
            self.recorder.rollout = None
            return topk_idx(scores, self.k)
        assert generator is not None
        return random_idx(batch, self.k, generator, device)


# ------------------------------------------------------------------------------------------- TAM ---
def tam_gap_ratio(epoch, epochs, g0, g1):
    """g(epoch) = g0 + (g1 - g0) * epoch / (epochs - 1)  (epoch is 0-based)."""
    return g0 if epochs <= 1 else g0 + (g1 - g0) * epoch / (epochs - 1)


def tam_select(T, S, k, g):
    """T, S: (B, 196) teacher attribution (crop-mapped) and student CLS attention. Returns (B, k).

    pool   = top m = min(196, 2k) tokens by T (the rest is pruned: the teacher does not rely on it)
    shared = top round((1-g) k) pool tokens by S
    gap    = top (k - k_shared) remaining pool tokens by T (teacher-important, student-underattended)
    """
    B, n = T.shape
    m = min(n, 2 * k)
    k_shared = int(round((1 - g) * k))
    k_gap = k - k_shared
    T, S = T.float(), S.float()
    pool = torch.topk(T, m, dim=1).indices                                     # (B, m)
    shared_pos = torch.topk(torch.gather(S, 1, pool), k_shared, dim=1).indices  # positions within pool
    t_pool = torch.gather(T, 1, pool).scatter(1, shared_pos, float("-inf"))
    gap_pos = torch.topk(t_pool, k_gap, dim=1).indices
    return torch.gather(pool, 1, torch.cat([shared_pos, gap_pos], dim=1))


def tam_bucket_ks(k, delta, n=NUM_PATCHES):
    return max(1, int(round(k * (1 - delta)))), k, min(n, int(round(k * (1 + delta))))


def tam_buckets(T, k, delta):
    """Split a micro-batch into 3 budget buckets by teacher concentration c = mass of the top-k tokens of
    T normalised to sum 1. Most concentrated third -> k_lo, least concentrated third -> k_hi, rest
    (incl. the remainder of B mod 3) -> k. Returns [(rows LongTensor, k_b), ...] for non-empty buckets."""
    B = T.shape[0]
    t = T.float().clamp_min(0)
    t = t / t.sum(1, keepdim=True).clamp_min(1e-12)
    c = torch.topk(t, k, dim=1).values.sum(1)
    order = torch.argsort(c, descending=True, stable=True)
    n3 = B // 3
    k_lo, k_mid, k_hi = tam_bucket_ks(k, delta)
    groups = [(order[:n3], k_lo), (order[n3:B - n3], k_mid), (order[B - n3:], k_hi)]
    out = [(rows, kb) for rows, kb in groups if rows.numel() > 0]
    mean_k = sum(rows.numel() * kb for rows, kb in out) / B
    assert abs(mean_k - k) <= 1.0, (mean_k, k, B)
    return out


class TamSelector:
    """Training-time TAM selection. Hooks the student's last block (same signal as maskedkd).

    select(batch, T) -> (B, k) indices (tam, tam_oracle) or [(rows, (n_b, k_b) indices), ...] (tam_var).
    T: (B, 196) teacher attribution for the micro-batch's pre-mixup view: crop-mapped cache (tam, tam_var) or a
    teacher forward on that view (tam_oracle).
    """

    def __init__(self, criterion, keep, student, gap=TAM_GAP, delta=TAM_DELTA):
        assert criterion in TAM_CRITERIA, criterion
        self.criterion, self.k = criterion, num_keep(keep)
        self.g0, self.g1 = gap
        self.delta = delta
        self.g = self.g0
        self.recorder = AttentionRecorder(student, need=("cls_last",))

    def set_epoch(self, epoch, epochs):
        self.g = tam_gap_ratio(epoch, epochs, self.g0, self.g1)

    def __enter__(self):
        self.recorder.__enter__()
        return self

    def __exit__(self, *exc):
        self.recorder.__exit__(*exc)

    def select(self, batch, T):
        S = self.recorder.cls_last
        assert S is not None and S.shape[0] == batch == T.shape[0], "student forward must run before select()"
        self.recorder.cls_last = None
        T = T.to(S.device)
        if self.criterion in ("tam", "tam_oracle"):
            return tam_select(T, S, self.k, self.g)
        return [(rows, tam_select(T[rows], S[rows], kb, self.g)) for rows, kb in tam_buckets(T, self.k, self.delta)]


# ---------------------------------------------------------------------- TAM under mixup / cutmix ---
from timm.data import Mixup  # noqa: E402
from timm.data.mixup import cutmix_bbox_and_lam  # noqa: E402


class RecordingMixup(Mixup):
    """timm Mixup (batch mode) that also records what it did in self.last = (lam, (yl, yh, xl, xh) or None).
    _mix_batch is timm 1.0's code line for line, so the RNG draws and the mixed images are identical."""

    last = (1.0, None)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            self.last = (1.0, None)
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
            self.last = (float(lam), (int(yl), int(yh), int(xl), int(xh)))
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
            self.last = (float(lam), None)
        return lam

    def __call__(self, x, target):
        assert self.mode == "batch", "RecordingMixup only records batch mode"
        return super().__call__(x, target)


def mix_attribution(T, mix, img_size=224, patch=16):
    """Teacher attribution of the MIXED batch from the per-view maps T (B, 196) of the unmixed views, following
    timm batch mixup (partner of row i = row B-1-i). Rows are normalised to sum 1 first.
      mixup : lam * T + (1 - lam) * T.flip(0)
      cutmix: per patch cell, the area fraction f inside the pasted box takes the partner's map:
              (1 - f) * T + f * T.flip(0)"""
    lam, box = mix
    T = T.float().clamp_min(0)
    T = T / T.sum(1, keepdim=True).clamp_min(1e-12)
    if lam == 1.0:
        return T
    Tf = T.flip(0)
    if box is None:
        return lam * T + (1 - lam) * Tf
    yl, yh, xl, xh = box
    g = img_size // patch
    lo = torch.arange(g, dtype=torch.float32, device=T.device) * patch
    fy = ((torch.minimum(lo + patch, torch.tensor(float(yh))) - torch.maximum(lo, torch.tensor(float(yl))))
          .clamp_min(0) / patch)
    fx = ((torch.minimum(lo + patch, torch.tensor(float(xh))) - torch.maximum(lo, torch.tensor(float(xl))))
          .clamp_min(0) / patch)
    f = (fy[:, None] * fx[None, :]).flatten()                                  # (196,)
    return (1 - f) * T + f * Tf
