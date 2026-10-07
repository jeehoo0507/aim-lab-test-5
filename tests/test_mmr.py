"""mk_filt / mk_mmr / mk_fmmr: selection logic and hooks."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0.masking import StudentMaskSelector, mmr_select, topk_idx  # noqa: E402
from stage0.models import create_student  # noqa: E402


def test_mmr_logic():
    torch.manual_seed(0)
    B, n, D, k = 4, 196, 16, 29
    S, H = torch.rand(B, n), torch.randn(B, n, D)
    assert torch.equal(mmr_select(S, H, k, lam=0.0), topk_idx(S, k))           # lam 0 = top-k
    idx = mmr_select(S, H, k, lam=0.5)
    assert idx.shape == (B, k) and all(len(set(r.tolist())) == k for r in idx)  # distinct
    assert torch.equal(idx[:, 0], S.argmax(1))                                  # first = most relevant
    pool = topk_idx(S, 2 * k)
    assert all(set(r.tolist()) <= set(p.tolist()) for r, p in zip(idx, pool))   # only inside the top-2k pool
    # duplicates: 10 identical copies of the best token -> mmr keeps 1, top-k keeps all
    H2, S2 = H.clone(), S.clone()
    H2[:, :10] = H2[:, :1]; S2[:, :10] = 1.0
    assert (mmr_select(S2, H2, k, lam=0.5)[:, :9] < 10).sum(1).max() == 1        # early picks skip the copies
    assert (topk_idx(S2, k) < 10).sum(1).min() == 10
    # sink filter: high-norm tokens with the highest S are not taken while others remain
    H3 = H.clone(); H3[:, :5] *= 100; S3 = S.clone(); S3[:, :5] = 5.0
    assert (mmr_select(S3, H3, k, lam=0.0, sink_ratio=3.0) >= 5).all()
    assert (mmr_select(S3, H3, k, lam=0.5, sink_ratio=3.0) >= 5).all()
    assert (mmr_select(S3, H3, 194, lam=0.0, sink_ratio=3.0) < 5).sum(1).min() == 3  # only if needed


def test_selector_hooks():
    torch.manual_seed(0)
    s = create_student(10, pretrained=False).train()
    x = torch.randn(3, 3, 224, 224)
    torch.manual_seed(1); ref = s(x)
    for c in ("mk_filt", "mk_mmr", "mk_fmmr"):
        with StudentMaskSelector(c, 0.15, s) as sel:
            torch.manual_seed(1); out = s(x)
            idx = sel.select(3, "cpu")
        assert torch.equal(out, ref), c
        assert idx.shape == (3, 29)
        d = sel.pop_diag()
        assert set(d) == set(StudentMaskSelector.DIAG_FIELDS) and sel.pop_diag() == {}
        assert all(0.0 <= d[f] <= 1.0 for f in ("sink_frac_topk", "sink_frac_sel", "overlap_topk"))
        if c == "mk_filt":
            assert d["sink_frac_sel"] == 0.0 or d["sink_frac_sel"] <= d["sink_frac_topk"]
    assert all(len(b.attn._forward_hooks) == 0 and len(b._forward_hooks) == 0 for b in s.blocks)


if __name__ == "__main__":
    test_mmr_logic(); test_selector_hooks(); print("mmr tests passed")
