"""Attention weights from timm ViTs without changing their forward.

timm's Attention uses fused SDPA, so the attention matrix never leaves the module. AttentionRecorder
registers forward hooks on blocks[*].attn, takes the hook's input (the norm1 output the module saw) and
recomputes softmax(q k^T * scale) with the module's own qkv / q_norm / k_norm weights, under no_grad and in
float32. The model's output is not touched. Hooks exist only while the recorder is attached, so runs that
do not need attention pay nothing.

What is kept per forward:
  cls_last : (B, N-1)  last block, head-averaged CLS -> patch attention     (MaskedKD losses.py:32)
  rollout  : (B, N-1)  CLS row of R = prod_l (0.5 * mean_h A_l + 0.5 * I), layer 1 applied first
                       (Abnar & Zuidema 2020; R_l = A_hat_l @ R_{l-1}), patch columns only
"""
import torch


def attention_probs(attn_mod, x, rows=None):
    """softmax(q k^T * scale) of a timm Attention for input x (B, N, C), float32, shape (B, H, R, N).

    rows: None = all query rows, or a slice (e.g. slice(0, 1) = CLS only, much cheaper).
    """
    B, N, _ = x.shape
    qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
    if rows is not None:
        q = q[:, :, rows]
    a = (q.float() * attn_mod.scale) @ k.float().transpose(-2, -1)
    return a.softmax(dim=-1)


class AttentionRecorder:
    """Context manager: with AttentionRecorder(model, need=("cls_last",)) as rec: model(x); rec.cls_last."""

    def __init__(self, model, need=("cls_last",)):
        assert set(need) <= {"cls_last", "rollout"}, need
        self.model, self.need = model, tuple(need)
        self.blocks = list(model.blocks)
        self.handles = []
        self.enabled = True
        self.cls_last = self.rollout = None
        self._R = None

    def __enter__(self):
        last = len(self.blocks) - 1
        layers = range(len(self.blocks)) if "rollout" in self.need else [last]
        for i in layers:
            self.handles.append(self.blocks[i].attn.register_forward_hook(self._hook(i, last)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, i, last):
        def hook(module, inputs, _output):
            if not self.enabled:
                return
            x = inputs[0].detach()
            with torch.no_grad():
                if "rollout" in self.need:
                    a = attention_probs(module, x).mean(1)                    # (B, N, N)
                    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
                    a_hat = 0.5 * a + 0.5 * eye
                    self._R = a_hat if i == 0 else a_hat @ self._R
                    if i == last:
                        self.rollout = self._R[:, 0, 1:]
                        self._R = None
                    if i == last and "cls_last" in self.need:
                        self.cls_last = a[:, 0, 1:]
                elif i == last:
                    self.cls_last = attention_probs(module, x, rows=slice(0, 1)).mean(1)[:, 0, 1:]
        return hook
