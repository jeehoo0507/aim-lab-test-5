"""Device / AMP helpers and the shared inference loop."""
import contextlib

import torch

from stage0 import common as C


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx(device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_loader(ds, batch_size, num_workers, **kw):
    if "batch_sampler" not in kw:
        kw["batch_size"] = batch_size
    return torch.utils.data.DataLoader(
        ds, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=C.worker_init_fn, persistent_workers=False, **kw)


@torch.no_grad()
def predict(model_fn, loader, device, amp=True):
    """Run model_fn over loader. Returns (logits float32 [N,C] on cpu, labels [N], groups [N])."""
    logits, labels, groups = [], [], []
    for x, y, g in loader:
        x = x.to(device, non_blocking=True)
        with amp_ctx(device, amp):
            out = model_fn(x)
        logits.append(out.float().cpu())
        labels.append(torch.as_tensor(y))
        groups.append(torch.as_tensor(g))
    return torch.cat(logits), torch.cat(labels), torch.cat(groups)


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item() * 100.0
