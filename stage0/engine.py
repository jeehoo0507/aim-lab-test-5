"""Device / AMP helpers and the shared inference loop."""
import contextlib

import numpy as np
import torch

from stage0 import common as C


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx(device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def numpy_collate(batch):
    """Collate (image, label, group) samples into numpy arrays inside the worker.

    DataLoader workers hand torch tensors to the main process through /dev/shm. Containers often
    mount only 64 MB there (Docker / Kubernetes default), and one batch of 128 RGB 224x224 float32
    images is 77 MB -> "Bus error ... out of shared memory". numpy arrays are pickled through the
    worker's pipe instead, so /dev/shm size no longer matters. Values are bit-identical.
    """
    xs, ys, gs = zip(*batch)
    x = np.stack([t.numpy() if torch.is_tensor(t) else np.asarray(t) for t in xs])
    return x, np.asarray(ys, dtype=np.int64), np.asarray(gs, dtype=np.int64)


class TensorBatches:
    """Wraps a DataLoader built with numpy_collate and yields torch tensors (zero-copy)."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield tuple(torch.from_numpy(a) for a in batch)


def numpy_collate_box(batch):
    """numpy_collate for datasets whose transform returns (image, box): yields x, y, g, box."""
    xs, ys, gs = zip(*batch)
    x = np.stack([t[0].numpy() for t in xs])
    box = np.stack([np.asarray(t[1], dtype=np.float32) for t in xs])
    return x, np.asarray(ys, dtype=np.int64), np.asarray(gs, dtype=np.int64), box


def numpy_collate_box_index(batch):
    """numpy_collate for WithIndex datasets: yields x, y, g, box, index."""
    xs, ys, gs = zip(*batch)
    x = np.stack([t[0].numpy() for t in xs])
    box = np.stack([np.asarray(t[1], dtype=np.float32) for t in xs])
    idx = np.asarray([t[2] for t in xs], dtype=np.int64)
    return x, np.asarray(ys, dtype=np.int64), np.asarray(gs, dtype=np.int64), box, idx


def make_loader(ds, batch_size, num_workers, collate_fn=numpy_collate, **kw):
    if "batch_sampler" not in kw:
        kw["batch_size"] = batch_size
    return TensorBatches(torch.utils.data.DataLoader(
        ds, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn,
        worker_init_fn=C.worker_init_fn, persistent_workers=False, **kw))


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
