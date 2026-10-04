"""Teacher / student construction and the teacher forward used during KD."""
import timm
import torch

from stage0 import common as C


def create_model(arch, num_classes, pretrained=True, drop_path=C.DROP_PATH):
    """timm DeiT with ImageNet-1k weights (timm default tag *.fb_in1k, fetched via HF Hub / $HF_HOME).

    The classifier head is re-initialised for num_classes (timm drops the 1000-way head when
    num_classes differs), so head init is controlled by the torch seed set before this call.
    """
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)


def create_teacher(num_classes, pretrained=True):
    return create_model(C.TEACHER_ARCH, num_classes, pretrained)


def create_student(num_classes, pretrained=True):
    return create_model(C.STUDENT_ARCH, num_classes, pretrained)


def load_finetuned_teacher(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    teacher = create_teacher(C.NUM_CLASSES[cfg["dataset"]], pretrained=False)
    teacher.load_state_dict(ckpt["model"])
    teacher.to(device).eval().requires_grad_(False)
    return teacher, ckpt


@torch.no_grad()
def teacher_forward(teacher, images, keep_idx=None):
    """Teacher logits for KD.

    keep_idx: None -> all 196 patch tokens (Stage 0 full KD, unchanged). Otherwise a (B, k) long tensor of
    patch-token indices in 0..195: after the positional embedding only the cls token and those patch tokens
    are kept, exactly where MaskedKD gathers them (models_teacher.py:261-266). Token order does not matter
    (positions are already encoded). The caller must have put the teacher in eval() mode.
    """
    assert not teacher.training, "teacher must be in eval() mode during KD"
    x = teacher.patch_embed(images)
    x = teacher._pos_embed(x)          # prepend cls token, add pos embed -> (B, 1 + 196, D)
    if keep_idx is not None:
        assert teacher.num_prefix_tokens == 1, "token selection assumes a single cls token (DeiT, no registers)"
        assert keep_idx.dim() == 2 and keep_idx.shape[0] == x.shape[0], tuple(keep_idx.shape)
        patches = x[:, 1:]
        idx = keep_idx.to(device=x.device, dtype=torch.long).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x[:, :1], torch.gather(patches, 1, idx)], dim=1)
    x = teacher.patch_drop(x)
    x = teacher.norm_pre(x)
    x = teacher.blocks(x)
    x = teacher.norm(x)
    return teacher.forward_head(x)


@torch.no_grad()
def teacher_forward_buckets(teacher, images, buckets):
    """Variable token budget without padding: one teacher_forward per bucket [(rows, keep_idx), ...],
    logits scattered back to the original row order. ViTs have no batch-coupled layers, so each image's
    logits equal its standalone forward."""
    out = None
    for rows, idx in buckets:
        rows = rows.to(images.device)
        logits = teacher_forward(teacher, images[rows], idx)
        if out is None:
            out = logits.new_empty((images.shape[0], logits.shape[1]))
        out[rows] = logits
    return out


def load_student_weights(path, device):
    """Student from a Stage 0/2 run checkpoint: last.pt ({'model', 'config', ...}) or ckpt_e*.pt (state dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    num_classes = state["head.weight"].shape[0]
    student = create_student(num_classes, pretrained=False)
    student.load_state_dict(state)
    return student.to(device).eval()
