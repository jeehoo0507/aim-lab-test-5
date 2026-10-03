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
def teacher_forward(teacher, images):
    """Teacher logits for KD. Full input (all 196 patch tokens) in Stage 0.

    Spelled out step by step (same ops as timm VisionTransformer.forward) so that a later stage can
    insert patch-token selection between the positional embedding and the transformer blocks, as
    MaskedKD does (models_teacher.py:261-266): keep cls token + gather the chosen patch tokens.
    The caller must have put the teacher in eval() mode; inputs are the student's (mixed) batch.
    """
    assert not teacher.training, "teacher must be in eval() mode during KD"
    x = teacher.patch_embed(images)
    x = teacher._pos_embed(x)          # prepend cls token, add pos embed -> (B, 1 + 196, D)
    # --- token selection goes here in later stages (x = x[:, [0] + keep]) ---
    x = teacher.patch_drop(x)
    x = teacher.norm_pre(x)
    x = teacher.blocks(x)
    x = teacher.norm(x)
    return teacher.forward_head(x)
