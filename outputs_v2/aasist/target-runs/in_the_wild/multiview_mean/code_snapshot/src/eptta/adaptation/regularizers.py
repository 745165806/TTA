"""Mutually exclusive source/parameter regularizers for mechanism tests."""
import torch
import torch.nn.functional as functional

from eptta.adaptation.math import apply_adapter, keep_loss


def regularizer(name, R, resources, gamma, fisher=None):
    if name == "none":
        return R.sum() * 0
    if name == "margin":
        adapted = apply_adapter(resources.anchors_z, resources.U, R)
        return keep_loss(adapted, resources.w, resources.b, resources.anchors_y,
                         resources.anchors_m0, resources.tau0, gamma)
    if name == "parameter_l2":
        return R.square().mean()
    if name == "source_logit":
        score = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b
        return (score - resources.anchors_s0).square().mean()
    if name == "fisher":
        if fisher is None or fisher.shape != R.shape or fisher.device != R.device or fisher.dtype != R.dtype:
            raise ValueError("Fisher regularizer requires a matching frozen tensor")
        return (fisher * R.square()).mean()
    if name == "source_ce":
        score = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b
        return functional.binary_cross_entropy_with_logits(score, resources.anchors_y.to(score.dtype))
    raise ValueError("unsupported regularizer: %s" % name)
