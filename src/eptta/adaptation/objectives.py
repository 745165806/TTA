"""Numerically stable auxiliary objectives for the shared EP adapter."""
import torch
import torch.nn.functional as functional

from eptta.adaptation.math import view_loss


def entropy_from_logits(logits):
    probability = torch.sigmoid(logits)
    return -(probability * functional.logsigmoid(logits) +
             (1 - probability) * functional.logsigmoid(-logits))


def mean_view_entropy(scores):
    return entropy_from_logits(scores).mean(dim=-1)


def marginal_entropy(scores):
    probability = torch.sigmoid(scores).mean(dim=-1)
    return -(torch.xlogy(probability, probability) +
             torch.xlogy(1 - probability, 1 - probability))


def target_objective(name, adapted_views, weight, bias):
    if name == "view_variance":
        return view_loss(adapted_views)
    scores = adapted_views @ weight + bias
    if name == "mean_view_entropy":
        return mean_view_entropy(scores)
    if name == "marginal_entropy":
        return marginal_entropy(scores)
    raise ValueError("unsupported target objective: %s" % name)


def calibrated_logits(scores, tau0, temperature):
    """Shift decision scores by the source calibration threshold, then scale.

    The threshold ``tau0`` is the frozen cal0 operating point, so the sign of
    ``score - tau0`` (not ``score > 0``) decides the frozen pseudo class.
    """
    return (scores - tau0) / temperature


def calibrated_pseudo_bce(logits, teacher):
    """Per-view pseudo-label BCE against a detached frozen teacher.

    ``teacher`` is the original-view frozen class (0/1) and is never optimized;
    it is expanded to every view so each adapted view is pulled toward the same
    task-space label.
    """
    target = teacher.to(logits.dtype).detach().expand_as(logits)
    return functional.binary_cross_entropy_with_logits(logits, target)


def task_logit_consistency(logits):
    """Variance of the per-view calibrated logits (task-space, not features)."""
    return (logits - logits.mean(dim=-1, keepdim=True)).square().mean(dim=-1)
