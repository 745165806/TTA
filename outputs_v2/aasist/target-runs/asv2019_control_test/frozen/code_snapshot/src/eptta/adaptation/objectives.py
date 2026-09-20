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
