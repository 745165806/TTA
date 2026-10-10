"""Differentiable unlabeled full-head adaptation on frozen 160-D views."""
import torch
from torch import nn
from torch.nn import functional as F


class AuxiliaryLoss(nn.Module):
    def __init__(self, dim=160, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 4 + 5, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))

    def forward(self, views, weight, bias):
        # Every statistic here is computed from the current unlabeled support.
        z = views[:, 0]
        delta = (views[:, 1:] - z[:, None]).abs().mean(1)
        logits = views @ weight + bias
        support_mean = z.mean(0).expand_as(z)
        support_std = z.std(0, unbiased=False).expand_as(z)
        logit_stats = torch.stack((logits[:, 0].mean(), logits[:, 0].std(unbiased=False)))
        stats = logit_stats.expand(len(z), 2)
        inputs = torch.cat((z, delta, support_mean, support_std, logits, stats), dim=1)
        return self.net(inputs).mean()


def adapt_head(aux, support_views, source_weight, source_bias, *, steps=5, lr=0.05,
               create_graph=False):
    """Only unlabeled views enter this function; head state starts fresh per call."""
    if support_views.ndim != 3 or support_views.shape[1:] != (3, source_weight.numel()):
        raise ValueError("support features must be [N,3,160]")
    if not torch.isfinite(support_views).all():
        raise ValueError("nonfinite support features")
    weight = source_weight.detach().clone().requires_grad_(True)
    bias = source_bias.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss = aux(support_views, weight, bias)
        dw, db = torch.autograd.grad(loss, (weight, bias), create_graph=create_graph)
        weight = weight - lr * dw
        bias = bias - lr * db
        if not create_graph:
            weight = weight.detach().requires_grad_(True)
            bias = bias.detach().requires_grad_(True)
    return weight, bias


def outer_loss(scores, labels, rank_weight):
    bce = F.binary_cross_entropy_with_logits(scores, labels.float())
    if rank_weight == 0:
        return bce
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("ranking query requires both classes")
    ranking = F.softplus(negative[:, None] - positive[None, :]).mean()
    return bce + rank_weight * ranking
