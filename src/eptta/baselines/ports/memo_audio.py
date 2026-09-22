"""MEMO audio port: single-input full-model adaptation via marginal entropy.

Audio mapping (audited official test_adapt.py default): SGD, lr=2.5e-4, wd=0,
niter=1.  Augmentations = project fixed audio probe3 policy.
"""
import torch

from eptta.baselines.ports.common import marginal_entropy
from eptta.baselines.ports.tent_audio import score_current


def memo_adapt(model, adapter, views, class_index_map, lr=2.5e-4, steps=None,
               niter=None, weight_decay=0.0):
    """Full-model marginal-entropy adaptation; returns same-sample post-update score.

    ``views`` is the (3, L) probe3 batch (view0 original, view1 noise, view2 FIR).
    """
    for p in model.parameters():
        p.requires_grad = True
    if steps is not None and niter is not None and steps != niter:
        raise ValueError("MEMO steps and niter disagree")
    iterations = niter if niter is not None else (steps if steps is not None else 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _step in range(iterations):
        _emb, logits = adapter.forward(views)
        loss = marginal_entropy(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return score_current(model, adapter, views[:1], class_index_map)


__all__ = ["marginal_entropy", "memo_adapt"]
