"""Offline shared-R source control; target samples are never read here."""
import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_, view_loss
from eptta.adaptation.regularizers import regularizer


def fit_fixed_source_adapter(source_view_sets, resources, steps, lr, rho, gamma, regularizer_weight):
    if source_view_sets.ndim != 3 or source_view_sets.shape[2] != resources.U.shape[0]:
        raise ValueError("fixed adapter requires source-only [P,N,d] view sets")
    rank = resources.U.shape[1]
    R = torch.zeros(rank, rank, dtype=source_view_sets.dtype, device=source_view_sets.device,
                    requires_grad=True)
    trace = []
    with torch.enable_grad():
        for step in range(steps):
            target = view_loss(apply_adapter(source_view_sets, resources.U, R)).mean()
            keep = regularizer("margin", R, resources, gamma)
            loss = target + regularizer_weight * keep
            gradient, = torch.autograd.grad(loss, R)
            with torch.no_grad():
                R.add_(gradient, alpha=-lr)
                project_frobenius_(R, rho)
            trace.append({"step": step + 1, "source_view_loss": float(target.detach()),
                          "source_keep_loss": float(keep.detach())})
    return R.detach().clone(), trace
