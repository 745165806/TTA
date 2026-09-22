"""EP primitives using the column-vector R convention and row feature storage."""
import math

import torch


def apply_adapter(Z, U, R):
    return Z + ((Z @ U) @ R.transpose(-1, -2)) @ U.T


def view_loss(adapted):
    centered = adapted - adapted.mean(dim=-2, keepdim=True)
    return centered.square().mean(dim=(-2, -1))  # Nd, includes the orthogonal complement


def margin_deficit(adapted_anchors, w, b, labels, m0, tau0, gamma):
    margin = (2 * labels.to(adapted_anchors.dtype) - 1) * (adapted_anchors @ w + b - tau0)
    return torch.relu((1 - gamma) * m0 - margin)


def margin_tolerance(resources, dtype):
    """Shared numerical tolerance for guarded source-margin feasibility."""
    scale = max(float(resources.anchors_m0.abs().max()), 1.0)
    return 64.0 * torch.finfo(dtype).eps * scale


def keep_loss(adapted_anchors, w, b, labels, m0, tau0, gamma):
    return margin_deficit(adapted_anchors, w, b, labels, m0, tau0, gamma).square().mean(dim=-1)


@torch.no_grad()
def project_frobenius_(R, rho):
    if R.ndim < 2 or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("expected matrices and 0 < rho < 1")
    norm = torch.linalg.vector_norm(R, dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise FloatingPointError("non-finite projection norm")
    R.mul_((rho / norm.clamp_min(torch.finfo(R.dtype).tiny)).clamp(max=1))
