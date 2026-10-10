"""Source-only, fixed-rank subspaces for capacity diagnostics.

These constructors do not use target features or labels.  A task-oriented
subspace is a supervised source diagnostic, not an unlabeled TTA result.
"""
import torch


def _check(w, anchors_z, anchors_y, rank):
    if (w.ndim != 1 or anchors_z.ndim != 2 or anchors_z.shape[1] != w.numel()
            or anchors_y.shape != (anchors_z.shape[0],) or rank != 8
            or w.numel() < rank or anchors_z.shape[0] < 2):
        raise ValueError("expected source head [d], anchors [m,d], labels [m], rank=8")
    if (w.dtype not in (torch.float32, torch.float64) or anchors_z.dtype != w.dtype
            or w.device.type != "cpu" or anchors_z.device.type != "cpu"
            or anchors_y.device.type != "cpu" or anchors_y.dtype != torch.long
            or w.requires_grad or anchors_z.requires_grad or anchors_y.requires_grad
            or not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(anchors_z).all())
            or not bool(((anchors_y == 0) | (anchors_y == 1)).all())
            or not bool((anchors_y == 0).any()) or not bool((anchors_y == 1).any())
            or float(torch.linalg.vector_norm(w)) == 0):
        raise ValueError("invalid or non-finite source-only task subspace inputs")


def _append(basis, candidate, tolerance=1e-9):
    v = candidate.to(torch.float64).clone()
    for _ in range(2):
        for u in basis:
            v -= torch.dot(v, u) * u
    norm = torch.linalg.vector_norm(v)
    if float(norm) > tolerance:
        basis.append(v / norm)


def _finish(basis, rank, dtype):
    if len(basis) != rank:
        raise ValueError("source candidate directions do not span rank=8")
    U = torch.stack(basis, dim=1).to(dtype)
    if not torch.allclose(U.T @ U, torch.eye(rank, dtype=dtype), atol=1e-5, rtol=1e-5):
        raise ValueError("constructed task subspace is not orthonormal")
    return U


def source_task_subspace(w, anchors_z, anchors_y, rank=8):
    """Head direction, source class contrast, then residual source PCA.

    The first direction is exactly the trained head's score direction.  All
    other directions use the fixed fit-source anchor memory only.
    """
    _check(w, anchors_z, anchors_y, rank)
    z = anchors_z.to(torch.float64)
    y = anchors_y
    basis = []
    _append(basis, w)
    _append(basis, z[y == 1].mean(0) - z[y == 0].mean(0))
    centered = z - z.mean(0)
    covariance = centered.T @ centered / z.shape[0]
    _, eigenvectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    for candidate in eigenvectors.flip(1).T:
        if len(basis) == rank:
            break
        _append(basis, candidate)
    return _finish(basis, rank, w.dtype)


def source_mixed_subspace(original, w, anchors_z, anchors_y, rank=8):
    """Head direction plus seven response-U directions after deflation."""
    _check(w, anchors_z, anchors_y, rank)
    if (original.shape != (w.numel(), rank) or original.dtype != w.dtype
            or original.device.type != "cpu" or original.requires_grad
            or not bool(torch.isfinite(original).all())
            or not torch.allclose(original.T @ original, torch.eye(rank, dtype=w.dtype),
                                   atol=1e-5, rtol=1e-5)):
        raise ValueError("original U must be finite and orthonormal [d,8]")
    basis = []
    _append(basis, w)
    for candidate in original.T:
        if len(basis) == rank:
            break
        _append(basis, candidate)
    return _finish(basis, rank, w.dtype)
