"""Deterministic source-only subspace estimators."""
import torch


def _top_eigenspace(second_moment, rank):
    if type(rank) is not int or not 1 <= rank <= second_moment.shape[0]:
        raise ValueError("invalid subspace rank")
    _, vectors = torch.linalg.eigh(second_moment.to(torch.float64))
    return vectors[:, -rank:].flip(1).to(second_moment.dtype)


def response_subspace(deltas, rank, weights=None):
    if deltas.ndim != 2 or not bool(torch.isfinite(deltas).all()):
        raise ValueError("response deltas must be finite [P,d]")
    if weights is None:
        weights = torch.ones(deltas.shape[0], dtype=deltas.dtype, device=deltas.device)
    if weights.shape != (deltas.shape[0],) or bool((weights <= 0).any()):
        raise ValueError("response weights must be positive [P]")
    moment = deltas.T @ (deltas * weights[:, None]) / weights.sum()
    return _top_eigenspace(moment, rank)


def feature_pca_subspace(features, rank, weights=None):
    if features.ndim != 2 or not bool(torch.isfinite(features).all()):
        raise ValueError("source features must be finite [P,d]")
    if weights is None:
        weights = torch.ones(features.shape[0], dtype=features.dtype, device=features.device)
    mean = (features * weights[:, None]).sum(0) / weights.sum()
    centered = features - mean
    moment = centered.T @ (centered * weights[:, None]) / weights.sum()
    return _top_eigenspace(moment, rank)


def random_subspace(dimension, rank, seed, dtype=torch.float32, device="cpu"):
    if not 1 <= rank <= dimension:
        raise ValueError("invalid random subspace shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dimension, rank, generator=generator, dtype=dtype)
    return torch.linalg.qr(matrix, mode="reduced")[0].to(device)
