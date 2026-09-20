"""Empirical per-anchor diagonal Fisher for the EP matrix at R=0."""
import torch


def empirical_diagonal_fisher(anchors_z, labels, U, weight, bias):
    if anchors_z.ndim != 2 or labels.shape != (anchors_z.shape[0],):
        raise ValueError("invalid Fisher anchors/labels")
    q = anchors_z @ U
    c = U.T @ weight
    scores = anchors_z @ weight + bias
    coefficient = torch.sigmoid(scores) - labels.to(scores.dtype)
    per_sample = coefficient[:, None, None] * c[None, :, None] * q[:, None, :]
    return per_sample.square().mean(0)
