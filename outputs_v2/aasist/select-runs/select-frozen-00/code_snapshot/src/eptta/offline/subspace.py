"""Deterministic source-only subspace estimators."""
import torch


def _top_eigenspace(second_moment, rank):
    if type(rank) is not int or not 1 <= rank <= second_moment.shape[0]:
        raise ValueError("invalid subspace rank")
    _, vectors = torch.linalg.eigh(second_moment.to(torch.float64))
    return vectors[:, -rank:].flip(1).to(second_moment.dtype)


def _sample_mix(seed, group, position):
    if type(group) is not int or type(position) is not int:
        raise ValueError("source sampling requires integer group/sample indices")
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15 * (group + 1) +
             0xBF58476D1CE4E5B9 * (position + 1)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def balanced_response_subspace(views, labels, groups, rank, treatment_families,
                               samples_per_group, pair_seed, accumulator_dtype=torch.float64):
    """Class x treatment-family balanced, source-group equal-unit uncentered second moment.

    views  : [P, N, d]  (N views: index 0 = identity, 1..N-1 = treatment families)
    labels : [P]        canonical 0=bonafide, 1=spoof
    groups : list[str] of length P (opaque source_group_id)
    Returns (U, diagnostics).  U has shape [d, rank].  Accumulation is FP64;
    the moment is symmetrized before ``eigh``.
    """
    import numpy as np
    P = views.shape[0]
    if views.ndim != 3 or labels.shape != (P,) or len(groups) != P:
        raise ValueError("balanced response inputs have incompatible shapes")
    if not bool(torch.isfinite(views).all()):
        raise ValueError("balanced response views must be finite")
    if type(treatment_families) is not tuple or not 1 <= len(treatment_families) <= views.shape[1] - 1:
        raise ValueError("treatment families must index transformed views only")
    if type(samples_per_group) is not int or samples_per_group < 1:
        raise ValueError("samples_per_group must be a positive integer")
    if type(pair_seed) is not int:
        raise ValueError("pair_seed must be an integer")
    label_array = labels.cpu().numpy().astype(np.int64)
    group_array = np.asarray(groups)
    deltas = (views[:, 1:] - views[:, 0:1]).cpu().numpy()  # [P, N-1, d]
    cell_moments = []
    diagnostics = {"cells": {}}
    for label in (0, 1):
        for family_index, family in enumerate(treatment_families):
            positions = np.nonzero(label_array == label)[0]
            if positions.size == 0:
                raise ValueError("missing cell for class %d family %r" % (label, family))
            by_group = {}
            for pos in positions.tolist():
                by_group.setdefault(group_array[pos], []).append(pos)
            sampled = []
            for group_index, group in enumerate(sorted(by_group)):
                members = by_group[group]
                if len(members) < samples_per_group:
                    raise ValueError("source group %r has fewer than samples_per_group units" % group)
                order = sorted(members, key=lambda p: _sample_mix(pair_seed, group_index, p))[:samples_per_group]
                sampled.extend(order)
            cell_delta = deltas[np.asarray(sampled, dtype=np.int64), family_index].astype(np.float64)
            moment = cell_delta.T @ cell_delta / cell_delta.shape[0]
            cell_moments.append(torch.from_numpy(moment))
            diagnostics["cells"]["%d:%s" % (label, family)] = {
                "groups": len(by_group), "sampled_units": int(cell_delta.shape[0])}
    C = sum(cell_moments) / len(cell_moments)  # equal weight per class x family cell
    C = (C + C.T) / 2  # symmetrize
    eigenvalues, vectors = torch.linalg.eigh(C.to(torch.float64))
    U = vectors[:, -rank:].flip(1).to(views.dtype)
    diagnostics.update({"eigenvalues": eigenvalues.detach().cpu().tolist(),
                        "rank": rank, "accumulator_dtype": str(accumulator_dtype)})
    return U, diagnostics


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
