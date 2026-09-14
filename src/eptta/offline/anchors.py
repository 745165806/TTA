"""Deterministic balanced positive-margin source memory construction."""
import torch


def build_anchor_memory(features, labels, weight, bias, tau0, per_class, seed):
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("anchor source arrays have incompatible shapes")
    scores = features @ weight + bias
    margins = (2 * labels.to(scores.dtype) - 1) * (scores - tau0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    for label in (0, 1):
        candidates = torch.nonzero((labels == label) & (margins > 1e-6), as_tuple=False).flatten().cpu()
        if len(candidates) < per_class:
            raise ValueError("insufficient distinct positive-margin anchors for class %d" % label)
        order = torch.randperm(len(candidates), generator=generator)[:per_class]
        selected.append(candidates[order].to(features.device))
    indices = torch.cat(selected)
    return {"anchors_z": features[indices].detach().clone(),
            "anchors_y": labels[indices].detach().clone(),
            "anchors_s0": scores[indices].detach().clone(),
            "anchors_m0": margins[indices].detach().clone(),
            "indices": indices.detach().clone()}
