"""Deterministic balanced positive-margin source memory construction."""
import torch


def build_anchor_memory(features, labels, weight, bias, tau0, per_class, seed,
                        margin_bins=1, margin_epsilon=1e-6):
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("anchor source arrays have incompatible shapes")
    if type(margin_bins) is not int or margin_bins < 1 or per_class < margin_bins:
        raise ValueError("margin_bins must be a positive integer no larger than per_class")
    scores = features @ weight + bias
    margins = (2 * labels.to(scores.dtype) - 1) * (scores - tau0)
    selected = []
    for label in (0, 1):
        candidates = torch.nonzero((labels == label) & (margins > margin_epsilon), as_tuple=False).flatten()
        if len(candidates) < per_class:
            raise ValueError("insufficient distinct positive-margin anchors for class %d" % label)
        candidate_margins = margins[candidates]
        order = torch.argsort(candidate_margins)  # ascending margin = hard first
        sorted_candidates = candidates[order]
        total = len(sorted_candidates)
        per_bin = per_class // margin_bins
        remainder = per_class % margin_bins
        boundaries = [(bin_index * total // margin_bins,
                       (bin_index + 1) * total // margin_bins)
                      for bin_index in range(margin_bins)]
        quotas = [per_bin] * margin_bins
        eligible = [index for index, (start, end) in enumerate(boundaries)
                    if end - start > per_bin]
        if len(eligible) < remainder:
            raise ValueError("margin bins cannot satisfy the requested anchor count")
        for index in eligible[:remainder]:
            quotas[index] += 1
        chosen = []
        for bin_index, ((start, end), count) in enumerate(zip(boundaries, quotas)):
            bin_candidates = sorted_candidates[start:end]
            generator = torch.Generator(device="cpu").manual_seed(seed + label * 7919 + bin_index * 31)
            permutation = torch.randperm(len(bin_candidates), generator=generator)[:count].to(
                bin_candidates.device)
            chosen.append(bin_candidates[permutation])
        class_selection = torch.cat(chosen)
        if len(class_selection) != per_class:
            raise RuntimeError("anchor stratification did not preserve the requested class count")
        selected.append(class_selection)
    indices = torch.cat(selected)
    return {"anchors_z": features[indices].detach().clone(),
            "anchors_y": labels[indices].detach().clone(),
            "anchors_s0": scores[indices].detach().clone(),
            "anchors_m0": margins[indices].detach().clone(),
            "indices": indices.detach().clone()}
