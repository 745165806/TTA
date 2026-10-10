"""Fresh projected EP adapter shared only within an ordered local buffer."""
import math
import time

import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_, view_loss
from eptta.adaptation.objectives import entropy_from_logits
from eptta.adaptation.validation import validate_inputs
from experiments.task_objective_discovery.objectives import objective_terms, soft_affinity, source_geometry


def ordered_buffers(items, size):
    if type(size) is not int or size < 1:
        raise ValueError("buffer size must be a positive integer")
    ids = [item.sample_id for item in items]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("ordered buffer input must have unique, nonempty IDs")
    return [items[start:start + size] for start in range(0, len(items), size)]


def source_damage(R, resources):
    anchors = apply_adapter(resources.anchors_z, resources.U, R)
    signs = 2 * resources.anchors_y.to(anchors.dtype) - 1
    margins = signs * (anchors @ resources.w + resources.b - resources.tau0)
    return (torch.relu(.9 * resources.anchors_m0 - margins) /
            resources.anchors_m0).mean()


def run_local_buffer(items, resources, cfg, objective, objective_config, geometry=None,
                     *, domain="synthetic", buffer_index=0):
    """One R from zero for this buffer only; no optimizer object or persisted state."""
    if objective not in ("Base", "O1") or not items or cfg.lambda_keep != 0:
        raise ValueError("local study supports Base/O1 and lambda_keep=0 only")
    before = [validate_inputs(item, resources, cfg) for item in items]
    dtype, device = items[0].features.dtype, items[0].features.device
    if any(item.features.dtype != dtype or item.features.device != device for item in items):
        raise ValueError("all buffer features must share dtype/device")
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("duplicate buffer sample ID")
    geometry = geometry or source_geometry(resources)
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=dtype, device=device, requires_grad=True)
    stacked = torch.stack([item.features for item in items])

    def loss_at(R):
        if objective == "Base":
            return view_loss(apply_adapter(stacked, resources.U, R)).mean()
        return torch.stack([objective_terms(R, item, resources, geometry,
                                             objective_config)[0] for item in items]).mean()

    start = time.perf_counter()
    trace = []
    with torch.enable_grad():
        initial = loss_at(parameter)
        for step in range(cfg.steps):
            loss = loss_at(parameter)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite local objective")
            gradient, = torch.autograd.grad(loss, parameter)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("nonfinite local gradient")
            norm_before = float(torch.linalg.vector_norm(parameter.detach()))
            with torch.no_grad():
                parameter.add_(gradient, alpha=-cfg.lr)
                project_frobenius_(parameter, cfg.rho)
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError("nonfinite local parameter")
            trace.append({"step": step + 1, "parameter_norm_before": norm_before,
                          "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                          "parameter_norm_after": float(torch.linalg.vector_norm(parameter))})
    R = parameter.detach().clone()
    with torch.no_grad():
        final = loss_at(R)
        adapted = apply_adapter(stacked, resources.U, R)
        logits_before = stacked @ resources.w + resources.b
        logits_after = adapted @ resources.w + resources.b
        score_after = logits_after[:, 0]
        before_affinity = [soft_affinity(item.features, resources, geometry,
                                         objective_config["affinity_temperature"]).mean(0)
                           for item in items]
        after_affinity = [soft_affinity(adapted[i], resources, geometry,
                                        objective_config["affinity_temperature"]).mean(0)
                          for i in range(len(items))]
        entropy_before = entropy_from_logits(logits_before).mean(dim=1)
        entropy_after = entropy_from_logits(logits_after).mean(dim=1)
        damage = float(source_damage(R, resources))
    elapsed = time.perf_counter() - start
    scores = [float(value) for value in score_after]
    deltas = [score - frozen for score, frozen in zip(scores, before)]
    numeric = [float(initial), float(final), damage, float(torch.linalg.vector_norm(R))]
    numeric += scores + deltas + [float(value) for value in entropy_before]
    numeric += [float(value) for value in entropy_after]
    numeric += [float(value) for pair in before_affinity + after_affinity for value in pair]
    if any(not math.isfinite(value) for value in numeric):
        raise FloatingPointError("nonfinite local score or diagnostic")
    sample_rows = []
    for i, item in enumerate(items):
        sample_rows.append({"sample_id": item.sample_id, "domain": domain,
                            "buffer_index": buffer_index, "buffer_size": len(items),
                            "score_frozen": before[i], "score_after": scores[i],
                            "score_delta": deltas[i],
                            "entropy_before": float(entropy_before[i]),
                            "entropy_after": float(entropy_after[i]),
                            "affinity_before": before_affinity[i].tolist(),
                            "affinity_after": after_affinity[i].tolist(),
                            "numeric_status": "ok"})
    buffer_row = {"domain": domain, "buffer_index": buffer_index,
                  "buffer_size": len(items), "sample_ids": [item.sample_id for item in items],
                  "objective": objective, "objective_before": float(initial),
                  "objective_after": float(final),
                  "R_norm": float(torch.linalg.vector_norm(R)),
                  "parameter_delta_norm": float(torch.linalg.vector_norm(R)),
                  "mean_score_delta": sum(deltas) / len(deltas),
                  "max_score_delta": max(abs(value) for value in deltas),
                  "mean_entropy_before": float(entropy_before.mean()),
                  "mean_entropy_after": float(entropy_after.mean()),
                  "source_evidence_damage": damage,
                  "runtime_seconds": elapsed, "numeric_status": "ok",
                  "trace": trace}
    return {"R": R, "samples": sample_rows, "buffer": buffer_row}
