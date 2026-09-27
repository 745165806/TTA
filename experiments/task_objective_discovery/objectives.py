"""Three preregistered label-free task objectives on the production EP geometry."""
import math

import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_
from eptta.adaptation.validation import validate_inputs


def source_geometry(resources):
    """Build immutable affinity geometry using source anchors and labels only."""
    anchors = resources.anchors_z.detach()
    labels = resources.anchors_y.detach().to(torch.long)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("source anchor geometry requires both source classes")
    projected = anchors @ resources.U.detach()
    scores = anchors @ resources.w.detach() + resources.b
    distances = torch.pdist(projected)
    nonzero = distances[distances > 0]
    if nonzero.numel() == 0:
        raise ValueError("degenerate source-anchor geometry")
    sigma_u = nonzero.median().clamp_min(1e-6)
    sigma_s = scores.std(unbiased=False).clamp_min(1e-6)
    return {"projected": projected, "scores": scores, "labels": labels,
            "sigma_u": sigma_u, "sigma_s": sigma_s}


def soft_affinity(adapted_views, resources, geometry, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("affinity temperature must be positive")
    projected = adapted_views @ resources.U
    scores = adapted_views @ resources.w + resources.b
    distance = ((projected[:, None, :] - geometry["projected"][None, :, :]) /
                geometry["sigma_u"]).square().sum(dim=-1)
    distance = distance + ((scores[:, None] - geometry["scores"][None, :]) /
                           geometry["sigma_s"]).square()
    kernel = -distance / (2.0 * temperature)
    logits = []
    for label in (0, 1):
        group = kernel[:, geometry["labels"] == label]
        logits.append(torch.logsumexp(group, dim=1) - math.log(group.shape[1]))
    return torch.softmax(torch.stack(logits, dim=1), dim=1)


def objective_terms(R, target, resources, geometry, config):
    adapted = apply_adapter(target.features, resources.U, R)
    probability = soft_affinity(adapted, resources, geometry,
                                config["affinity_temperature"])
    consensus = probability.mean(dim=0)
    detached = consensus.detach().clamp_min(1e-12)
    agreement = (detached[None, :] *
                 (detached[None, :].log() - probability.clamp_min(1e-12).log())).sum(dim=1).mean()
    entropy = -(consensus * consensus.clamp_min(1e-12).log()).sum()
    o1 = agreement + config["affinity_entropy_weight"] * entropy
    scores = adapted @ resources.w + resources.b
    o2 = ((scores - scores.mean()) / geometry["sigma_s"]).square().mean()
    return o1, o2, probability, agreement, entropy


def frozen_reliability(target, resources, geometry, config):
    with torch.no_grad():
        probability = soft_affinity(target.features, resources, geometry,
                                    config["affinity_temperature"])
        agreement = (1 - probability[:, 1].std(unbiased=False)).clamp(0, 1)
        gap = (probability[:, 1] - probability[:, 0]).mean().abs()
        sharpness = config["o3_reliability_sharpness"]
        weight = torch.sigmoid(sharpness * (agreement - 0.5)) * \
            torch.sigmoid(sharpness * (gap - 0.5))
        return float(agreement), float(gap), float(weight)


def run_objective(arm, target, resources, cfg, config, geometry=None):
    """Per-sample projected SGD; numerical failures raise instead of silently falling back."""
    if arm not in ("O1", "O2", "O3"):
        raise ValueError("unknown task objective")
    before = validate_inputs(target, resources, cfg)
    geometry = geometry or source_geometry(resources)
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    agreement, gap, weight = frozen_reliability(target, resources, geometry, config)

    def evaluate(R):
        o1, o2, _probability, affinity_agreement, affinity_entropy = objective_terms(
            R, target, resources, geometry, config)
        if arm == "O1":
            total = o1
        elif arm == "O2":
            total = config["decision_consistency_weight"] * o2
        else:
            total = config["o3_affinity_weight"] * weight * o1 + \
                config["decision_consistency_weight"] * o2
        return total, o1, o2, affinity_agreement, affinity_entropy

    trace = []
    with torch.enable_grad():
        initial = evaluate(parameter)
        for step in range(cfg.steps):
            total, o1, o2, _, _ = evaluate(parameter)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("nonfinite task objective")
            gradient, = torch.autograd.grad(total, parameter)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("nonfinite task objective gradient")
            norm = float(torch.linalg.vector_norm(gradient))
            with torch.no_grad():
                parameter.add_(gradient, alpha=-cfg.lr)
                project_frobenius_(parameter, cfg.rho)
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError("nonfinite task update")
            trace.append({"step": step + 1, "objective": float(total.detach()),
                          "o1": float(o1.detach()), "o2": float(o2.detach()),
                          "gradient_norm": norm,
                          "update_norm": float(torch.linalg.vector_norm(parameter))})
    R = parameter.detach().clone()
    with torch.no_grad():
        final = evaluate(R)
        score = float((apply_adapter(target.features[:1], resources.U, R) @ resources.w + resources.b)[0])
    if not math.isfinite(score) or not all(math.isfinite(float(v)) for v in final):
        raise FloatingPointError("nonfinite task result")
    return {"sample_id": target.sample_id, "arm": arm, "score_before": before,
            "score_after": score, "R": R, "steps_completed": cfg.steps,
            "objective_before": float(initial[0].detach()),
            "objective_after": float(final[0].detach()),
            "o1_before": float(initial[1].detach()), "o1_after": float(final[1]),
            "o2_before": float(initial[2].detach()), "o2_after": float(final[2]),
            "affinity_agreement": agreement, "affinity_gap": gap,
            "reliability_weight": weight, "trace": trace, "status": "ok"}
