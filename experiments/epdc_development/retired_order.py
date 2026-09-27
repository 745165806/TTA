"""EPDC v0-A: label-free target adaptation with soft source decision evidence."""
import math

import torch

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.math import apply_adapter, project_frobenius_, view_loss
from eptta.adaptation.types import EPConfig
from eptta.adaptation.validation import validate_inputs


def source_order_pairs(resources):
    """Pair source class anchors by decision hardness, using source labels only.

    A hard bonafide anchor has a high score; a hard spoof anchor has a low
    score. Equal-rank pairing preserves the source decision ordering geometry
    while allowing both anchors and target features to move.
    """
    bona = torch.nonzero(resources.anchors_y == 0, as_tuple=False).flatten()
    spoof = torch.nonzero(resources.anchors_y == 1, as_tuple=False).flatten()
    if bona.numel() != spoof.numel() or bona.numel() == 0:
        raise ValueError("balanced source anchors required")
    bona = bona[torch.argsort(resources.anchors_s0[bona], descending=True, stable=True)]
    spoof = spoof[torch.argsort(resources.anchors_s0[spoof], stable=True)]
    gap0 = resources.anchors_s0[spoof] - resources.anchors_s0[bona]
    if not bool(torch.isfinite(gap0).all()) or not bool((gap0 > 0).all()):
        raise ValueError("source ordering gap must be finite and positive")
    return bona, spoof, gap0


def decision_order_loss(R, resources, bona, spoof, gap0, retention=0.9):
    """Mean squared *relative* deficit of matched cross-class source gaps.

    L_evidence = mean_i [relu(eta*g_i^0 - g_i(R)) / g_i^0]^2,
    g_i(R) = s_spoof,i(R) - s_bonafide,i(R), eta=retention.
    """
    if type(retention) not in (int, float) or not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    scores = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b
    gap = scores[spoof] - scores[bona]
    return (torch.relu(retention * gap0 - gap) / gap0).square().mean()


def run_soft_preserve(target, resources, cfg, *, lambda_preserve=1.0, retention=0.9):
    """Fresh episodic R; no target label, hard guard, or cross-sample state."""
    before = validate_inputs(target, resources, cfg)
    if cfg.lambda_keep != 0:
        raise ValueError("EPDC v0-A base objective requires lambda_keep=0")
    if type(lambda_preserve) not in (int, float) or not math.isfinite(lambda_preserve) or lambda_preserve < 0:
        raise ValueError("lambda_preserve must be finite and nonnegative")
    if lambda_preserve == 0:
        # This is exactly the existing production Base Adapt path.
        return run_cache_method("ep_no_keep", target, resources, cfg)
    bona, spoof, gap0 = source_order_pairs(resources)
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    trace = []
    with torch.enable_grad():
        for step in range(cfg.steps):
            adapted = apply_adapter(target.features, resources.U, parameter)
            adapt_loss = view_loss(adapted)
            evidence_loss = decision_order_loss(parameter, resources, bona, spoof, gap0, retention)
            total = adapt_loss + lambda_preserve * evidence_loss
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("non-finite EPDC objective")
            gradient, = torch.autograd.grad(total, parameter)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("non-finite EPDC gradient")
            with torch.no_grad():
                parameter.add_(gradient, alpha=-cfg.lr)
                project_frobenius_(parameter, cfg.rho)
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError("non-finite EPDC parameter")
                trace.append({"step": step + 1, "target_loss": float(adapt_loss.detach()),
                              "evidence_loss": float(evidence_loss.detach()),
                              "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                              "update_norm": float(torch.linalg.vector_norm(parameter))})
    R = parameter.detach().clone()
    score = float((apply_adapter(target.features[:1], resources.U, R) @ resources.w + resources.b)[0])
    if not math.isfinite(score):
        raise FloatingPointError("non-finite EPDC score")
    return {"method_id": "epdc_soft_preserve_v0a", "sample_id": target.sample_id,
            "score_before": before, "score": score, "R": R, "status": "ok",
            "steps_completed": cfg.steps, "objective_evaluations": cfg.steps,
            "lambda_preserve": float(lambda_preserve), "retention": float(retention),
            "trace": trace}
