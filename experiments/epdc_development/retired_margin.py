"""EPDC v0-B prototype: soft preservation of threshold-relevant source margins."""
import math

import torch

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.math import apply_adapter, project_frobenius_, view_loss
from eptta.adaptation.validation import validate_inputs


def normalized_margin_loss(R, resources, retention=0.9):
    """Mean squared relative loss of source-anchor margin against source tau0."""
    if type(retention) not in (int, float) or not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    adapted = apply_adapter(resources.anchors_z, resources.U, R)
    signs = 2 * resources.anchors_y.to(adapted.dtype) - 1
    margin = signs * (adapted @ resources.w + resources.b - resources.tau0)
    m0 = resources.anchors_m0
    return (torch.relu(retention * m0 - margin) / m0).square().mean()


def run_soft_preserve(target, resources, cfg, *, lambda_preserve=1.0, retention=0.9):
    """Fresh episodic target update; source labels only in the evidence term."""
    before = validate_inputs(target, resources, cfg)
    if cfg.lambda_keep != 0:
        raise ValueError("EPDC v0-B base objective requires lambda_keep=0")
    if type(lambda_preserve) not in (int, float) or not math.isfinite(lambda_preserve) or lambda_preserve < 0:
        raise ValueError("lambda_preserve must be finite and nonnegative")
    if lambda_preserve == 0:
        return run_cache_method("ep_no_keep", target, resources, cfg)
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    trace = []
    with torch.enable_grad():
        for step in range(cfg.steps):
            target_loss = view_loss(apply_adapter(target.features, resources.U, parameter))
            evidence_loss = normalized_margin_loss(parameter, resources, retention)
            total = target_loss + lambda_preserve * evidence_loss
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
                trace.append({"step": step + 1, "target_loss": float(target_loss.detach()),
                              "evidence_loss": float(evidence_loss.detach()),
                              "gradient_norm": float(torch.linalg.vector_norm(gradient)),
                              "update_norm": float(torch.linalg.vector_norm(parameter))})
    R = parameter.detach().clone()
    score = float((apply_adapter(target.features[:1], resources.U, R) @ resources.w + resources.b)[0])
    if not math.isfinite(score):
        raise FloatingPointError("non-finite EPDC score")
    return {"method_id": "epdc_normalized_margin_v0b", "sample_id": target.sample_id,
            "score_before": before, "score": score, "R": R, "status": "ok",
            "steps_completed": cfg.steps, "objective_evaluations": cfg.steps,
            "lambda_preserve": float(lambda_preserve), "retention": float(retention),
            "trace": trace}
