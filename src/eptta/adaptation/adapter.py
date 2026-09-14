"""Dispatch the required cache-based mechanism controls with fresh per-sample state."""
import math

import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_
from eptta.adaptation.objectives import target_objective
from eptta.adaptation.regularizers import regularizer
from eptta.adaptation.types import EPConfig
from eptta.adaptation.validation import validate_inputs


METHODS = {
    "ep_tta": ("view_variance", "margin"),
    "ep_no_keep": ("view_variance", "none"),
    "ep_random_U": ("view_variance", "margin"),
    "ep_feature_pca_U": ("view_variance", "margin"),
    "ep_no_projection": ("view_variance", "margin"),
    "entropy_same_adapter_no_keep": ("mean_view_entropy", "none"),
    "entropy_same_adapter": ("mean_view_entropy", "margin"),
    "memo_same_adapter_no_keep": ("marginal_entropy", "none"),
    "memo_same_adapter_keep": ("marginal_entropy", "margin"),
    "ep_keep_l2": ("view_variance", "parameter_l2"),
    "ep_keep_logit": ("view_variance", "source_logit"),
    "ep_keep_fisher": ("view_variance", "fisher"),
    "source_ce_only": (None, "source_ce"),
    "ep_diagonal_R": ("view_variance", "margin"),
}


def _score(features, resources):
    return features @ resources.w + resources.b


def run_cache_method(method_id, target, resources, cfg=EPConfig(), params=None):
    """Every call creates new optimization state; no distributed collective exists here."""
    params = {} if params is None else dict(params)
    before = validate_inputs(target, resources, cfg)
    if method_id == "frozen":
        return {"method_id": method_id, "sample_id": target.sample_id, "score": before,
                "score_before": before, "steps_completed": 0, "objective_evaluations": 0}
    if method_id == "multiview_mean":
        score = float(_score(target.features, resources).mean())
        return {"method_id": method_id, "sample_id": target.sample_id, "score": score,
                "score_before": before, "steps_completed": 0, "objective_evaluations": 0}
    if method_id == "frozen_source_shift":
        shift = params.get("score_shift")
        if not isinstance(shift, (int, float)) or not math.isfinite(shift):
            raise ValueError("frozen source shift must be a finite source-selected scalar")
        return {"method_id": method_id, "sample_id": target.sample_id, "score": before + float(shift),
                "score_before": before, "steps_completed": 0, "objective_evaluations": 0}
    rank = resources.U.shape[1]
    if method_id == "static_subspace":
        amount = params.get("amount")
        maximum = cfg.rho / math.sqrt(rank)
        if not isinstance(amount, (int, float)) or not 0 <= amount <= maximum:
            raise ValueError("static amount exceeds the matched Frobenius budget")
        R = -float(amount) * torch.eye(rank, dtype=target.features.dtype, device=target.features.device)
        return {"method_id": method_id, "sample_id": target.sample_id,
                "score": float(_score(apply_adapter(target.features[:1], resources.U, R), resources)[0]),
                "score_before": before, "R": R, "steps_completed": 0, "objective_evaluations": 0}
    if method_id == "fixed_source_adapter":
        R = params.get("fixed_R")
        if not isinstance(R, torch.Tensor) or R.shape != (rank, rank) or R.requires_grad:
            raise ValueError("fixed source adapter requires a detached [r,r] artifact")
        return {"method_id": method_id, "sample_id": target.sample_id,
                "score": float(_score(apply_adapter(target.features[:1], resources.U, R), resources)[0]),
                "score_before": before, "R": R.clone(), "steps_completed": 0, "objective_evaluations": 0}
    if method_id == "ep_scalar_adaptive":
        size = params.get("grid_size", 17)
        if size != 17:
            raise ValueError("ep_scalar_adaptive uses the preregistered 17-point grid")
        objective, keep = "view_variance", "margin"
        best = None
        with torch.no_grad():
            for index in range(size):
                amount = cfg.rho / math.sqrt(rank) * index / (size - 1)
                R = -amount * torch.eye(rank, dtype=target.features.dtype, device=target.features.device)
                loss = target_objective(objective, apply_adapter(target.features, resources.U, R),
                                        resources.w, resources.b)
                loss = loss + cfg.lambda_keep * regularizer(keep, R, resources, cfg.gamma)
                candidate = (float(loss), amount, R)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        return {"method_id": method_id, "sample_id": target.sample_id,
                "score": float(_score(apply_adapter(target.features[:1], resources.U, best[2]), resources)[0]),
                "score_before": before, "R": best[2], "selected_amount": best[1],
                "steps_completed": 0, "objective_evaluations": 17}
    if method_id not in METHODS:
        raise ValueError("unsupported cache method: %s" % method_id)
    objective_name, regularizer_name = METHODS[method_id]
    diagonal = method_id == "ep_diagonal_R"
    parameter = torch.zeros(rank if diagonal else (rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    trace = []
    with torch.enable_grad():
        for step in range(cfg.steps):
            R = torch.diag(parameter) if diagonal else parameter
            objective = R.sum() * 0 if objective_name is None else target_objective(
                objective_name, apply_adapter(target.features, resources.U, R), resources.w, resources.b)
            keep = regularizer(regularizer_name, R, resources, cfg.gamma, params.get("fisher"))
            loss = objective + cfg.lambda_keep * keep
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite mechanism loss")
            gradient, = torch.autograd.grad(loss, parameter)
            with torch.no_grad():
                parameter.add_(gradient, alpha=-cfg.lr)
                if method_id != "ep_no_projection":
                    if diagonal:
                        norm = torch.linalg.vector_norm(parameter)
                        if norm > cfg.rho:
                            parameter.mul_(cfg.rho / norm)
                    else:
                        project_frobenius_(parameter, cfg.rho)
            trace.append({"step": step + 1, "target_objective": float(objective.detach()),
                          "regularizer": float(keep.detach())})
    R = torch.diag(parameter) if diagonal else parameter
    score = float(_score(apply_adapter(target.features[:1], resources.U, R), resources)[0])
    return {"method_id": method_id, "sample_id": target.sample_id, "score": score,
            "score_before": before, "R": R.detach().clone(), "steps_completed": cfg.steps,
            "objective_evaluations": cfg.steps, "trace": trace}
