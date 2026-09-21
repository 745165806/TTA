"""Cache-based mechanism dispatch with fresh, isolated per-sample state."""
import math

import torch

from eptta.adaptation.math import apply_adapter, margin_deficit, margin_tolerance, project_frobenius_
from eptta.adaptation.objectives import target_objective
from eptta.adaptation.regularizers import regularizer
from eptta.adaptation.types import EPConfig
from eptta.adaptation.validation import validate_inputs


METHODS = {
    "ep_tta": ("view_variance", "margin"),
    "ep_tta_guarded": ("view_variance", "margin"),
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


def _finite(value, message):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(message)
    return value


@torch.no_grad()
def _enforce_margin_guard_(parameter, previous, resources, gamma, max_backtracks=16):
    """Make a projected SGD candidate source-margin feasible.

    Only the frozen source-anchor memory is inspected.  The backtracking
    schedule is deterministic and never depends on a target prediction or
    target label.
    """
    if parameter.ndim != 2 or previous.shape != parameter.shape:
        raise ValueError("margin guard requires matching [r,r] matrices")

    tolerance = margin_tolerance(resources, parameter.dtype)

    def is_feasible(candidate):
        deficit = margin_deficit(
            apply_adapter(resources.anchors_z, resources.U, candidate),
            resources.w,
            resources.b,
            resources.anchors_y,
            resources.anchors_m0,
            resources.tau0,
            gamma,
        )
        _finite(deficit, "non-finite margin guard deficit")
        return not bool((deficit > tolerance).any())

    if is_feasible(parameter):
        return False, 0, False
    if not is_feasible(previous):
        raise FloatingPointError("margin guard previous state is infeasible")

    candidate = parameter.detach().clone()
    for backtracks in range(1, max_backtracks + 1):
        candidate.copy_(previous + 0.5 * (candidate - previous))
        if is_feasible(candidate):
            parameter.copy_(candidate)
            return True, backtracks, False

    parameter.copy_(previous)
    return True, max_backtracks, True


def validate_method_setup(method_id, resources, cfg, params=None):
    """Validate method/resource parameters before a suite opens its sample loop."""
    params = {} if params is None else params
    if type(params) is not dict:
        raise ValueError("method params must be an object")
    rank = resources.U.shape[1]
    allowed_params = {"static_subspace": {"amount"}, "fixed_source_adapter": {"fixed_R"},
                      "frozen_source_shift": {"score_shift"}, "ep_scalar_adaptive": {"grid_size"},
                      "ep_keep_fisher": {"fisher"}}
    unknown = set(params) - allowed_params.get(method_id, set())
    if unknown:
        raise ValueError("unknown params for %s: %s" % (method_id, sorted(unknown)))
    if method_id in ("frozen", "multiview_mean") or method_id in METHODS:
        pass
    elif method_id == "static_subspace":
        if "amount" not in params:
            raise ValueError("static_subspace requires explicit params.amount")
        amount = params["amount"]
        maximum = cfg.rho / math.sqrt(rank)
        if type(amount) not in (int, float) or not math.isfinite(amount) or not 0 <= amount <= maximum:
            raise ValueError("static amount exceeds the matched Frobenius budget")
    elif method_id == "fixed_source_adapter":
        fixed = params.get("fixed_R")
        if (not isinstance(fixed, torch.Tensor) or fixed.shape != (rank, rank) or fixed.requires_grad or
                fixed.dtype != resources.U.dtype or fixed.device != resources.U.device or
                not bool(torch.isfinite(fixed).all())):
            raise ValueError("fixed source adapter requires a finite detached matching [r,r] artifact")
    elif method_id == "frozen_source_shift":
        shift = params.get("score_shift")
        if type(shift) not in (int, float) or not math.isfinite(shift):
            raise ValueError("frozen source shift must be a finite source-selected scalar")
    elif method_id == "ep_scalar_adaptive":
        if params.get("grid_size", 17) != 17:
            raise ValueError("ep_scalar_adaptive uses the preregistered 17-point grid")
    else:
        raise ValueError("unsupported cache method: %s" % method_id)
    if method_id == "ep_keep_fisher":
        fisher = params.get("fisher")
        if (not isinstance(fisher, torch.Tensor) or fisher.shape != (rank, rank) or
                fisher.dtype != resources.U.dtype or fisher.device != resources.U.device or
                fisher.requires_grad or not bool(torch.isfinite(fisher).all())):
            raise ValueError("Fisher regularizer requires a finite detached matching tensor")
    return params


@torch.no_grad()
def adaptation_diagnostics(result, target, resources, cfg):
    """Recompute final diagnostics from the actual final R and regularizer contract."""
    if result.get("status") == "fallback_numeric":
        diagnostic = {name: None for name in (
            "final_regularizer_loss", "final_margin_loss", "final_margin_violation_fraction",
            "min_margin_change", "max_abs_margin_change", "final_R_norm", "delta_score",
            "delta_z_norm", "regularizer_active_steps", "regularizer_gradient_norm_max",
            "projection_count")}
        if result.get("method_id") == "ep_tta_guarded":
            diagnostic.update(margin_guard_count=None, margin_guard_backtracks=None,
                              margin_guard_reverts=None)
        diagnostic.update(completed_steps=int(result["steps_completed"]),
                          objective_evaluations=int(result["objective_evaluations"]),
                          gradient_trace_status="failed_numeric")
        return diagnostic
    R = result.get("R")
    if R is None:
        rank = resources.U.shape[1]
        R = torch.zeros((rank, rank), dtype=target.features.dtype, device=target.features.device)
    _finite(R, "non-finite final R")
    adapted = apply_adapter(resources.anchors_z, resources.U, R)
    deficit = margin_deficit(adapted, resources.w, resources.b, resources.anchors_y,
                             resources.anchors_m0, resources.tau0, cfg.gamma)
    _finite(deficit, "non-finite final margin deficit")
    signs = 2 * resources.anchors_y.to(adapted.dtype) - 1
    margins = signs * (adapted @ resources.w + resources.b - resources.tau0)
    changes = margins - resources.anchors_m0
    regularizer_name = result.get("regularizer_name", "none")
    if regularizer_name == "none":
        final_regularizer = 0.0
    else:
        value = regularizer(regularizer_name, R, resources, cfg.gamma, result.get("fisher"))
        final_regularizer = float(_finite(value, "non-finite final regularizer"))
    delta = apply_adapter(target.features[:1], resources.U, R) - target.features[:1]
    trace = result.get("trace") or []
    gradient_norms = [float(row["regularizer_gradient_norm"])
                      for row in trace if row.get("regularizer_gradient_norm") is not None]
    violation_tolerance = (margin_tolerance(resources, adapted.dtype)
                           if result.get("method_id") == "ep_tta_guarded" else 0.0)
    diagnostic = {
        "final_regularizer_loss": final_regularizer,
        "final_margin_loss": float(deficit.square().mean()),
        "final_margin_violation_fraction": float(
            (deficit > violation_tolerance).to(adapted.dtype).mean()),
        "min_margin_change": float(changes.min()),
        "max_abs_margin_change": float(changes.abs().max()),
        "projection_count": int(sum(bool(row.get("projection_applied")) for row in trace)),
        "final_R_norm": float(torch.linalg.vector_norm(R)),
        "delta_score": float(result["score"] - result["score_before"]),
        "delta_z_norm": float(torch.linalg.vector_norm(delta)),
        "completed_steps": int(result["steps_completed"]),
        "regularizer_active_steps": int(sum(value > 0 for value in gradient_norms)),
        "regularizer_gradient_norm_max": max(gradient_norms, default=0.0),
        "objective_evaluations": int(result["objective_evaluations"]),
        "gradient_trace_status": "not_applicable" if result.get("solver") == "fixed_17_grid" else "recorded",
    }
    if result.get("method_id") == "ep_tta_guarded":
        diagnostic.update(
            margin_guard_count=int(sum(bool(row.get("margin_guard_applied")) for row in trace)),
            margin_guard_backtracks=int(sum(
                int(row.get("margin_guard_backtracks", 0)) for row in trace)),
            margin_guard_reverts=int(sum(
                bool(row.get("margin_guard_reverted")) for row in trace)),
        )
    return diagnostic


def _base_result(method_id, target, before, score, **extra):
    result = {"method_id": method_id, "sample_id": target.sample_id, "score": float(score),
              "score_before": before, "status": "ok", "steps_completed": 0,
              "objective_evaluations": 0, "regularizer_name": "none", "trace": []}
    result.update(extra)
    if not math.isfinite(result["score"]):
        raise FloatingPointError("non-finite final score")
    return result


def _numeric_fallback(method_id, target, before, completed, evaluations, exc, rank):
    return {"method_id": method_id, "sample_id": target.sample_id, "score": before,
            "score_before": before, "status": "fallback_numeric", "steps_completed": completed,
            "objective_evaluations": evaluations,
            "R": torch.zeros((rank, rank), dtype=target.features.dtype, device=target.features.device),
            "trace": [],
            "regularizer_name": None, "error_type": type(exc).__name__, "error_message": str(exc)}


def run_cache_method(method_id, target, resources, cfg=EPConfig(), params=None):
    """Run one method; only numerical failures become a per-sample frozen fallback."""
    params = {} if params is None else dict(params)
    before = validate_inputs(target, resources, cfg)
    validate_method_setup(method_id, resources, cfg, params)
    rank = resources.U.shape[1]
    if method_id == "frozen":
        return _base_result(method_id, target, before, before)
    if method_id == "multiview_mean":
        return _base_result(method_id, target, before, float(_finite(
            _score(target.features, resources).mean(), "non-finite multiview score")))
    if method_id == "frozen_source_shift":
        return _base_result(method_id, target, before, before + float(params["score_shift"]))
    if method_id == "static_subspace":
        amount = float(params["amount"])
        R = -amount * torch.eye(rank, dtype=target.features.dtype, device=target.features.device)
        score = before if amount == 0 else float(_finite(
            _score(apply_adapter(target.features[:1], resources.U, R), resources)[0],
            "non-finite static score"))
        return _base_result(method_id, target, before, score, R=R)
    if method_id == "fixed_source_adapter":
        R = params["fixed_R"].clone()
        score = float(_finite(_score(apply_adapter(target.features[:1], resources.U, R), resources)[0],
                              "non-finite fixed adapter score"))
        return _base_result(method_id, target, before, score, R=R)
    if method_id == "ep_scalar_adaptive":
        best = None
        try:
            with torch.no_grad():
                for index in range(17):
                    amount = cfg.rho / math.sqrt(rank) * index / 16
                    R = -amount * torch.eye(rank, dtype=target.features.dtype, device=target.features.device)
                    loss = target_objective("view_variance", apply_adapter(target.features, resources.U, R),
                                            resources.w, resources.b)
                    loss = _finite(loss + cfg.lambda_keep * regularizer(
                        "margin", R, resources, cfg.gamma), "non-finite scalar-grid objective")
                    candidate = (float(loss), amount, R)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            score = float(_finite(_score(apply_adapter(target.features[:1], resources.U, best[2]), resources)[0],
                                  "non-finite scalar-grid score"))
            return _base_result(method_id, target, before, score, R=best[2], selected_amount=best[1],
                                objective_evaluations=17, regularizer_name="margin",
                                solver="fixed_17_grid")
        except FloatingPointError as exc:
            return _numeric_fallback(method_id, target, before, 0, 0, exc, rank)

    objective_name, regularizer_name = METHODS[method_id]
    diagonal = method_id == "ep_diagonal_R"
    guarded = method_id == "ep_tta_guarded"
    parameter = torch.zeros(rank if diagonal else (rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    trace, completed = [], 0
    try:
        with torch.enable_grad():
            for step in range(cfg.steps):
                R = torch.diag(parameter) if diagonal else parameter
                objective = R.sum() * 0 if objective_name is None else target_objective(
                    objective_name, apply_adapter(target.features, resources.U, R), resources.w, resources.b)
                keep = regularizer(regularizer_name, R, resources, cfg.gamma, params.get("fisher"))
                loss = _finite(objective + cfg.lambda_keep * keep, "non-finite mechanism loss")
                gradient, = torch.autograd.grad(loss, parameter, retain_graph=regularizer_name != "none")
                _finite(gradient, "non-finite mechanism gradient")
                regularizer_gradient_norm = 0.0
                if regularizer_name != "none" and cfg.lambda_keep > 0:
                    reg_gradient, = torch.autograd.grad(cfg.lambda_keep * keep, parameter)
                    regularizer_gradient_norm = float(torch.linalg.vector_norm(
                        _finite(reg_gradient, "non-finite regularizer gradient")))
                with torch.no_grad():
                    previous = parameter.detach().clone() if guarded else None
                    parameter.add_(gradient, alpha=-cfg.lr)
                    _finite(parameter, "non-finite parameter before projection")
                    pre_norm = float(_finite(torch.linalg.vector_norm(parameter),
                                             "non-finite pre-projection norm"))
                    projection_applied = method_id != "ep_no_projection" and pre_norm > cfg.rho
                    if method_id != "ep_no_projection":
                        if diagonal:
                            if projection_applied:
                                parameter.mul_(cfg.rho / pre_norm)
                        else:
                            project_frobenius_(parameter, cfg.rho)
                    _finite(parameter, "non-finite parameter after projection")
                    margin_guard_applied = False
                    margin_guard_backtracks = 0
                    margin_guard_reverted = False
                    if guarded:
                        (margin_guard_applied,
                         margin_guard_backtracks,
                         margin_guard_reverted) = _enforce_margin_guard_(
                             parameter, previous, resources, cfg.gamma)
                        _finite(parameter, "non-finite parameter after margin guard")
                    post_norm = float(_finite(torch.linalg.vector_norm(parameter),
                                              "non-finite post-projection norm"))
                completed = step + 1
                trace_row = {"step": completed, "target_objective": float(objective.detach()),
                             "regularizer": float(keep.detach()),
                             "regularizer_gradient_norm": regularizer_gradient_norm,
                             "pre_projection_norm": pre_norm, "projection_applied": projection_applied,
                             "R_norm_after": post_norm}
                if guarded:
                    trace_row.update(
                        margin_guard_applied=margin_guard_applied,
                        margin_guard_backtracks=margin_guard_backtracks,
                        margin_guard_reverted=margin_guard_reverted,
                    )
                trace.append(trace_row)
        R = (torch.diag(parameter) if diagonal else parameter).detach().clone()
        score = before if cfg.steps == 0 else float(_finite(
            _score(apply_adapter(target.features[:1], resources.U, R), resources)[0],
            "non-finite final score"))
        return _base_result(method_id, target, before, score, R=R, steps_completed=completed,
                            objective_evaluations=completed, trace=trace,
                            regularizer_name=regularizer_name,
                            fisher=params.get("fisher") if regularizer_name == "fisher" else None)
    except FloatingPointError as exc:
        return _numeric_fallback(method_id, target, before, completed, completed, exc, rank)
