"""Cache-based mechanism dispatch with fresh, isolated per-sample state."""
import math

import torch

from eptta.adaptation.math import apply_adapter, margin_deficit, margin_tolerance, project_frobenius_
from eptta.adaptation.objectives import (calibrated_logits, calibrated_pseudo_bce,
                                         target_objective, task_logit_consistency)
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

# Task-aware EP-TTA v1 hyperparameters.  They intentionally ride on method.params
# (not EPConfig) so the shared per-sample reset/Frobenius geometry stays untouched.
TASKAWARE_PARAMS = (
    "lambda_pseudo", "lambda_consistency", "lambda_source", "lambda_r",
    "temperature", "confidence_margin", "min_agreement",
)
_TASKAWARE_DEFAULTS = {
    "lambda_pseudo": 1.0,
    "lambda_consistency": 0.25,
    "lambda_source": 1.0,
    "lambda_r": 0.01,
    "temperature": 1.0,
    "confidence_margin": 0.5,
    "min_agreement": 1.0,
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
                      "ep_keep_fisher": {"fisher"}, "ep_tta_taskaware_v1": set(TASKAWARE_PARAMS)}
    unknown = set(params) - allowed_params.get(method_id, set())
    if unknown:
        raise ValueError("unknown params for %s: %s" % (method_id, sorted(unknown)))
    if method_id == "ep_tta_taskaware_v1":
        merged = dict(_TASKAWARE_DEFAULTS)
        merged.update(params)
        for name in ("lambda_pseudo", "lambda_consistency", "lambda_source", "lambda_r"):
            value = merged[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("task-aware %s must be a finite nonnegative scalar" % name)
        temperature = merged["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("task-aware temperature must be a finite positive scalar")
        confidence_margin = merged["confidence_margin"]
        if type(confidence_margin) not in (int, float) or not math.isfinite(confidence_margin) or confidence_margin < 0:
            raise ValueError("task-aware confidence_margin must be a finite nonnegative scalar")
        if not isinstance(merged["min_agreement"], (int, float)) or not math.isfinite(merged["min_agreement"]) or \
                not 0 <= merged["min_agreement"] <= 1:
            raise ValueError("task-aware min_agreement must be in [0,1]")
        return merged
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
    if result.get("method_id") == "ep_tta_taskaware_v1":
        return _taskaware_diagnostics(result, target, resources)
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


def _taskaware_numeric_fallback(target, before, completed, evaluations, exc, rank):
    result = _numeric_fallback("ep_tta_taskaware_v1", target, before, completed, evaluations, exc, rank)
    result.update({
        "adaptation_applied": False, "abstain_reason": None, "teacher_label": None,
        "gate_agreement": None, "gate_confidence": None, "numeric_fallback": True,
        "solver": "selective_sgd",
        "pseudo_loss_before": None, "task_consistency_loss_before": None,
        "source_logit_loss_before": None, "parameter_l2_before": None,
        "total_objective_before": None,
        "pseudo_loss_final": None, "task_consistency_loss_final": None,
        "source_logit_loss_final": None, "parameter_l2_final": None,
        "total_objective_final": None,
        "attempted_pseudo_loss": None, "attempted_task_consistency_loss": None,
        "attempted_source_logit_loss": None, "attempted_parameter_l2": None,
        "attempted_total_objective": None,
        "attempted_R_norm": 0.0, "attempted_source_anchor_flip_count": None,
        "source_anchor_flip_count": None, "final_source_anchor_flip_count": None,
        "final_R_norm": 0.0, "safety_rejected": False, "delta_score": 0.0,
    })
    return result


def _taskaware_gate_state(target, resources):
    """Compute the frozen teacher, per-view agreement and confidence.

    The teacher is the original-view class at the *calibrated* threshold tau0
    (``score - tau0 >= 0``), never the raw ``score > 0`` boundary.
    """
    m = target.features @ resources.w + resources.b - resources.tau0
    teacher = int(bool((m[0] >= 0).item()))
    agreement = float(((m >= 0) == teacher).to(target.features.dtype).mean().item())
    confidence = float(m.abs().min().item())
    return teacher, agreement, confidence


def _taskaware_loss_components(R, target, resources, teacher, params):
    """Return (pseudo, consistency, source, parameter_l2) at the given R.

    This is the single source of truth for the P1 objective math; the SGD loop
    and the final/attempted diagnostics both call it so they cannot drift.
    """
    teacher_tensor = torch.tensor(teacher, dtype=target.features.dtype, device=target.features.device)
    adapted_views = apply_adapter(target.features, resources.U, R)
    s_adapt = adapted_views @ resources.w + resources.b
    u = calibrated_logits(s_adapt, resources.tau0, params["temperature"])
    pseudo = calibrated_pseudo_bce(u, teacher_tensor)
    consistency = task_logit_consistency(u)
    source_scores = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b
    source = (source_scores - resources.anchors_s0).square().mean()
    parameter_l2 = R.square().mean()
    return pseudo, consistency, source, parameter_l2


def _taskaware_total(components, params):
    pseudo, consistency, source, parameter_l2 = components
    return (params["lambda_pseudo"] * pseudo +
            params["lambda_consistency"] * consistency +
            params["lambda_source"] * source +
            params["lambda_r"] * parameter_l2)


def _run_taskaware_v1(target, resources, cfg, params):
    """Task-Aware + Selective + Evidence-Preserving EP-TTA v1.

    Gate high-confidence, view-consistent samples; otherwise abstain (return the
    frozen score with R=0).  When adapting, optimize calibrated pseudo-BCE plus
    task-logit consistency, soft source-logit preservation and parameter L2, then
    apply one final source-anchor class-crossing safety check (no backtracking).

    Loss semantics: ``*_before`` is computed at R=0 (the frozen state),
    ``attempted_*`` at the SGD-completed candidate R *before* the safety check,
    and ``*_final`` at the R actually returned to the caller (R=0 when rejected).
    ``source_anchor_flip_count`` is kept as the attempted (pre-reject) count for
    backward compatibility; ``final_source_anchor_flip_count`` is always 0.
    """
    before = validate_inputs(target, resources, cfg)
    params = validate_method_setup("ep_tta_taskaware_v1", resources, cfg, params)
    rank = resources.U.shape[1]
    device, dtype = target.features.device, target.features.dtype
    zero_R = torch.zeros((rank, rank), dtype=dtype, device=device)
    teacher, agreement, confidence = _taskaware_gate_state(target, resources)
    common = {
        "method_id": "ep_tta_taskaware_v1", "regularizer_name": "taskaware",
        "solver": "selective_sgd", "trace": [],
        "teacher_label": teacher, "gate_agreement": agreement,
        "gate_confidence": confidence, "numeric_fallback": False,
    }

    def _components_to_fields(components, pseudo_key, consistency_key, source_key, l2_key):
        pseudo, consistency, source, parameter_l2 = components
        return {
            pseudo_key: float(pseudo.detach()),
            consistency_key: float(consistency.detach()),
            source_key: float(source.detach()),
            l2_key: float(parameter_l2.detach()),
        }

    def _before_fields(components):
        fields = _components_to_fields(components, "pseudo_loss_before",
                                       "task_consistency_loss_before",
                                       "source_logit_loss_before", "parameter_l2_before")
        fields["total_objective_before"] = float(_taskaware_total(components, params).detach())
        return fields

    def _final_fields(components):
        fields = _components_to_fields(components, "pseudo_loss_final",
                                       "task_consistency_loss_final",
                                       "source_logit_loss_final", "parameter_l2_final")
        fields["total_objective_final"] = float(_taskaware_total(components, params).detach())
        return fields

    def _attempted_fields(components):
        fields = _components_to_fields(components, "attempted_pseudo_loss",
                                       "attempted_task_consistency_loss",
                                       "attempted_source_logit_loss", "attempted_parameter_l2")
        fields["attempted_total_objective"] = float(_taskaware_total(components, params).detach())
        return fields

    before_components = _taskaware_loss_components(zero_R, target, resources, teacher, params)

    # Selective gate: abstain (never a numeric fallback) unless every view agrees
    # with the frozen teacher and the sample is confidently separated from tau0.
    if agreement < params["min_agreement"] or confidence < params["confidence_margin"]:
        result = _base_result("ep_tta_taskaware_v1", target, before, before, R=zero_R.clone(),
                              adaptation_applied=False,
                              abstain_reason="low_confidence_or_view_disagreement",
                              teacher_label=teacher, gate_agreement=agreement,
                              gate_confidence=confidence, safety_rejected=False,
                              delta_score=0.0, final_R_norm=0.0,
                              source_anchor_flip_count=0,
                              attempted_R_norm=0.0, attempted_source_anchor_flip_count=0,
                              final_source_anchor_flip_count=0)
        result.update(common)
        result.update(_before_fields(before_components))
        result.update(_final_fields(before_components))
        result.update(_attempted_fields(before_components))
        return result

    parameter = torch.zeros((rank, rank), dtype=dtype, device=device, requires_grad=True)
    completed = 0
    try:
        with torch.enable_grad():
            for step in range(cfg.steps):
                pseudo, consistency, source, parameter_l2 = _taskaware_loss_components(
                    parameter, target, resources, teacher, params)
                loss = _taskaware_total((pseudo, consistency, source, parameter_l2), params)
                loss = _finite(loss, "non-finite task-aware loss")
                gradient, = torch.autograd.grad(loss, parameter)
                _finite(gradient, "non-finite task-aware gradient")
                with torch.no_grad():
                    parameter.add_(gradient, alpha=-cfg.lr)
                    _finite(parameter, "non-finite task-aware parameter before projection")
                    project_frobenius_(parameter, cfg.rho)
                    _finite(parameter, "non-finite task-aware parameter after projection")
                completed = step + 1
                common["trace"].append({
                    "step": completed, "pseudo_loss": float(pseudo.detach()),
                    "task_consistency_loss": float(consistency.detach()),
                    "source_logit_loss": float(source.detach()),
                    "parameter_l2": float(parameter_l2.detach()),
                    "total_loss": float(loss.detach()),
                    "R_norm_after": float(torch.linalg.vector_norm(parameter)),
                })
        attempted_R = parameter.detach().clone()
        attempted_components = _taskaware_loss_components(
            attempted_R, target, resources, teacher, params)
        attempted_flips = int(((apply_adapter(resources.anchors_z, resources.U, attempted_R)
                                @ resources.w + resources.b - resources.tau0 >= 0).to(torch.long) !=
                               resources.anchors_y.to(torch.long)).sum().item())

        # Final one-shot source safety check: no source anchor may cross class.
        if attempted_flips > 0:
            final_R = zero_R.clone()
            score = before
            adaptation_applied, safety_rejected = False, True
        else:
            final_R = attempted_R
            score = float(_finite(
                (apply_adapter(target.features[:1], resources.U, final_R) @ resources.w + resources.b)[0],
                "non-finite task-aware final score"))
            adaptation_applied, safety_rejected = True, False

        final_components = _taskaware_loss_components(final_R, target, resources, teacher, params)
        result = _base_result("ep_tta_taskaware_v1", target, before, score, R=final_R,
                              steps_completed=completed, objective_evaluations=completed,
                              trace=common["trace"], regularizer_name="taskaware",
                              adaptation_applied=adaptation_applied,
                              abstain_reason=None, teacher_label=teacher,
                              gate_agreement=agreement, gate_confidence=confidence,
                              source_anchor_flip_count=attempted_flips,
                              safety_rejected=safety_rejected,
                              final_R_norm=float(torch.linalg.vector_norm(final_R)),
                              final_source_anchor_flip_count=0,
                              attempted_R_norm=float(torch.linalg.vector_norm(attempted_R)),
                              attempted_source_anchor_flip_count=attempted_flips,
                              delta_score=float(score - before))
        result.update(_before_fields(before_components))
        result.update(_final_fields(final_components))
        result.update(_attempted_fields(attempted_components))
        return result
    except FloatingPointError as exc:
        return _taskaware_numeric_fallback(target, before, completed, completed, exc, rank)


_TASKAWARE_LOSS_FIELDS = (
    "pseudo_loss_before", "task_consistency_loss_before", "source_logit_loss_before",
    "parameter_l2_before", "total_objective_before",
    "pseudo_loss_final", "task_consistency_loss_final", "source_logit_loss_final",
    "parameter_l2_final", "total_objective_final",
    "attempted_pseudo_loss", "attempted_task_consistency_loss",
    "attempted_source_logit_loss", "attempted_parameter_l2", "attempted_total_objective",
)


@torch.no_grad()
def _taskaware_diagnostics(result, target, resources):
    base = {
        "completed_steps": int(result["steps_completed"]),
        "objective_evaluations": int(result["objective_evaluations"]),
        "adaptation_applied": bool(result.get("adaptation_applied", False)),
        "abstain_reason": result.get("abstain_reason"),
        "teacher_label": result.get("teacher_label"),
        "gate_agreement": result.get("gate_agreement"),
        "gate_confidence": result.get("gate_confidence"),
        "safety_rejected": bool(result.get("safety_rejected", False)),
        "source_anchor_flip_count": result.get("source_anchor_flip_count"),
        "attempted_source_anchor_flip_count": result.get("attempted_source_anchor_flip_count"),
        "final_source_anchor_flip_count": result.get("final_source_anchor_flip_count"),
    }
    if result.get("status") == "fallback_numeric":
        base.update({
            "gradient_trace_status": "failed_numeric",
            "delta_score": 0.0, "delta_z_norm": None,
            "final_R_norm": 0.0, "attempted_R_norm": 0.0,
        })
        for name in _TASKAWARE_LOSS_FIELDS:
            base[name] = None
        return base
    R = result.get("R")
    if R is None:
        rank = resources.U.shape[1]
        R = torch.zeros((rank, rank), dtype=target.features.dtype, device=target.features.device)
    _finite(R, "non-finite final R")
    delta = apply_adapter(target.features[:1], resources.U, R) - target.features[:1]
    base.update({
        "gradient_trace_status": "recorded",
        "delta_score": float(result["score"] - result["score_before"]),
        "delta_z_norm": float(torch.linalg.vector_norm(delta)),
        "final_R_norm": float(torch.linalg.vector_norm(R)),
        "attempted_R_norm": result.get("attempted_R_norm"),
    })
    for name in _TASKAWARE_LOSS_FIELDS:
        base[name] = result.get(name)
    return base


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
    if method_id == "ep_tta_taskaware_v1":
        return _run_taskaware_v1(target, resources, cfg, params)

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
