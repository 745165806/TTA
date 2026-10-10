"""Calibration-aware selective EP-TTA.

This module is deliberately separate from ``ep_tta_taskaware_v1``: P3 changes
the target calibration/teacher only, while source-anchor safety remains tied to
the source threshold ``tau0``.
"""
import math

import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_
from eptta.adaptation.objectives import (calibrated_logits, calibrated_pseudo_bce,
                                         task_logit_consistency)
from eptta.adaptation.validation import validate_inputs


METHOD_ID = "ep_tta_calibrated_selective_v1"
OBJECTIVE_DEFAULTS = {
    "lambda_pseudo": 1.0,
    "lambda_consistency": 0.25,
    "lambda_source": 1.0,
    "lambda_r": 0.01,
    "temperature": 1.0,
    "confidence_threshold": 0.90,
    "min_agreement": 1.0,
    "selection_enabled": True,
}
GMM_FIELDS = (
    "mu_bona", "mu_spoof", "var_bona", "var_spoof", "pi_bona", "pi_spoof",
    "tau_hat",
)
ALLOWED_PARAMS = frozenset((*OBJECTIVE_DEFAULTS, *GMM_FIELDS))


def validate_params(params):
    """Validate the locked P3 method parameters; labels are not an input."""
    if type(params) is not dict:
        raise ValueError("calibrated-selective params must be an object")
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        raise ValueError("unknown calibrated-selective params: %s" % sorted(unknown))
    missing = [name for name in GMM_FIELDS if name not in params]
    if missing:
        raise ValueError("calibrated-selective missing GMM params: %s" % missing)
    merged = dict(OBJECTIVE_DEFAULTS)
    merged.update(params)
    for name in GMM_FIELDS:
        value = merged[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("calibrated-selective %s must be finite" % name)
    if not merged["mu_bona"] < merged["mu_spoof"]:
        raise ValueError("calibrated-selective components must be low=bona, high=spoof")
    if merged["var_bona"] <= 0 or merged["var_spoof"] <= 0:
        raise ValueError("calibrated-selective variances must be positive")
    if merged["pi_bona"] <= 0 or merged["pi_spoof"] <= 0 or not math.isclose(
            merged["pi_bona"] + merged["pi_spoof"], 1.0, abs_tol=1e-8):
        raise ValueError("calibrated-selective mixture weights must be positive and sum to one")
    for name, locked in OBJECTIVE_DEFAULTS.items():
        value = merged[name]
        if name == "selection_enabled":
            if type(value) is not bool:
                raise ValueError("selection_enabled must be boolean")
        elif value != locked:
            raise ValueError("P3 parameter %s is locked at %s" % (name, locked))
    if not merged["mu_bona"] <= merged["tau_hat"] <= merged["mu_spoof"]:
        raise ValueError("tau_hat must lie between fitted component means")
    if abs(posterior_spoof(merged["tau_hat"], merged) - 0.5) > 1e-6:
        raise ValueError("tau_hat must be the fitted posterior 0.5 crossing")
    return merged


def posterior_spoof(score, params):
    """Return the deterministic two-Gaussian posterior P(spoof | score)."""
    if type(score) not in (int, float) or not math.isfinite(score):
        raise FloatingPointError("non-finite frozen score for calibrated teacher")
    log_bona = (math.log(params["pi_bona"]) - 0.5 * math.log(params["var_bona"]) -
                0.5 * (score - params["mu_bona"]) ** 2 / params["var_bona"])
    log_spoof = (math.log(params["pi_spoof"]) - 0.5 * math.log(params["var_spoof"]) -
                 0.5 * (score - params["mu_spoof"]) ** 2 / params["var_spoof"])
    delta = log_bona - log_spoof
    if delta >= 0:
        probability = math.exp(-delta) / (1.0 + math.exp(-delta))
    else:
        probability = 1.0 / (1.0 + math.exp(delta))
    if not math.isfinite(probability):
        raise FloatingPointError("non-finite calibrated teacher posterior")
    return probability


def _finite(value, message):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(message)
    return value


def _loss_components(R, target, resources, teacher, params):
    teacher_tensor = torch.tensor(teacher, dtype=target.features.dtype,
                                  device=target.features.device)
    adapted_views = apply_adapter(target.features, resources.U, R)
    scores = adapted_views @ resources.w + resources.b
    # P3's target objective is calibrated by tau_hat.
    logits = calibrated_logits(scores, params["tau_hat"], params["temperature"])
    pseudo = calibrated_pseudo_bce(logits, teacher_tensor)
    consistency = task_logit_consistency(logits)
    source_scores = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b
    source = (source_scores - resources.anchors_s0).square().mean()
    parameter_l2 = R.square().mean()
    return pseudo, consistency, source, parameter_l2


def _total(components, params):
    pseudo, consistency, source, parameter_l2 = components
    return (params["lambda_pseudo"] * pseudo +
            params["lambda_consistency"] * consistency +
            params["lambda_source"] * source +
            params["lambda_r"] * parameter_l2)


def _fields(components, suffix, params):
    pseudo, consistency, source, parameter_l2 = components
    return {
        "pseudo_loss_" + suffix: float(pseudo.detach()),
        "task_consistency_loss_" + suffix: float(consistency.detach()),
        "source_logit_loss_" + suffix: float(source.detach()),
        "parameter_l2_" + suffix: float(parameter_l2.detach()),
        "total_objective_" + suffix: float(_total(components, params).detach()),
    }


def _base(target, resources, before, zero_R, params, teacher, posterior, agreement, reason):
    components = _loss_components(zero_R, target, resources, teacher, params)
    result = {
        "method_id": METHOD_ID, "sample_id": target.sample_id,
        "score": before, "score_before": before, "status": "ok",
        "steps_completed": 0, "objective_evaluations": 0,
        "regularizer_name": "taskaware", "solver": "selective_sgd", "trace": [],
        "R": zero_R.clone(), "adaptation_applied": False,
        "abstain_reason": reason, "teacher_label": teacher,
        "teacher_p_spoof": posterior, "gate_confidence": max(posterior, 1.0 - posterior),
        "gate_agreement": agreement, "tau_hat": params["tau_hat"],
        "safety_rejected": False, "numeric_fallback": False,
        "source_anchor_flip_count": 0, "attempted_source_anchor_flip_count": 0,
        "final_source_anchor_flip_count": 0, "attempted_R_norm": 0.0,
        "final_R_norm": 0.0, "delta_score": 0.0,
    }
    result.update(_fields(components, "before", params))
    result.update(_fields(components, "attempted", params))
    result.update(_fields(components, "final", params))
    return result


def run(target, resources, cfg, raw_params):
    """Run one independent P3 episode using a fitted, label-free target GMM."""
    before = validate_inputs(target, resources, cfg)
    params = validate_params(raw_params)
    rank = resources.U.shape[1]
    dtype, device = target.features.dtype, target.features.device
    zero_R = torch.zeros((rank, rank), dtype=dtype, device=device)
    posterior = posterior_spoof(before, params)
    teacher = int(posterior >= 0.5)
    view_scores = target.features @ resources.w + resources.b
    view_labels = view_scores >= params["tau_hat"]
    agreement = float((view_labels == bool(teacher)).to(dtype).mean().item())
    confidence = max(posterior, 1.0 - posterior)

    if params["selection_enabled"]:
        if confidence < params["confidence_threshold"]:
            return _base(target, resources, before, zero_R, params, teacher, posterior, agreement,
                         "low_teacher_confidence")
        if agreement < params["min_agreement"]:
            return _base(target, resources, before, zero_R, params, teacher, posterior, agreement,
                         "view_disagreement")

    before_components = _loss_components(zero_R, target, resources, teacher, params)
    parameter = torch.zeros((rank, rank), dtype=dtype, device=device, requires_grad=True)
    trace = []
    completed = 0
    try:
        with torch.enable_grad():
            for step in range(cfg.steps):
                components = _loss_components(parameter, target, resources, teacher, params)
                loss = _finite(_total(components, params), "non-finite calibrated-selective loss")
                gradient, = torch.autograd.grad(loss, parameter)
                _finite(gradient, "non-finite calibrated-selective gradient")
                with torch.no_grad():
                    parameter.add_(gradient, alpha=-cfg.lr)
                    _finite(parameter, "non-finite calibrated-selective parameter")
                    project_frobenius_(parameter, cfg.rho)
                completed = step + 1
                trace.append({
                    "step": completed, "total_loss": float(loss.detach()),
                    "R_norm_after": float(torch.linalg.vector_norm(parameter)),
                })
        attempted_R = parameter.detach().clone()
        attempted_components = _loss_components(
            attempted_R, target, resources, teacher, params)
        # Source evidence is always evaluated at source tau0, never tau_hat.
        source_classes = ((apply_adapter(resources.anchors_z, resources.U, attempted_R)
                           @ resources.w + resources.b - resources.tau0) >= 0).to(torch.long)
        attempted_flips = int((source_classes != resources.anchors_y.to(torch.long)).sum().item())
        safety_rejected = attempted_flips > 0
        final_R = zero_R.clone() if safety_rejected else attempted_R
        if safety_rejected:
            score = before
        else:
            score = float(_finite(
                (apply_adapter(target.features[:1], resources.U, final_R)
                 @ resources.w + resources.b)[0],
                "non-finite calibrated-selective score"))
        final_components = _loss_components(final_R, target, resources, teacher, params)
        result = {
            "method_id": METHOD_ID, "sample_id": target.sample_id,
            "score": score, "score_before": before, "status": "ok",
            "steps_completed": completed, "objective_evaluations": completed,
            "regularizer_name": "taskaware", "solver": "selective_sgd", "trace": trace,
            "R": final_R, "adaptation_applied": not safety_rejected,
            "abstain_reason": None, "teacher_label": teacher,
            "teacher_p_spoof": posterior, "gate_confidence": confidence,
            "gate_agreement": agreement, "tau_hat": params["tau_hat"],
            "safety_rejected": safety_rejected, "numeric_fallback": False,
            "source_anchor_flip_count": attempted_flips,
            "attempted_source_anchor_flip_count": attempted_flips,
            "final_source_anchor_flip_count": 0,
            "attempted_R_norm": float(torch.linalg.vector_norm(attempted_R)),
            "final_R_norm": float(torch.linalg.vector_norm(final_R)),
            "delta_score": float(score - before),
        }
        result.update(_fields(before_components, "before", params))
        result.update(_fields(attempted_components, "attempted", params))
        result.update(_fields(final_components, "final", params))
        return result
    except FloatingPointError as exc:
        return {
            "method_id": METHOD_ID, "sample_id": target.sample_id,
            "score": before, "score_before": before, "status": "fallback_numeric",
            "steps_completed": completed, "objective_evaluations": completed,
            "regularizer_name": None, "solver": "selective_sgd", "trace": [],
            "R": zero_R, "adaptation_applied": False, "abstain_reason": None,
            "teacher_label": teacher, "teacher_p_spoof": posterior,
            "gate_confidence": confidence, "gate_agreement": agreement,
            "tau_hat": params["tau_hat"], "safety_rejected": False,
            "numeric_fallback": True, "source_anchor_flip_count": None,
            "attempted_source_anchor_flip_count": None,
            "final_source_anchor_flip_count": None, "attempted_R_norm": 0.0,
            "final_R_norm": 0.0, "delta_score": 0.0,
            "error_type": type(exc).__name__, "error_message": str(exc),
        }
