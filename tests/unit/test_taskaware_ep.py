"""Synthetic tests for the Task-Aware + Selective + Evidence-Preserving EP-TTA v1.

A skipped module is NOT_RUN, never numerical equivalence PASS.
"""
from dataclasses import fields, replace

import pytest

torch = pytest.importorskip("torch", reason="NOT_RUN: CPU PyTorch is not installed; no tensor checks executed")

from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.baselines.dispatch import run_method

torch.set_num_threads(1)

TASKAWARE = "ep_tta_taskaware_v1"
PARAMS = {"lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 1.0,
          "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.5,
          "min_agreement": 1.0}


def make_fixture(dtype=torch.float64, d=7, r=3, tau=-0.2, b=0.1):
    generator = torch.Generator().manual_seed(9)
    U = torch.linalg.qr(torch.randn(d, r, generator=generator, dtype=dtype))[0]
    w = torch.randn(d, generator=generator, dtype=dtype)
    labels = torch.arange(8) % 2
    anchors = torch.randn(8, d, generator=generator, dtype=dtype)
    signs = 2 * labels.to(dtype) - 1
    desired = tau + signs * torch.linspace(0.2, 1.0, 8, dtype=dtype)
    anchors += ((desired - anchors @ w - b) / w.square().sum())[:, None] * w
    scores = anchors @ w + b
    margins = signs * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "bundle")
    return resources


def confident_views(resources, label=1, spread=0.01, scale=1.2, seed=11):
    """Replicate a confident anchor of the requested class as three near-views."""
    generator = torch.Generator().manual_seed(seed)
    anchors = resources.anchors_z
    indices = (resources.anchors_y == label).nonzero(as_tuple=True)[0]
    base = anchors[indices[-1]]
    return base.unsqueeze(0).repeat(3, 1) + torch.randn(
        3, base.shape[0], generator=generator, dtype=base.dtype) * spread


def test_calibrated_threshold_uses_score_minus_tau0():
    """The gate and teacher must use score - tau0, never the raw score > 0 boundary."""
    dtype = torch.float64
    U = torch.tensor([[1.0], [0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.0], dtype=dtype)
    b, tau = 0.0, 0.8
    anchors = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.6, 0.0], [1.6, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w + b
    margins = (2 * labels.to(dtype) - 1) * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "cal-threshold")

    # All three views score 0.5: raw score > 0 would say "spoof", but calibrated
    # 0.5 - 0.8 = -0.3 says "bonafide".
    target = TargetViews("cal-sample", torch.tensor(
        [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=dtype), "cal-features")
    params = dict(PARAMS, confidence_margin=0.2)
    result = run_method(TASKAWARE, target, resources, EPConfig(steps=2, lr=0.03, rho=0.05), params)
    assert result["status"] == "ok"
    assert result["teacher_label"] == 0  # bonafide, not the raw-score spoof side
    assert result["gate_agreement"] == 1.0
    assert result["gate_confidence"] == pytest.approx(0.3)
    assert result["adaptation_applied"] is True


def test_abstention_on_view_disagreement_returns_frozen():
    dtype = torch.float64
    U = torch.tensor([[1.0], [0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.0], dtype=dtype)
    b, tau = 0.0, 0.8
    anchors = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.6, 0.0], [1.6, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w + b
    margins = (2 * labels.to(dtype) - 1) * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "disagree-bundle")
    # View 0 says bonafide (score 0 < tau) while views 1/2 say spoof (1.6 > tau).
    target = TargetViews("disagree", torch.tensor(
        [[0.0, 0.0], [1.6, 0.0], [1.6, 0.0]], dtype=dtype), "features")
    result = run_method(TASKAWARE, target, resources, EPConfig(steps=3, lr=0.03, rho=0.05), dict(PARAMS))
    assert result["status"] == "ok"
    assert result["adaptation_applied"] is False
    assert result["abstain_reason"] == "low_confidence_or_view_disagreement"
    assert result["score"] == result["score_before"]
    assert torch.count_nonzero(result["R"]) == 0
    assert result["numeric_fallback"] is False


def test_confident_adaptation_applies_nonzero_R():
    resources = make_fixture()
    Z = confident_views(resources, label=1)
    target = TargetViews("confident", Z, "features")
    result = run_method(TASKAWARE, target, resources, EPConfig(steps=3, lr=0.03, rho=0.05), dict(PARAMS))
    assert result["status"] == "ok"
    assert result["gate_agreement"] == 1.0
    assert result["adaptation_applied"] is True
    assert torch.count_nonzero(result["R"]) > 0
    assert result["source_anchor_flip_count"] == 0


def test_per_sample_reset_is_order_independent():
    resources = make_fixture()
    a = TargetViews("A", confident_views(resources, label=1, seed=31), "fa")
    b = TargetViews("B", confident_views(resources, label=0, seed=32), "fb")
    cfg = EPConfig(steps=3, lr=0.03, rho=0.05)
    first = run_method(TASKAWARE, a, resources, cfg, dict(PARAMS))
    run_method(TASKAWARE, b, resources, cfg, dict(PARAMS))
    again = run_method(TASKAWARE, a, resources, cfg, dict(PARAMS))
    torch.testing.assert_close(first["R"], again["R"], atol=0, rtol=0)
    assert first["score"] == again["score"]


def test_target_views_have_no_ground_truth_label():
    names = {field.name for field in fields(TargetViews)}
    assert "label" not in names and "y" not in names and "canonical_label" not in names
    resources = make_fixture()
    target = TargetViews("opaque", confident_views(resources, label=1, seed=41), "features")
    # A sidecar label dict must not be visible to or mutate the adaptation result.
    sidecar = {target.sample_id: 1}
    first = run_method(TASKAWARE, target, resources, EPConfig(steps=3, lr=0.03, rho=0.05), dict(PARAMS))
    sidecar[target.sample_id] = 0
    second = run_method(TASKAWARE, target, resources, EPConfig(steps=3, lr=0.03, rho=0.05), dict(PARAMS))
    assert first["score"] == second["score"]


def test_source_safety_reject_returns_frozen_score():
    dtype = torch.float64
    U = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.5, 1.0], dtype=dtype)
    b, tau = 0.0, 0.0
    anchors = torch.tensor([[-0.2, 0.3, 0.0], [-0.1, -0.3, 0.0],
                            [0.1, 0.3, 0.0], [0.2, -0.3, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w + b
    margins = (2 * labels.to(dtype) - 1) * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "safety-bundle")
    Z = torch.tensor([[2.0, 2.0, 0.0], [2.1, 2.0, 0.0], [1.9, 2.0, 0.0]], dtype=dtype)
    target = TargetViews("safety", Z, "features")
    # No source preservation lets a strong pseudo-target flip near-boundary anchors.
    params = dict(PARAMS, lambda_source=0.0, lambda_r=0.0)
    result = run_method(TASKAWARE, target, resources,
                        EPConfig(steps=20, lr=5.0, rho=0.9), params)
    assert result["status"] == "ok"
    assert result["safety_rejected"] is True
    assert result["source_anchor_flip_count"] > 0
    assert result["score"] == result["score_before"]
    assert result["adaptation_applied"] is False
    assert torch.count_nonzero(result["R"]) == 0


def test_old_methods_unchanged_regression():
    resources = make_fixture()
    target = TargetViews("regression", confident_views(resources, label=1, seed=51), "features")
    cfg = EPConfig(steps=3, lr=0.03, rho=0.05, gamma=0.1, lambda_keep=1.0)
    frozen = run_method("frozen", target, resources, EPConfig(steps=0), {})
    guarded = run_method("ep_tta_guarded", target, resources, cfg, {})
    plain = run_method("ep_tta", target, resources, cfg, {})
    assert frozen["score"] == frozen["score_before"]
    assert guarded["method_id"] == "ep_tta_guarded"
    assert plain["method_id"] == "ep_tta"
    # The old view-variance mechanisms must not gain task-aware fields.
    for result in (frozen, guarded, plain):
        assert "adaptation_applied" not in result


def _recompute_components(target, resources, teacher, params, R):
    """Independently recompute the four P1 loss components at R (no adapter internals)."""
    from eptta.adaptation.math import apply_adapter
    from eptta.adaptation.objectives import (calibrated_logits, calibrated_pseudo_bce,
                                             task_logit_consistency)
    teacher_t = torch.tensor(teacher, dtype=target.features.dtype, device=target.features.device)
    adapted = apply_adapter(target.features, resources.U, R)
    u = calibrated_logits(adapted @ resources.w + resources.b, resources.tau0,
                          params["temperature"])
    pseudo = calibrated_pseudo_bce(u, teacher_t)
    consistency = task_logit_consistency(u)
    source_scores = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b
    source = (source_scores - resources.anchors_s0).square().mean()
    parameter_l2 = R.square().mean()
    return pseudo, consistency, source, parameter_l2


def test_final_loss_matches_returned_R():
    resources = make_fixture()
    target = TargetViews("confident-final", confident_views(resources, label=1, seed=61), "features")
    cfg = EPConfig(steps=3, lr=0.03, rho=0.05)
    result = run_method(TASKAWARE, target, resources, cfg, dict(PARAMS))
    assert result["adaptation_applied"] is True
    pseudo, consistency, source, parameter_l2 = _recompute_components(
        target, resources, result["teacher_label"], dict(PARAMS), result["R"])
    assert result["pseudo_loss_final"] == pytest.approx(float(pseudo), rel=1e-9, abs=1e-12)
    assert result["task_consistency_loss_final"] == pytest.approx(float(consistency), rel=1e-9, abs=1e-12)
    assert result["source_logit_loss_final"] == pytest.approx(float(source), rel=1e-9, abs=1e-12)
    assert result["parameter_l2_final"] == pytest.approx(float(parameter_l2), rel=1e-9, abs=1e-12)


def test_safety_reject_final_loss_is_R_zero():
    dtype = torch.float64
    U = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.5, 1.0], dtype=dtype)
    b, tau = 0.0, 0.0
    anchors = torch.tensor([[-0.2, 0.3, 0.0], [-0.1, -0.3, 0.0],
                            [0.1, 0.3, 0.0], [0.2, -0.3, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w + b
    margins = (2 * labels.to(dtype) - 1) * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "safety-final")
    Z = torch.tensor([[2.0, 2.0, 0.0], [2.1, 2.0, 0.0], [1.9, 2.0, 0.0]], dtype=dtype)
    target = TargetViews("safety-final", Z, "features")
    params = dict(PARAMS, lambda_source=0.0, lambda_r=0.0)
    result = run_method(TASKAWARE, target, resources, EPConfig(steps=20, lr=5.0, rho=0.9), params)

    assert result["safety_rejected"] is True
    assert result["score"] == result["score_before"]
    assert torch.count_nonzero(result["R"]) == 0

    # attempted state is the unsafe candidate (non-zero R, anchor crossings).
    assert result["attempted_R_norm"] > 0
    assert result["attempted_source_anchor_flip_count"] > 0
    assert result["final_source_anchor_flip_count"] == 0

    # final losses must be recomputed at R=0, not the unsafe attempted R.
    zero_R = torch.zeros_like(result["R"])
    pseudo0, consistency0, source0, l2_0 = _recompute_components(
        target, resources, result["teacher_label"], params, zero_R)
    assert result["pseudo_loss_final"] == pytest.approx(float(pseudo0), rel=1e-9, abs=1e-12)
    assert result["task_consistency_loss_final"] == pytest.approx(float(consistency0), rel=1e-9, abs=1e-12)
    assert result["source_logit_loss_final"] == pytest.approx(float(source0), rel=1e-9, abs=1e-12)
    assert result["parameter_l2_final"] == pytest.approx(float(l2_0), rel=1e-9, abs=1e-12)
    # attempted losses differ from final (the unsafe candidate was not R=0).
    assert result["attempted_pseudo_loss"] != result["pseudo_loss_final"]
