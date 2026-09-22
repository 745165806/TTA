"""P3 calibration-aware selective EP-TTA contract tests."""
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.objectives import calibrated_logits, calibrated_pseudo_bce
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.baselines.dispatch import run_method
from eptta.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/p3_calibrated_teacher"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "scripts"))
from calibration.mixture import CalibrationFitError, fit_gmm
from scripts.split_target10 import split_ids
from workers.run_variant import frozen_record


METHOD = "ep_tta_calibrated_selective_v1"


def params(tau_hat=1.0, selection=True, mu_bona=0.0, mu_spoof=2.0):
    return {
        "mu_bona": mu_bona, "mu_spoof": mu_spoof,
        "var_bona": 0.04, "var_spoof": 0.04,
        "pi_bona": 0.5, "pi_spoof": 0.5, "tau_hat": tau_hat,
        "lambda_pseudo": 1.0, "lambda_consistency": 0.25,
        "lambda_source": 1.0, "lambda_r": 0.01, "temperature": 1.0,
        "confidence_threshold": 0.9, "min_agreement": 1.0,
        "selection_enabled": selection,
    }


def fixture(tau0=0.0):
    dtype = torch.float64
    U = torch.tensor([[1.0], [0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.0], dtype=dtype)
    anchors = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w
    margins = (2 * labels.to(dtype) - 1) * (scores - tau0)
    return FrozenResources(U, w, 0.0, anchors, labels, margins, scores, tau0, "p3-test")


def target(scores, sample_id="sample"):
    return TargetViews(sample_id, torch.tensor([[s, 0.0] for s in scores], dtype=torch.float64),
                       "p3-features")


def test_split_is_deterministic_and_input_order_invariant():
    ids = ["id-%03d" % i for i in range(101)]
    first = split_ids(ids, "p3-main-2026")
    second = split_ids(list(reversed(ids)), "p3-main-2026")
    assert first == second
    assert len(first[0]) == 10 and set(first[0]).isdisjoint(first[1])


def test_calibrator_api_is_label_free_and_component_semantics():
    assert "labels" not in inspect.signature(fit_gmm).parameters
    rng = np.random.default_rng(7)
    scores = np.r_[rng.normal(-2.0, 0.25, 200), rng.normal(3.0, 0.4, 200)]
    model = fit_gmm(scores)
    assert model.converged and model.mu_bona < model.mu_spoof
    assert model.posterior(model.mu_bona) < 0.5
    assert model.posterior(model.mu_spoof) > 0.5
    assert model.posterior(model.tau_hat) == pytest.approx(0.5, abs=1e-10)
    with pytest.raises(TypeError):
        fit_gmm(scores, labels=np.zeros(len(scores)))


def test_calibrator_collapse_fails_closed():
    with pytest.raises(CalibrationFitError):
        fit_gmm(np.ones(100))


def test_high_confidence_agreement_adapts_and_uses_tau_hat_objective():
    resources = fixture()
    sample = target([2.0, 2.1, 1.9])
    result = run_method(METHOD, sample, resources, EPConfig(steps=3, lr=0.03, rho=0.05), params())
    assert result["adaptation_applied"] is True
    assert result["gate_agreement"] == 1.0
    assert result["gate_confidence"] >= 0.9
    teacher = torch.tensor(1.0, dtype=sample.features.dtype)
    expected = calibrated_pseudo_bce(
        calibrated_logits(sample.features @ resources.w + resources.b, 1.0, 1.0), teacher)
    wrong = calibrated_pseudo_bce(
        calibrated_logits(sample.features @ resources.w + resources.b, resources.tau0, 1.0), teacher)
    assert result["pseudo_loss_before"] == pytest.approx(float(expected))
    assert result["pseudo_loss_before"] != pytest.approx(float(wrong))


def test_low_confidence_and_view_disagreement_abstain_exactly():
    resources = fixture()
    low = run_method(METHOD, target([1.0, 1.0, 1.0], "low"), resources,
                     EPConfig(steps=3, lr=0.03, rho=0.05), params())
    assert low["abstain_reason"] == "low_teacher_confidence"
    assert low["score"] == low["score_before"] and torch.count_nonzero(low["R"]) == 0
    disagree = run_method(METHOD, target([2.0, 0.0, 2.0], "disagree"), resources,
                          EPConfig(steps=3, lr=0.03, rho=0.05), params())
    assert disagree["abstain_reason"] == "view_disagreement"
    assert disagree["score"] == disagree["score_before"] and torch.count_nonzero(disagree["R"]) == 0


def test_calibration_fit_fail_disables_adaptation_exactly():
    record = frozen_record("CalibratedSelective", "x", 1.25, "calibration_fit_fail")
    assert record["score_after"] == record["score_before"] == 1.25
    assert record["adaptation_applied"] is False
    assert record["abstain_reason"] == "calibration_fit_fail"


def test_source_safety_reject_is_exact_frozen(monkeypatch):
    dtype = torch.float64
    U = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.5, 1.0], dtype=dtype)
    anchors = torch.tensor([[-0.2, 0.3, 0.0], [-0.1, -0.3, 0.0],
                            [0.1, 0.3, 0.0], [0.2, -0.3, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w
    margins = (2 * labels.to(dtype) - 1) * scores
    resources = FrozenResources(U, w, 0.0, anchors, labels, margins, scores, 0.0,
                                "p3-safety")
    unsafe = torch.tensor([[0.25077886917760434, 0.34602217042636363],
                           [0.4226489741089841, 0.6698854088812906]], dtype=dtype)

    def force_unsafe(parameter, _rho):
        parameter.copy_(unsafe)
        return parameter

    monkeypatch.setattr("eptta.adaptation.calibrated_selective.project_frobenius_", force_unsafe)
    result = run_method(METHOD, TargetViews("unsafe", torch.tensor(
        [[2.0, 2.0, 0.0], [2.1, 2.0, 0.0], [1.9, 2.0, 0.0]], dtype=dtype), "features"),
        resources, EPConfig(steps=1, lr=0.03, rho=0.9), params())
    assert result["safety_rejected"] is True
    assert result["source_anchor_flip_count"] > 0
    assert result["score"] == result["score_before"]
    assert result["adaptation_applied"] is False
    assert torch.count_nonzero(result["R"]) == 0


def test_source_safety_is_source_tau0_not_target_tau_hat():
    resources = fixture(tau0=0.0)
    result = run_method(METHOD, target([3.0, 3.1, 2.9]), resources,
                        EPConfig(steps=3, lr=0.03, rho=0.05),
                        params(tau_hat=1.5, mu_bona=0.0, mu_spoof=3.0))
    # At target tau_hat=1.5, the source spoof anchor at score 1 would be called
    # bonafide. Correct source safety uses tau0=0 and accepts the safe episode.
    assert result["adaptation_applied"] is True
    assert result["final_source_anchor_flip_count"] == 0


def test_calonly_ranking_invariant():
    scores = [-2.0, -1.0, 1.0, 2.0]
    labels = [0, 0, 1, 1]
    frozen = binary_metrics(scores, labels, 0.0)
    calonly = binary_metrics(scores, labels, 0.7)
    assert calonly["eer"] == frozen["eer"]
    assert calonly["auroc"] == frozen["auroc"]


def test_old_taskaware_v1_regression_unchanged():
    resources = fixture()
    sample = target([2.0, 2.1, 1.9], "old")
    old_params = {"lambda_pseudo": 1.0, "lambda_consistency": 0.25,
                  "lambda_source": 1.0, "lambda_r": 0.01, "temperature": 1.0,
                  "confidence_margin": 0.5, "min_agreement": 1.0}
    result = run_method("ep_tta_taskaware_v1", sample, resources,
                        EPConfig(steps=3, lr=0.03, rho=0.05), old_params)
    assert result["method_id"] == "ep_tta_taskaware_v1"
    assert result["teacher_label"] == 1
    assert "teacher_p_spoof" not in result
