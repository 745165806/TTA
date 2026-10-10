import numpy as np
import pytest
import torch

from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from experiments.p3_calibrated_teacher.calibrated_selective import run
from experiments.p3_calibrated_teacher.calibration.mixture import CalibrationFitError, fit_gmm
from experiments.p3_calibrated_teacher.scripts.followup_stats import task_gain_supported


def _params():
    return {"mu_bona": 0.0, "mu_spoof": 2.0, "var_bona": 0.04,
            "var_spoof": 0.04, "pi_bona": 0.5, "pi_spoof": 0.5, "tau_hat": 1.0}


def _resources():
    dtype = torch.float64
    U = torch.tensor([[1.0], [0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.0], dtype=dtype)
    anchors = torch.tensor([[-2.0, 0.0], [-1.0, 0.0],
                            [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w
    margins = (2 * labels.to(dtype) - 1) * scores
    return FrozenResources(U, w, 0.0, anchors, labels, margins, scores, 0.0, "p3-source")


def _target(sample_id, scores):
    return TargetViews(sample_id, torch.tensor([[x, 0.0] for x in scores],
                                               dtype=torch.float64), "p3-features")


def test_label_free_calibrator_has_ordered_components_and_rejects_collapse():
    rng = np.random.default_rng(7)
    scores = np.r_[rng.normal(-2.0, 0.25, 200), rng.normal(3.0, 0.4, 200)]
    model = fit_gmm(scores)
    assert model.mu_bona < model.tau_hat < model.mu_spoof
    assert model.posterior(model.tau_hat) == pytest.approx(0.5, abs=1e-10)
    with pytest.raises(CalibrationFitError):
        fit_gmm(np.ones(100))


def test_calibrated_episode_resets_and_abstention_is_exact_frozen():
    resources = _resources()
    cfg = EPConfig(steps=3, lr=0.03, rho=0.05)
    first = run(_target("first", [2.0, 2.1, 1.9]), resources, cfg, _params())
    repeated = run(_target("again", [2.0, 2.1, 1.9]), resources, cfg, _params())
    low = run(_target("low", [1.0, 1.0, 1.0]), resources, cfg, _params())
    assert first["adaptation_applied"] is True
    assert first["score"] == repeated["score"]
    assert torch.equal(first["R"], repeated["R"])
    assert low["abstain_reason"] == "low_teacher_confidence"
    assert low["score"] == low["score_before"]
    assert torch.count_nonzero(low["R"]) == 0


def test_paired_task_gain_requires_interval_to_exclude_zero():
    point_only = {"delta_EER_ci95": [-0.001, 0.001],
                  "delta_AUC_ci95": [-0.0001, 0.0005]}
    assert task_gain_supported(point_only) is False
    assert task_gain_supported({**point_only,
                                "delta_AUC_ci95": [0.00001, 0.0005]}) is True
