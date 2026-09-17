import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.execution.r6 import compute_diagnostics, grid_candidates, verify_k_gradient


def _fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(11)
    d, r = 7, 3
    U = torch.linalg.qr(torch.randn(d, r, generator=generator, dtype=dtype))[0]
    w = torch.randn(d, generator=generator, dtype=dtype)
    labels = torch.arange(8) % 2
    anchors = torch.randn(8, d, generator=generator, dtype=dtype)
    b, tau = 0.1, -0.2
    signs = 2 * labels.to(dtype) - 1
    anchors += ((tau + signs - anchors @ w - b) / w.square().sum())[:, None] * w
    scores = anchors @ w + b
    margins = signs * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "bundle")
    targets = [TargetViews("s%d" % i, torch.randn(3, d, generator=generator, dtype=dtype), "f") for i in range(3)]
    return targets, resources


def test_grid_is_nine_candidates_and_eta_uses_ref():
    candidates = grid_candidates()
    assert len(candidates) == 9
    etas = {c["eta"] for c in candidates}
    assert etas == {0.003, 0.01, 0.03}
    ks = {c["K"] for c in candidates}
    assert ks == {1, 3, 5}


def test_k_gradient_reference_matches_production_fp64():
    targets, resources = _fixture(torch.float64)
    cfg = EPConfig(steps=1, lr=0.01, rho=0.2, gamma=0.1, lambda_keep=1.0)
    report = verify_k_gradient(targets, resources, cfg)
    assert report["max_abs_grad_err"] < 1e-8
    for row in report["per_sample"]:
        assert row["steps_completed"] == 1
        assert row["status"] == "ok"


def test_k_gradient_reference_matches_production_fp32():
    targets, resources = _fixture(torch.float32)
    cfg = EPConfig(steps=1, lr=0.01, rho=0.2, gamma=0.1, lambda_keep=1.0)
    report = verify_k_gradient(targets, resources, cfg)
    assert report["max_abs_grad_err"] < 1e-3


def test_diagnostics_are_finite_and_bounded():
    from eptta.adaptation.adapter import run_cache_method
    targets, resources = _fixture(torch.float32)
    cfg = EPConfig(steps=2, lr=0.01, rho=0.2, gamma=0.1, lambda_keep=1.0)
    result = run_cache_method("ep_tta", targets[0], resources, cfg)
    diag = compute_diagnostics(result, targets[0], resources, cfg)
    for key in ("r_fro", "delta_z_norm", "delta_score", "margin_violation_fraction", "worst_margin_change"):
        assert key in diag and diag[key] is not None
    assert diag["r_fro"] <= cfg.rho + 1e-6
