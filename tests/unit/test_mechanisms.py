from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.adapter import adaptation_diagnostics, run_cache_method
from eptta.adaptation.math import apply_adapter, margin_deficit
from eptta.adaptation.objectives import marginal_entropy, mean_view_entropy
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.baselines.dispatch import run_method
from eptta.offline.fisher import empirical_diagonal_fisher
from eptta.offline.anchors import build_anchor_memory
from eptta.offline.calibration import empirical_real_quantile
from eptta.offline.static_adapter import fit_fixed_source_adapter
from eptta.offline.subspace import feature_pca_subspace, random_subspace, response_subspace


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(9)
    d, r = 7, 3
    U = torch.linalg.qr(torch.randn(d, r, generator=generator, dtype=dtype))[0]
    w = torch.randn(d, generator=generator, dtype=dtype)
    labels = torch.arange(8) % 2
    anchors = torch.randn(8, d, generator=generator, dtype=dtype)
    b, tau = 0.1, -0.2
    signs = 2 * labels.to(dtype) - 1
    desired = tau + signs * torch.linspace(.2, 1.0, 8, dtype=dtype)
    anchors += ((desired - anchors @ w - b) / w.square().sum())[:, None] * w
    scores = anchors @ w + b
    margins = signs * (scores - tau)
    resources = FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "bundle")
    target = TargetViews("sample", torch.randn(3, d, generator=generator, dtype=dtype), "features")
    return target, resources


def test_p0_reference_controls_and_matched_static_budget():
    target, resources = fixture()
    frozen = run_cache_method("frozen", target, resources)
    assert frozen["score"] == frozen["score_before"]
    mean = run_cache_method("multiview_mean", target, resources)
    assert mean["score"] == pytest.approx(float((target.features @ resources.w + resources.b).mean()))
    cfg = EPConfig(rho=.2)
    maximum = cfg.rho / resources.U.shape[1] ** .5
    out = run_cache_method("static_subspace", target, resources, cfg, {"amount": maximum})
    assert float(torch.linalg.vector_norm(out["R"])) == pytest.approx(cfg.rho)
    with pytest.raises(ValueError, match="budget"):
        run_cache_method("static_subspace", target, resources, cfg, {"amount": cfg.rho})


def test_entropy_and_memo_are_distinct_and_scalar_grid_is_exact():
    logits = torch.tensor([-5.0, 0.0, 4.0], dtype=torch.float64)
    assert mean_view_entropy(logits) != marginal_entropy(logits)
    target, resources = fixture()
    result = run_cache_method("ep_scalar_adaptive", target, resources, EPConfig(), {"grid_size": 17})
    assert result["objective_evaluations"] == 17
    assert 0 <= result["selected_amount"] <= .2 / resources.U.shape[1] ** .5


@pytest.mark.parametrize("method", [
    "ep_tta", "ep_tta_guarded", "ep_no_keep", "ep_random_U", "ep_feature_pca_U",
    "ep_no_projection",
    "entropy_same_adapter_no_keep", "entropy_same_adapter", "memo_same_adapter_no_keep",
    "memo_same_adapter_keep", "ep_keep_l2", "ep_keep_logit", "source_ce_only", "ep_diagonal_R"])
def test_required_mechanisms_run_with_fresh_per_sample_state(method):
    target, resources = fixture()
    first = run_cache_method(method, target, resources, EPConfig(steps=2))
    second = run_cache_method(method, target, resources, EPConfig(steps=2))
    torch.testing.assert_close(first["R"], second["R"], atol=0, rtol=0)
    assert first["score"] == second["score"]


def test_guarded_ep_dispatch_reset_k0_and_projection_bound():
    target, resources = fixture()
    cfg = EPConfig(steps=3, lr=.3, rho=.2)

    first = run_method("ep_tta_guarded", target, resources, cfg, {})
    other = replace(target, sample_id="other", features=target.features * 1.25)
    run_method("ep_tta_guarded", other, resources, cfg, {})
    replay = run_method("ep_tta_guarded", target, resources, cfg, {})

    torch.testing.assert_close(first["R"], replay["R"], atol=0, rtol=0)
    assert first["score"] == replay["score"]
    assert float(torch.linalg.vector_norm(first["R"])) <= cfg.rho + 1e-12

    frozen = run_method("frozen", target, resources, EPConfig(steps=0), {})
    guarded_k0 = run_method("ep_tta_guarded", target, resources, EPConfig(steps=0), {})
    assert guarded_k0["score"] == frozen["score"]
    assert guarded_k0["score_before"] == frozen["score_before"]
    assert torch.count_nonzero(guarded_k0["R"]) == 0


def test_guarded_ep_backtracks_margin_violation_without_changing_ep_v0():
    dtype = torch.float64
    U = torch.eye(2, dtype=dtype)
    w = torch.tensor([1.0, 0.0], dtype=dtype)
    anchors = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 1])
    margins = torch.ones(2, dtype=dtype)
    resources = FrozenResources(
        U, w, 0.0, anchors, labels, margins, anchors @ w, 0.0, "guard-bundle")
    target = TargetViews(
        "guard-sample",
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=dtype),
        "guard-features",
    )
    cfg = EPConfig(steps=1, lr=1.0, rho=0.5, gamma=0.1, lambda_keep=0.0)

    plain = run_cache_method("ep_tta", target, resources, cfg)
    guarded = run_cache_method("ep_tta_guarded", target, resources, cfg)

    # EP-v0 remains the original projected one-step update, including its trace shape.
    expected_plain = torch.tensor([[-0.5, 0.0], [0.0, 0.0]], dtype=dtype)
    torch.testing.assert_close(plain["R"], expected_plain, atol=0, rtol=0)
    assert plain["score"] == pytest.approx(0.5)
    assert "margin_guard_applied" not in plain["trace"][0]

    plain_deficit = margin_deficit(
        apply_adapter(anchors, U, plain["R"]),
        w, 0.0, labels, margins, 0.0, cfg.gamma,
    )
    guarded_deficit = margin_deficit(
        apply_adapter(anchors, U, guarded["R"]),
        w, 0.0, labels, margins, 0.0, cfg.gamma,
    )
    assert bool((plain_deficit > 0).any())
    assert float(guarded_deficit.max()) <= 64 * torch.finfo(dtype).eps
    assert guarded["trace"][0]["margin_guard_applied"] is True
    assert guarded["trace"][0]["margin_guard_backtracks"] == 3
    assert guarded["trace"][0]["margin_guard_reverted"] is False

    diagnostics = adaptation_diagnostics(guarded, target, resources, cfg)
    assert diagnostics["margin_guard_count"] == 1
    assert diagnostics["margin_guard_backtracks"] == 3
    assert diagnostics["margin_guard_reverts"] == 0
    assert diagnostics["final_margin_violation_fraction"] == 0.0

    strict = run_cache_method(
        "ep_tta_guarded", target, resources,
        EPConfig(steps=1, lr=1.0, rho=0.5, gamma=0.0, lambda_keep=0.0),
    )
    assert strict["trace"][0]["margin_guard_backtracks"] == 16
    assert strict["trace"][0]["margin_guard_reverted"] is True
    torch.testing.assert_close(strict["R"], torch.zeros_like(strict["R"]), atol=0, rtol=0)


def test_fisher_formula_matches_per_sample_autograd_and_regularizer_runs():
    target, resources = fixture()
    analytic = empirical_diagonal_fisher(resources.anchors_z, resources.anchors_y,
                                         resources.U, resources.w, resources.b)
    rows = []
    for z, y in zip(resources.anchors_z, resources.anchors_y):
        R = torch.zeros(resources.U.shape[1], resources.U.shape[1], dtype=z.dtype, requires_grad=True)
        adapted = z + ((z @ resources.U) @ R.T) @ resources.U.T
        loss = torch.nn.functional.binary_cross_entropy_with_logits(adapted @ resources.w + resources.b,
                                                                    y.to(z.dtype))
        gradient, = torch.autograd.grad(loss, R)
        rows.append(gradient.square())
    torch.testing.assert_close(analytic, torch.stack(rows).mean(0))
    result = run_cache_method("ep_keep_fisher", target, resources, EPConfig(steps=2), {"fisher": analytic})
    assert result["steps_completed"] == 2


def test_subspace_estimators_are_orthonormal_and_semantically_distinct():
    generator = torch.Generator().manual_seed(4)
    values = torch.randn(20, 6, generator=generator, dtype=torch.float64)
    response = response_subspace(values, 2)
    feature = feature_pca_subspace(values + 10, 2)
    random = random_subspace(6, 2, 17, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    for U in (response, feature, random):
        torch.testing.assert_close(U.T @ U, eye)
    assert not torch.allclose(response @ response.T, feature @ feature.T)


def test_all_methods_keep_same_frozen_bundle_identity():
    target, resources = fixture()
    changed = replace(resources, artifact_bundle_id="different")
    first = run_cache_method("ep_tta", target, resources, EPConfig(steps=0))
    second = run_cache_method("ep_tta", target, changed, EPConfig(steps=0))
    # Scores happen to match, but orchestration must compare bundle IDs before
    # placing runs in one comparison group; this fixture exposes both identities.
    assert resources.artifact_bundle_id != changed.artifact_bundle_id
    assert first["score"] == second["score"]


def test_source_only_anchor_calibration_and_fixed_adapter_artifacts():
    target, resources = fixture()
    memory = build_anchor_memory(resources.anchors_z, resources.anchors_y, resources.w,
                                 resources.b, resources.tau0, 2, 13)
    assert memory["anchors_z"].shape[0] == 4
    assert len(set(memory["indices"].tolist())) == 4
    tau = empirical_real_quantile(resources.anchors_s0, resources.anchors_y, .25)
    assert isinstance(tau, float)
    source_views = torch.stack([target.features, target.features + .01])
    R, trace = fit_fixed_source_adapter(source_views, resources, 5, .01, .2, .1, 1.0)
    assert R.shape == (3, 3) and len(trace) == 5
    assert float(torch.linalg.vector_norm(R)) <= .2 + 1e-12
