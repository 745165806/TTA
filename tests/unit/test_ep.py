"""Synthetic only. A skipped module is NOT_RUN, never numerical equivalence PASS."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch", reason="NOT_RUN: CPU PyTorch is not installed; no tensor checks executed")

from eptta.adaptation.batch import run_batch
from eptta.adaptation.episode import run_episode
from eptta.adaptation.math import apply_adapter, keep_loss, project_frobenius_, view_loss
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from core_reference import CoreConfig, run_core_episode
from batch_reference import run_batch_valid

torch.set_num_threads(1)


def fixture(B=7, d=11, r=3, dtype=torch.float64):
    g = torch.Generator().manual_seed(415)
    U = torch.linalg.qr(torch.randn(d, r, dtype=dtype, generator=g))[0]
    w = torch.randn(d, dtype=dtype, generator=g)
    Za = torch.randn(16, d, dtype=dtype, generator=g)
    ya = torch.arange(16) % 2
    b, tau = .13, -.17
    signs = 2 * ya.to(dtype) - 1
    desired = tau + signs * torch.linspace(.05, 1.6, 16, dtype=dtype)
    Za += ((desired - Za @ w - b) / w.square().sum())[:, None] * w
    s0 = Za @ w + b
    m0 = signs * (s0 - tau)
    Z = torch.randn(B, 3, d, dtype=dtype, generator=g)
    resources = FrozenResources(U, w, b, Za, ya, m0, s0, tau, "fixture-source-bundle")
    targets = [TargetViews(f"opaque-{i}", z, f"fixture-feature-{i}") for i, z in enumerate(Z)]
    return targets, resources


def ref_args(target, f):
    return (target.features, f.U, f.w, f.b, f.anchors_z, f.anchors_y, f.anchors_m0, f.tau0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("B", [1, 2, 7, 128])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_t01_t13_t14_production_serial_batch_and_document_reference(dtype, B, steps):
    targets, f = fixture(B=B, dtype=dtype)
    cfg = EPConfig(steps=steps, lr=.15, lambda_keep=10)
    rcfg = CoreConfig(steps=steps, lr=.15, lambda_keep=10)
    serial = [run_episode(t, f, cfg) for t in targets]
    batch = run_batch(targets, f, cfg)
    reference = [run_core_episode(*ref_args(t, f), rcfg) for t in targets]
    reference_batch = run_batch_valid(torch.stack([t.features for t in targets]), *ref_args(targets[0], f)[1:], rcfg)
    tol = 3e-5 if dtype == torch.float32 else 1e-9
    for i, (s, b, ref) in enumerate(zip(serial, batch.episodes, reference)):
        assert s.status == b.status == ref.status == "ok"
        for matrix in (b.R, ref.R, reference_batch[0][i]):
            torch.testing.assert_close(s.R, matrix, atol=tol, rtol=tol)
        for score in (b.score_after, ref.score_after, float(reference_batch[2][i])):
            assert s.score_after == pytest.approx(score, abs=tol, rel=tol)
        assert s.steps_completed == b.steps_completed == steps
        assert s.final_loss_view == pytest.approx(b.final_loss_view, abs=tol, rel=tol)
        assert s.final_loss_keep == pytest.approx(b.final_loss_keep, abs=tol, rel=tol)
        if steps == 0:
            assert s.score_before == s.score_after and torch.count_nonzero(s.R) == 0


def test_t02_t03_t04_gradient_and_geometry():
    targets, f = fixture(B=1)
    Z, U = targets[0].features, f.U
    R = torch.zeros(3, 3, dtype=Z.dtype, requires_grad=True)
    adapted = apply_adapter(Z, U, R)
    torch.testing.assert_close(adapted, Z, atol=0, rtol=0)
    centered = Z - Z.mean(0)
    torch.testing.assert_close(view_loss(Z), centered.square().sum() / Z.numel())
    assert not torch.allclose(view_loss(Z), centered.square().sum() / Z.shape[0])
    grad, = torch.autograd.grad(view_loss(adapted), R)
    Qc = Z @ U - (Z @ U).mean(0)
    torch.testing.assert_close(grad, 2 * Qc.T @ Qc / Z.numel())
    R = torch.tensor([[.01, .03, -.02], [.06, -.07, .04], [.02, .01, -.03]], dtype=Z.dtype)
    adapted = apply_adapter(Z, U, R)
    torch.testing.assert_close(adapted, (Z.T + U @ R @ U.T @ Z.T).T)
    complement = torch.eye(Z.shape[1], dtype=Z.dtype) - U @ U.T
    torch.testing.assert_close(adapted @ complement, Z @ complement)
    bound = torch.linalg.vector_norm(R) * torch.linalg.vector_norm(U.T @ f.w) * torch.linalg.vector_norm(Z @ U, dim=-1)
    assert bool((((adapted - Z) @ f.w).abs() <= bound + 1e-12).all())


def test_t05_t06_initial_margin_and_one_step_lambda_invariance():
    targets, f = fixture(B=1)
    R = torch.zeros(3, 3, dtype=f.U.dtype, requires_grad=True)
    loss = keep_loss(apply_adapter(f.anchors_z, f.U, R), f.w, f.b, f.anchors_y, f.anchors_m0, f.tau0, .1)
    grad, = torch.autograd.grad(loss, R)
    assert float(loss) == 0 and torch.count_nonzero(grad) == 0
    out0 = run_episode(targets[0], f, EPConfig(steps=1, lambda_keep=0))
    out1 = run_episode(targets[0], f, EPConfig(steps=1, lambda_keep=999))
    torch.testing.assert_close(out0.R, out1.R, atol=0, rtol=0)
    assert out0.score_after == out1.score_after


def test_t07_one_sided_margin_activation_and_final_diagnostics():
    dtype = torch.float64
    U = torch.eye(2, dtype=dtype)
    w = torch.tensor([1., 0.], dtype=dtype)
    Za = torch.tensor([[-1., 0.], [1., 0.]], dtype=dtype)
    ya = torch.tensor([0, 1])
    f = FrozenResources(U, w, 0., Za, ya, torch.ones(2, dtype=dtype), Za @ w, 0., "fixture-active")
    t = TargetViews("opaque", torch.tensor([[1., 0.], [2., 0.], [3., 0.]], dtype=dtype), "fixture")
    plain = run_episode(t, f, EPConfig(steps=3, lr=.1, gamma=0, lambda_keep=0))
    kept = run_episode(t, f, EPConfig(steps=3, lr=.1, gamma=0, lambda_keep=10))
    assert not torch.allclose(plain.R, kept.R)
    final = keep_loss(apply_adapter(Za, U, kept.R), w, 0., ya, f.anchors_m0, 0., 0.)
    assert kept.final_loss_keep == pytest.approx(float(final), abs=1e-12)
    assert kept.final_loss_keep != pytest.approx(kept.trace[-1]["loss_keep_before"], abs=1e-12)
    assert kept.final_violation_fraction > 0
    # Increasing a correct source margin has no penalty (unlike two-sided MSE).
    assert float(keep_loss(Za * 2, w, 0., ya, f.anchors_m0, 0., 0.)) == 0


def test_t08_projection_independent_matrices():
    R = torch.stack([torch.zeros(3, 3), torch.eye(3) * .01, torch.eye(3) * 9]).double()
    previous = R.clone()
    project_frobenius_(R, .2)
    torch.testing.assert_close(R[:2], previous[:2], atol=0, rtol=0)
    assert bool((torch.linalg.vector_norm(R, dim=(-2, -1)) <= .2 + 1e-12).all())
    assert float(torch.linalg.vector_norm(R[2])) == pytest.approx(.2)


def test_t09_head_nullspace_score_invariant():
    targets, f = fixture(B=1, d=3, r=1)
    U = torch.tensor([[1.], [0.], [0.]], dtype=torch.float64)
    w = torch.tensor([0., 1., 0.], dtype=torch.float64)
    Za = torch.tensor([[1., -1., 0.], [1., 1., 0.]], dtype=torch.float64)
    ya = torch.tensor([0, 1])
    f = FrozenResources(U, w, 0., Za, ya, torch.ones(2, dtype=U.dtype), Za @ w, 0., "fixture-null")
    out = run_episode(targets[0], f, EPConfig(steps=5))
    assert out.score_before == out.score_after


def test_t10_t13_reset_order_tail_shard_and_rotation():
    targets, f = fixture()
    cfg = EPConfig(steps=5, lr=.1)
    full = run_batch(targets, f, cfg).episodes
    order = [3, 1, 6, 2, 4, 0, 5]
    shuffled = run_batch([targets[i] for i in order], f, cfg).episodes
    for out, index in zip(shuffled, order):
        torch.testing.assert_close(out.R, full[index].R)
    chunks = tuple(o for start in range(0, 7, 3) for o in run_batch(targets[start:start + 3], f, cfg).episodes)
    for a, b in zip(full, chunks):
        torch.testing.assert_close(a.R, b.R)
    a = run_episode(targets[0], f, cfg)
    run_episode(targets[1], f, cfg)
    again = run_episode(targets[0], f, cfg)
    torch.testing.assert_close(a.R, again.R, atol=0, rtol=0)
    Q = torch.linalg.qr(torch.tensor([[1., 2., 3.], [-2., 1., 1.], [3., 0., -1.]], dtype=f.U.dtype))[0]
    rotated = run_batch(targets, replace(f, U=f.U @ Q), cfg).episodes
    for a, b in zip(full, rotated):
        assert a.score_after == pytest.approx(b.score_after, abs=1e-9)
        torch.testing.assert_close(b.R, Q.T @ a.R @ Q, atol=1e-9, rtol=1e-9)


def test_t14_detect_mean_loss_and_shared_R_errors():
    targets, f = fixture(B=2)
    out = run_batch(targets, f, EPConfig(steps=1, lr=.01)).episodes
    Z = torch.stack([t.features for t in targets])
    R = torch.zeros(2, 3, 3, dtype=Z.dtype, requires_grad=True)
    losses = view_loss(apply_adapter(Z, f.U, R))
    wrong_grad, = torch.autograd.grad(losses.mean(), R)
    assert not torch.allclose(out[0].R, -.01 * wrong_grad[0])
    assert not torch.allclose(out[0].R, out[1].R)  # catches one shared episode matrix


def test_c03_c07_context_immutable_and_target_sidecar_invisible():
    targets, f = fixture(B=2)
    arrays = (f.U, f.w, f.anchors_z, f.anchors_y, f.anchors_m0, f.anchors_s0, targets[0].features)
    copies = [a.clone() for a in arrays]
    sidecar = {targets[0].sample_id: {"label": 0, "attack": "fixture_a"}}
    first = run_episode(targets[0], f)
    sidecar[targets[0].sample_id] = {"label": 1, "attack": "fixture_b"}
    again = run_episode(targets[0], f)
    assert first.score_after == again.score_after
    for original, snapshot in zip(arrays, copies):
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)
        assert original.grad is None
    assert first.R.numel() == f.U.shape[1] ** 2


def test_c10_numeric_failure_isolation_and_frozen_fallback():
    targets, f = fixture(B=3, dtype=torch.float32)
    bad = replace(targets[1], features=targets[1].features * 1e20)
    out = run_batch([targets[0], bad, targets[2]], f)
    assert out.serial_replay_count == 3
    assert out.serial_replay_seconds >= 0
    assert [r.status for r in out.episodes] == ["ok", "fallback_numeric", "ok"]
    failed = out.episodes[1]
    assert failed.score_before == failed.score_after
    assert torch.count_nonzero(failed.R) == 0 and failed.reason
    for i in (0, 2):
        reference = run_episode(targets[i], f)
        assert out.episodes[i].score_after == reference.score_after


@pytest.mark.parametrize("bad", ["nan", "shape", "orthogonal", "margin", "s0", "grad", "half", "role", "channel"])
def test_invalid_contracts_never_numeric_fallback(bad):
    targets, f = fixture(B=1)
    t = targets[0]
    if bad == "nan":
        Z = t.features.clone(); Z[0, 0] = float("nan"); t = replace(t, features=Z)
    elif bad == "shape": f = replace(f, anchors_y=f.anchors_y[:, None])
    elif bad == "orthogonal": f = replace(f, U=f.U * 2)
    elif bad == "margin": f = replace(f, anchors_m0=f.anchors_m0 + 1)
    elif bad == "s0": f = replace(f, anchors_s0=f.anchors_s0 + 1)
    elif bad == "grad": t = replace(t, features=t.features.clone().requires_grad_())
    elif bad == "half": t = replace(t, features=t.features.half())
    elif bad == "role": f = replace(f, source_role="audit")
    elif bad == "channel": f = replace(f, channel="real")
    with pytest.raises(ValueError):
        run_episode(t, f)
    with pytest.raises(ValueError):
        run_batch([t], f)


def test_original_score_overflow_rejected_and_no_grad_supported():
    targets, f = fixture(B=1, dtype=torch.float32)
    Z = targets[0].features.clone()
    Z[0] = torch.sign(f.w) * torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="frozen score"):
        run_episode(replace(targets[0], features=Z), f)
    with torch.no_grad():
        assert run_episode(targets[0], f).status == "ok"
    with torch.inference_mode():
        with pytest.raises(ValueError, match="inference_mode"):
            run_episode(targets[0], f)
