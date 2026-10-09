"""Soft source decision-order preservation and exact base equivalence."""
import pytest

torch = pytest.importorskip("torch")

from experiments.epdc_development.retired_order import decision_order_loss, run_soft_preserve, source_order_pairs
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.adaptation.adapter import run_cache_method


def fixture():
    dtype = torch.float64
    z = torch.tensor([[-1., 0.], [-2., 0.], [1., 0.], [2., 0.]], dtype=dtype)
    y = torch.tensor([0, 0, 1, 1])
    scores = z[:, 0]
    resources = FrozenResources(torch.eye(2, dtype=dtype), torch.tensor([1., 0.], dtype=dtype),
                                0., z, y, scores.abs(), scores, 0., "source")
    target = TargetViews("opaque", torch.tensor([[1., 0.], [2., 0.], [3., 0.]], dtype=dtype), "cache")
    return target, resources


def test_source_pairs_and_soft_loss_protect_order_without_feature_equality():
    _, resources = fixture()
    bona, spoof, gap0 = source_order_pairs(resources)
    assert bona.tolist() == [0, 1] and spoof.tolist() == [2, 3]
    torch.testing.assert_close(gap0, torch.tensor([2., 4.], dtype=torch.float64))
    zero = torch.zeros((2, 2), dtype=torch.float64)
    assert float(decision_order_loss(zero, resources, bona, spoof, gap0)) == 0
    compressed = torch.tensor([[-.2, 0.], [0., 0.]], dtype=torch.float64)
    assert float(decision_order_loss(compressed, resources, bona, spoof, gap0)) > 0


def test_base_path_is_exact_and_episodic_replay_is_deterministic():
    target, resources = fixture()
    cfg = EPConfig(steps=5, lr=.03, rho=.1, gamma=.1, lambda_keep=0.)
    base = run_cache_method("ep_no_keep", target, resources, cfg)
    via_epdc = run_soft_preserve(target, resources, cfg, lambda_preserve=0.)
    assert via_epdc["score"] == base["score"]
    torch.testing.assert_close(via_epdc["R"], base["R"], atol=0, rtol=0)
    first = run_soft_preserve(target, resources, cfg, lambda_preserve=1.)
    replay = run_soft_preserve(target, resources, cfg, lambda_preserve=1.)
    assert first["score"] == replay["score"]
    torch.testing.assert_close(first["R"], replay["R"], atol=0, rtol=0)
    assert first["steps_completed"] == 5
    assert float(torch.linalg.vector_norm(first["R"])) <= .1 + 1e-12
    assert len(first["trace"]) == 5


def test_invalid_preservation_configuration_is_rejected():
    target, resources = fixture()
    with pytest.raises(ValueError, match="lambda_keep=0"):
        run_soft_preserve(target, resources, EPConfig(lambda_keep=1.))
    with pytest.raises(ValueError, match="lambda_preserve"):
        run_soft_preserve(target, resources, EPConfig(lambda_keep=0.), lambda_preserve=-1.)
