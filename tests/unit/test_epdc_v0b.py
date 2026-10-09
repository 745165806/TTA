import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.adapter import run_cache_method
from experiments.epdc_development.retired_margin import normalized_margin_loss, run_soft_preserve
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews


def fixture():
    dtype = torch.float64
    anchors = torch.tensor([[-1., 0.], [1., 0.]], dtype=dtype)
    resources = FrozenResources(torch.eye(2, dtype=dtype), torch.tensor([1., 0.], dtype=dtype),
                                0., anchors, torch.tensor([0, 1]),
                                torch.ones(2, dtype=dtype), torch.tensor([-1., 1.], dtype=dtype),
                                0., "source")
    target = TargetViews("opaque", torch.tensor([[1., 0.], [2., 0.], [3., 0.]], dtype=dtype), "cache")
    return target, resources


def test_normalized_margin_is_active_when_source_threshold_evidence_is_damaged():
    _, resources = fixture()
    zero = torch.zeros((2, 2), dtype=torch.float64)
    damage = torch.tensor([[-.2, 0.], [0., 0.]], dtype=torch.float64)
    assert float(normalized_margin_loss(zero, resources)) == 0
    assert float(normalized_margin_loss(damage, resources)) == pytest.approx(.01)


def test_base_is_exact_and_preserve_is_deterministic():
    target, resources = fixture()
    cfg = EPConfig(steps=5, lr=.03, rho=.1, gamma=.1, lambda_keep=0.)
    base = run_cache_method("ep_no_keep", target, resources, cfg)
    via = run_soft_preserve(target, resources, cfg, lambda_preserve=0.)
    assert base["score"] == via["score"]
    torch.testing.assert_close(base["R"], via["R"], atol=0, rtol=0)
    first = run_soft_preserve(target, resources, cfg)
    second = run_soft_preserve(target, resources, cfg)
    assert first["score"] == second["score"]
    torch.testing.assert_close(first["R"], second["R"], atol=0, rtol=0)
    assert float(torch.linalg.vector_norm(first["R"])) <= .1 + 1e-12
    assert len(first["trace"]) == 5
