"""Finite label-free task objectives and episodic replay."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from experiments.task_objective_discovery.objectives import (
    objective_terms, run_objective, soft_affinity, source_geometry,
)


CONFIG = json.loads((Path(__file__).parents[2] /
                     "experiments/task_objective_discovery/objective_config.json").read_text())


def fixture():
    dtype = torch.float64
    z = torch.tensor([[-2., -.3], [-1., .2], [1., -.2], [2., .4]], dtype=dtype)
    y = torch.tensor([0, 0, 1, 1])
    weight = torch.tensor([1., .2], dtype=dtype)
    scores = z @ weight
    resources = FrozenResources(torch.eye(2, dtype=dtype), weight,
                                0., z, y, scores.abs(), scores, 0., "source")
    target = TargetViews("opaque", torch.tensor([[.9, .1], [1.4, -.3], [.7, .4]],
                                                dtype=dtype), "cache")
    return target, resources


@pytest.mark.parametrize("arm", ("O1", "O2", "O3"))
def test_finite_nonzero_deterministic_and_k0_identity(arm):
    target, resources = fixture()
    cfg = EPConfig(steps=5, lr=.03, rho=.1, gamma=.1, lambda_keep=0.)
    source_before = resources.anchors_z.clone()
    first = run_objective(arm, target, resources, cfg, CONFIG)
    replay = run_objective(arm, target, resources, cfg, CONFIG)
    assert first["status"] == "ok"
    assert first["score_after"] == replay["score_after"]
    torch.testing.assert_close(first["R"], replay["R"], atol=0, rtol=0)
    assert first["trace"][0]["gradient_norm"] > 0
    assert float(torch.linalg.vector_norm(first["R"])) <= .1 + 1e-12
    torch.testing.assert_close(resources.anchors_z, source_before, atol=0, rtol=0)
    zero = run_objective(arm, target, resources,
                         EPConfig(steps=0, lr=.03, rho=.1, gamma=.1, lambda_keep=0.), CONFIG)
    assert zero["score_after"] == zero["score_before"]
    assert zero["objective_after"] == zero["objective_before"]


def test_soft_affinity_uses_source_labels_but_no_target_label():
    target, resources = fixture()
    geometry = source_geometry(resources)
    probabilities = soft_affinity(target.features, resources, geometry)
    assert probabilities.shape == (3, 2)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3, dtype=torch.float64))
    zero = torch.zeros((2, 2), dtype=torch.float64)
    terms = objective_terms(zero, target, resources, geometry, CONFIG)
    assert all(torch.isfinite(value).all() for value in terms)


def test_worker_has_no_audit_or_label_import():
    source = (Path(__file__).parents[2] /
              "experiments/task_objective_discovery/objective_worker.py").read_text()
    assert "load_select" in source
    assert "load_audit" not in source
    assert "canonical_label" not in source
