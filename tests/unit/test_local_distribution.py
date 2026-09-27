"""Local shared-R semantics and label isolation."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from experiments.local_distribution_tta.local import ordered_buffers, run_local_buffer
from experiments.task_objective_discovery.objectives import run_objective, source_geometry


ROOT = Path(__file__).parents[2]
CONFIG = json.loads((ROOT / "experiments/task_objective_discovery/objective_config.json").read_text())


def setup():
    dtype = torch.float64
    z = torch.tensor([[-2., -.3], [-1., .2], [1., -.2], [2., .4]], dtype=dtype)
    y = torch.tensor([0, 0, 1, 1])
    w = torch.tensor([1., .2], dtype=dtype)
    s = z @ w
    resources = FrozenResources(torch.eye(2, dtype=dtype), w, 0., z, y,
                                s.abs(), s, 0., "source")
    samples = [TargetViews("id-%02d" % i,
                           torch.tensor([[.9 + i * .1, .1], [1.4 + i * .1, -.3],
                                         [.7 + i * .1, .4]], dtype=dtype), "cache")
               for i in range(3)]
    cfg = EPConfig(steps=5, lr=.03, rho=.1, gamma=.1, lambda_keep=0.)
    return samples, resources, cfg


@pytest.mark.parametrize("objective,method", [("Base", "ep_no_keep"), ("O1", "O1")])
def test_b1_matches_existing_episodic(objective, method):
    samples, resources, cfg = setup()
    geometry = source_geometry(resources)
    local = run_local_buffer(samples[:1], resources, cfg, objective, CONFIG, geometry)
    existing = (run_cache_method(method, samples[0], resources, cfg) if objective == "Base" else
                run_objective(method, samples[0], resources, cfg, CONFIG, geometry))
    expected_score = existing["score"] if objective == "Base" else existing["score_after"]
    assert local["samples"][0]["score_after"] == pytest.approx(expected_score, abs=1e-12)
    torch.testing.assert_close(local["R"], existing["R"], atol=1e-12, rtol=0)


@pytest.mark.parametrize("objective", ["Base", "O1"])
def test_determinism_boundary_reset_and_k0(objective):
    samples, resources, cfg = setup()
    before = [getattr(resources, name).clone() for name in ("U", "w", "anchors_z", "anchors_y")]
    one = run_local_buffer(samples[:2], resources, cfg, objective, CONFIG)
    two = run_local_buffer(samples[:2], resources, cfg, objective, CONFIG)
    next_buffer = run_local_buffer(samples[2:], resources, cfg, objective, CONFIG, buffer_index=1)
    torch.testing.assert_close(one["R"], two["R"], atol=0, rtol=0)
    assert one["samples"] == two["samples"]
    assert one["buffer"]["trace"][0]["parameter_norm_before"] == 0
    assert next_buffer["buffer"]["trace"][0]["parameter_norm_before"] == 0
    assert len(ordered_buffers(samples, 2)) == 2
    assert [s.sample_id for b in ordered_buffers(samples, 2) for s in b] == [s.sample_id for s in samples]
    zero_cfg = EPConfig(steps=0, lr=.03, rho=.1, gamma=.1, lambda_keep=0.)
    zero = run_local_buffer(samples, resources, zero_cfg, objective, CONFIG)
    assert zero["buffer"]["R_norm"] == 0
    assert all(row["score_after"] == row["score_frozen"] for row in zero["samples"])
    for name, value in zip(("U", "w", "anchors_z", "anchors_y"), before):
        torch.testing.assert_close(getattr(resources, name), value, atol=0, rtol=0)


def test_coverage_and_target_label_isolation():
    samples, resources, cfg = setup()
    with pytest.raises(ValueError, match="unique"):
        ordered_buffers([samples[0], samples[0]], 2)
    worker = (ROOT / "experiments/local_distribution_tta/run_study.py").read_text()
    assert "load_audit" not in worker
    assert "selected_labels" not in worker
    assert "label_source_ref" not in worker
    assert "target_labels_read\": False" in worker
