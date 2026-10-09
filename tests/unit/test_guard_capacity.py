"""Guard-only engineering checks; no target labels or target90 data."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.math import apply_adapter, margin_deficit
from eptta.adaptation.objectives import target_objective
from eptta.adaptation.regularizers import regularizer
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.baselines.dispatch import run_method
from experiments.multidomain_mechanism import guard_worker as worker


def simple_case():
    dtype = torch.float64
    anchors = torch.tensor([[-1., 0.], [1., 0.]], dtype=dtype)
    resources = FrozenResources(
        torch.eye(2, dtype=dtype), torch.tensor([1., 0.], dtype=dtype), 0.,
        anchors, torch.tensor([0, 1]), torch.ones(2, dtype=dtype),
        torch.tensor([-1., 1.], dtype=dtype), 0., "source")
    target = TargetViews("opaque", torch.tensor([[1., 0.], [2., 0.], [3., 0.]], dtype=dtype), "cache")
    return target, resources


def test_hard_is_production_and_guard_severity_is_only_change():
    target, resources = simple_case()
    cfg = EPConfig(steps=1, lr=1., rho=.5, gamma=.1, lambda_keep=0.)
    hard = run_cache_method("ep_tta_guarded", target, resources, cfg)
    production = run_method("ep_tta_guarded", target, resources, cfg, {})
    assert hard["score"] == production["score"]
    assert hard["trace"] == production["trace"]
    torch.testing.assert_close(hard["R"], production["R"], atol=0, rtol=0)
    relaxed = run_cache_method("ep_tta_guard_relaxed", target, resources, cfg)
    plain = run_cache_method("ep_tta", target, resources, cfg)
    assert all(result["score_before"] == 1. for result in (hard, relaxed, plain))
    assert float(hard["R"][0, 0]) > float(relaxed["R"][0, 0]) > float(plain["R"][0, 0])
    assert hard["trace"][0]["margin_guard_applied"]
    assert relaxed["trace"][0]["margin_guard_applied"]
    assert "margin_guard_applied" not in plain["trace"][0]
    # Identical differentiable objective at a fixed R; only the feasibility test changes.
    probe = torch.tensor([[-.15, 0.], [0., 0.]], dtype=torch.float64)
    adapt = target_objective("view_variance", apply_adapter(target.features, resources.U, probe),
                             resources.w, resources.b)
    keep = regularizer("margin", probe, resources, cfg.gamma)
    assert float(adapt + cfg.lambda_keep * keep) == pytest.approx(float(adapt))
    hard_deficit = margin_deficit(apply_adapter(resources.anchors_z, resources.U, probe),
                                  resources.w, resources.b, resources.anchors_y,
                                  resources.anchors_m0, resources.tau0, .1)
    relaxed_deficit = margin_deficit(apply_adapter(resources.anchors_z, resources.U, probe),
                                     resources.w, resources.b, resources.anchors_y,
                                     resources.anchors_m0, resources.tau0, .2)
    assert float(hard_deficit.max()) > 0
    assert float(relaxed_deficit.max()) == 0


def test_settings_frozen_and_per_sample_diagnostics():
    assert worker.SETTINGS == {"A": (5, .01, .05), "B": (5, .03, .1), "C": (10, .3, .2)}
    target, resources = simple_case()
    frozen = run_cache_method("frozen", target, resources, EPConfig(steps=0))
    assert frozen["score"] == frozen["score_before"]
    for name, (steps, lr, rho) in worker.SETTINGS.items():
        cfg = EPConfig(steps=steps, lr=lr, rho=rho, gamma=.1, lambda_keep=1.)
        result = run_cache_method("ep_tta_guard_relaxed", target, resources, cfg)
        assert result["steps_completed"] == steps
        assert float(torch.linalg.vector_norm(result["R"])) <= rho + 1e-12
        row = worker.diagnostic(target, resources, cfg, result, frozen["score"],
                                domain="synthetic", setting=name, arm="relaxed_guard", elapsed=0.)
        assert row["score_before"] == row["score_frozen"]
        assert row["numeric_status"] == "ok"
        assert row["evidence_damage"] >= 0


def test_select_loader_forbids_labels_and_requires_exact_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "SELECT", tmp_path)
    path = tmp_path / "in_the_wild_mechanism_select.json"
    rows = [{"sample_id": s, "dataset_id": "in_the_wild", "selection_seed": 2026,
             "root_ref": "/explicit", "audio_relpath": s} for s in ("a", "b")]
    doc = {"role": "mechanism_select", "dataset_id": "in_the_wild", "count": 2, "records": rows}
    path.write_text(json.dumps(doc))
    assert len(worker.load_select("in_the_wild")) == 2
    doc["records"][0]["label"] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="label/attack"):
        worker.load_select("in_the_wild")
    del doc["records"][0]["label"]
    doc["records"][1]["sample_id"] = "a"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sorted and unique"):
        worker.load_select("in_the_wild")
    doc["records"] = rows[:1]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="incomplete"):
        worker.load_select("in_the_wild")
