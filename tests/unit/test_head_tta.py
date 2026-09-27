"""Synthetic, label-free H-UA1 math and worker-boundary checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.head_tta.mixture import ALPHAS, adapt_buffer, make_source_geometry


def synthetic():
    rng = np.random.default_rng(2026)
    anchors = rng.normal(0, .5, (256, 160)).astype(np.float32)
    labels = np.array([0]*128+[1]*128)
    anchors[128:, :8] += .4
    resources = SimpleNamespace(anchors_z=torch.tensor(anchors),
                                anchors_y=torch.tensor(labels),
                                w=torch.tensor(rng.normal(0, .1, 160).astype(np.float32)),
                                b=.1)
    views = rng.normal(0, .5, (32, 3, 160)).astype(np.float32)
    views[:, 1:] = views[:, :1]+.05*views[:, 1:]
    return make_source_geometry(resources), views


def test_source_geometry_and_soft_assignment_are_finite_and_continuous():
    source, views = synthetic()
    result = adapt_buffer(views, source)
    assert set(result["scores"]) == set(ALPHAS)
    assert np.isfinite(result["frozen"]).all()
    assert np.all((result["q_spoof"] > 0) & (result["q_spoof"] < 1))
    assert np.all((result["reliability"] >= 0) & (result["reliability"] <= 1))
    assert np.isfinite(np.array(list(result["scores"].values()))).all()
    assert result["diagnostic"]["numeric_status"] == "ok"
    assert np.linalg.norm(result["scores"][.5]-result["frozen"]) > 0


def test_deterministic_and_no_cross_buffer_state():
    source, views = synthetic()
    first = adapt_buffer(views[:16], source)
    adapt_buffer(views[16:], source)
    again = adapt_buffer(views[:16], source)
    for alpha in ALPHAS:
        np.testing.assert_array_equal(first["scores"][alpha], again["scores"][alpha])
    np.testing.assert_array_equal(first["frozen"], again["frozen"])
    # Different buffer members are allowed to yield a different local head.
    other = adapt_buffer(views[16:], source)
    assert not np.array_equal(first["scores"][.5], other["scores"][.5])


def test_worker_has_no_target_label_import_or_label_fields():
    root = Path(__file__).resolve().parents[2]
    worker = (root / "experiments/head_tta/run_scores.py").read_text()
    math = (root / "experiments/head_tta/mixture.py").read_text()
    assert "supervised_labels" not in worker + math
    assert "load_labels" not in worker + math
    assert "target_labels_loaded\": False" in worker


def test_rejects_wrong_view_shape_and_nonfinite():
    source, views = synthetic()
    try:
        adapt_buffer(views[:, :1], source)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong view count accepted")
    views[0, 0, 0] = np.nan
    try:
        adapt_buffer(views, source)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("NaN accepted")
