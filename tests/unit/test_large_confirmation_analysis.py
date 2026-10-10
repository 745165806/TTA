"""Metric parity and complete-score gate before opening development labels."""
import json

import numpy as np
import pytest

from eptta.evaluation.metrics import binary_metrics
from experiments.large_scale_confirmation import analyze as audit


def test_fast_bootstrap_metric_matches_production_with_ties():
    rng = np.random.default_rng(2026)
    for count in (30, 101, 512):
        labels = [0] * (count // 3) + [1] * (count - count // 3)
        scores = rng.normal(size=count).round(2).tolist()
        production = binary_metrics(scores, labels, 0.)
        eer, auc = audit.fast_eer_auc(scores, labels)
        assert eer == pytest.approx(production["eer"], abs=1e-12)
        assert auc == pytest.approx(production["auroc"], abs=1e-12)


def test_missing_score_marker_prevents_label_read(tmp_path, monkeypatch):
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "run_config.json").write_text(json.dumps({"role": "large_dev_scores_no_labels",
        "datasets": {"in_the_wild": 3178, "codecfake": 5000, "asv2019_la_dev": 5000},
        "orders": audit.ORDERS, "arms": audit.ARMS, "target_labels_read": False}))
    (tmp_path / "diagnostics/score_completion.json").write_text(json.dumps({
        "status": "INCOMPLETE", "audit_labels_read": False}))
    monkeypatch.setattr(audit, "selected_labels", lambda _assignments:
                        (_ for _ in ()).throw(AssertionError("label reader reached")))
    with pytest.raises(ValueError, match="before label audit"):
        audit.analyze(tmp_path)
