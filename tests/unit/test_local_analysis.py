"""Post-score gate must reject incomplete experiments before opening labels."""
import json

import pytest

from experiments.local_distribution_tta import analyze as audit


def test_incomplete_scores_prevent_label_audit(tmp_path, monkeypatch):
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "run_config.json").write_text(json.dumps({"role": "two_domain_development_scores_no_labels",
                                                           "target_labels_read": False,
                                                           "datasets": {"in_the_wild": 512, "codecfake": 512},
                                                           "arms": audit.ARMS}))
    (tmp_path / "diagnostics/score_completion.json").write_text(json.dumps({
        "status": "INCOMPLETE", "audit_labels_read": False}))
    def forbidden(_ids):
        raise AssertionError("label loader reached before complete scores")
    monkeypatch.setattr(audit, "selected_itw_labels", forbidden)
    monkeypatch.setattr(audit, "selected_codecfake_labels", forbidden)
    with pytest.raises(ValueError, match="complete fixed two-domain"):
        audit.analyze(tmp_path)
