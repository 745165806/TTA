import json

import pytest

from experiments.epdc_development import v0a_analysis as analysis


def test_audit_reader_is_not_called_on_missing_score(tmp_path, monkeypatch):
    run = tmp_path
    for part in ("diagnostics", "manifests", "scores", "analysis"):
        (run / part).mkdir()
    (run / "run_config.json").write_text(json.dumps({"scientific_role": "mechanism_dev",
                                                        "datasets": {"in_the_wild": 1}}))
    (run / "diagnostics/score_completion.json").write_text(json.dumps({"status": "SCORES_COMPLETE_LABELS_NOT_READ"}))
    (run / "manifests/in_the_wild_mechanism_select.json").write_text(json.dumps({"records": [{"sample_id": "a"}]}))
    row = {"sample_id": "a", "domain": "in_the_wild", "arm": "Frozen",
           "numeric_status": "ok", "score_frozen": 1., "score_before": 1., "score_after": 1.}
    (run / "scores/in_the_wild.jsonl").write_text(json.dumps(row) + "\n")
    def forbidden(*args, **kwargs):
        pytest.fail("audit labels were opened before complete score coverage")
    monkeypatch.setattr(analysis, "selected_labels", forbidden)
    with pytest.raises(ValueError, match="score count"):
        analysis.analyze(run)
