"""Score coverage precedes every audit-label access."""
import json

import pytest

from experiments.multidomain_mechanism import guard_analysis as analysis
from experiments.multidomain_mechanism.guard_worker import ARMS, SETTINGS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_run(tmp_path):
    write(tmp_path / "run_config.json", {"datasets": {"asv2021_la": 2}, "scientific_role": "mechanism_dev"})
    write(tmp_path / "analysis/summary.json", {"status": "SCORES_COMPLETE_LABELS_NOT_READ", "audit_labels_read": False})
    write(tmp_path / "manifests/asv2021_la_mechanism_select.json",
          {"records": [{"sample_id": "a"}, {"sample_id": "b"}]})
    rows = []
    for sid in ("a", "b"):
        for setting, arm in (("Frozen", "Frozen"), *[(s, a) for s in SETTINGS for a in ARMS]):
            rows.append({"sample_id": sid, "domain": "asv2021_la", "setting": setting,
                         "arm": arm, "score_frozen": 0., "score_before": 0.,
                         "score_after": 0., "numeric_status": "ok"})
    path = tmp_path / "scores/asv2021_la.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_exact_score_coverage_before_label_access(tmp_path, monkeypatch):
    path = synthetic_run(tmp_path)
    _, scores = analysis.complete_scores(tmp_path)
    assert set(scores["asv2021_la"]) == {"a", "b"}
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n")
    def forbidden(*args, **kwargs):
        pytest.fail("audit label reader called before complete score coverage")
    monkeypatch.setattr(analysis, "selected_labels", forbidden)
    with pytest.raises(ValueError, match="score row count"):
        analysis.analyze(tmp_path)


def test_selected_label_loader_never_opens_full_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "AUDIT_LABELS", tmp_path)
    (tmp_path / "asv2021_la.json").write_text(json.dumps({
        "role": "selected_only_post_score_audit", "dataset_id": "asv2021_la", "count": 2,
        "records": [{"sample_id": "a", "label": 0}, {"sample_id": "b", "label": 1}]}))
    audit = {"role": "mechanism_audit", "dataset_id": "asv2021_la",
             "records": [{"sample_id": "a"}, {"sample_id": "b"}],
             "label_source_ref": "/must/not/open/full-eval-labels.jsonl"}
    assert analysis.selected_labels("asv2021_la", audit, {"a", "b"}) == {"a": 0, "b": 1}
    with pytest.raises(ValueError, match="mismatch"):
        analysis.selected_labels("asv2021_la", audit, {"a", "holdout"})
