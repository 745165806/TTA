import json

import pytest

from eptta.evaluation.metrics import binary_metrics
from eptta.evaluation.seal import evaluate_sealed, seal_scores
from eptta.errors import ContractError, DataError


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_t20_metrics_and_labels_only_after_seal(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"schema_version": "0.1.0", "sample_id": str(i), "score": score,
             "score_before": before, "status": "ok", "method_id": "ep_tta"}
            for i, (score, before) in enumerate([(-2, 2), (-1, -1), (1, -1), (2, 2)])]
    write_jsonl(run / "scores.jsonl", rows)
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [{"schema_version": "0.1.0", "sample_id": str(i), "canonical_label": i // 2}
                         for i in range(4)])
    with pytest.raises(ContractError, match="sealed"):
        evaluate_sealed(run, labels, tmp_path / "early.json", 0.0)
    seal_scores(run)
    result = evaluate_sealed(run, labels, tmp_path / "metrics.json", 0.0)
    assert result["metrics"]["auroc"] == result["metrics"]["balanced_accuracy"] == 1.0
    assert result["metrics"]["eer"] == 0.0
    assert result["metrics"]["helpful_flips"] == 2
    assert result["metrics"]["helpful_flips_by_class"] == {"0": 1, "1": 1}
    assert result["metrics"]["harmful_flips_by_class"] == {"0": 0, "1": 0}
    assert result["labels_read"] is True and result["evaluator_role"] == "independent_post_seal"


def test_score_seal_rejects_target_annotations(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(run / "scores.jsonl", [{"sample_id": "x", "score": 0.0, "canonical_label": 1}])
    with pytest.raises(DataError, match="forbidden"):
        seal_scores(run)
