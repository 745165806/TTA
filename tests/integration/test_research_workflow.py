import csv
import json

import pytest

from eptta.errors import ContractError, DataError
from eptta.research import evaluate, report, run_tta


def _run(tmp_path):
    run = tmp_path / "runs" / "control"
    run.mkdir(parents=True)
    config = {"schema_version": "0.1.0", "command": "run-tta", "role": "control_test",
              "threshold": 0.0, "threshold_source": "source/cal0.json",
              "selection_source": "source/select.json", "method": {"method_id": "ep_tta"}}
    (run / "config.yaml").write_text(json.dumps(config))
    (run / "expected_ids.json").write_text(json.dumps(["real", "fake"]))
    with (run / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "score", "score_before", "status"])
        writer.writeheader()
        writer.writerows([{"sample_id": "real", "score": -1.0, "score_before": -0.5, "status": "ok"},
                          {"sample_id": "fake", "score": 1.0, "score_before": 0.5, "status": "ok"}])
    (run / "meta.json").write_text(json.dumps({"run_id": "fixture"}))
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,canonical_label\nreal,0\nfake,1\n")
    return run, labels


def test_evaluate_without_seal_then_report(tmp_path):
    run, labels = _run(tmp_path)
    result = evaluate(run, labels)
    assert result["metrics"]["eer"] == 0.0
    summary = tmp_path / "summary.csv"
    assert report(tmp_path / "runs", summary)["run_count"] == 1
    assert "ep_tta" in summary.read_text()


def test_evaluate_rejects_equal_count_wrong_ids(tmp_path):
    run, labels = _run(tmp_path)
    labels.write_text("sample_id,canonical_label\nreal,0\nother,1\n")
    with pytest.raises(DataError, match="ID sets differ"):
        evaluate(run, labels)


def test_model_update_method_is_not_forced_through_feature_cache(tmp_path):
    config = {"schema_version": "0.1.0", "command": "run-tta", "run_name": "tent",
              "output_root": str(tmp_path), "role": "select", "scope_id": "source/select",
              "input_manifest": str(tmp_path / "missing.jsonl"),
              "feature_cache": str(tmp_path / "missing-cache"),
              "resources": str(tmp_path / "missing-resources"),
              "frozen_bundle": str(tmp_path / "missing-bundle.json"),
              "method": {"method_id": "tent_audio_ep", "config": {
                  "steps": 1, "lr": 0.01, "rho": 0.2, "gamma": 0.1, "lambda_keep": 1.0},
                  "params": {}},
              "threshold": 0.0, "threshold_source": "source/cal0", "seed": 13}
    with pytest.raises(ContractError, match="NOT_IMPLEMENTED.*waveform/model-update"):
        run_tta(config)
