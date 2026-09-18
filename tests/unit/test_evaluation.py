import json
import importlib.util
from pathlib import Path

import pytest

from eptta.evaluation.metrics import binary_metrics
from eptta.evaluation.checkpoint import compile_checkpoint_evaluation
from eptta.evaluation.seal import evaluate_sealed, seal_scores
from eptta.errors import ContractError, DataError
from eptta.data.io import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_worker_module():
    path = ROOT / "workers/checkpoint_eval_bridge.py"
    spec = importlib.util.spec_from_file_location("eptta_test_checkpoint_eval_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_t20_metrics_and_labels_only_after_seal(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"schema_version": "0.1.0", "sample_id": str(i), "score": score,
             "score_before": before, "status": "ok", "method_id": "ep_tta"}
            for i, (score, before) in enumerate([(-2, 2), (-1, -1), (1, -1), (2, 2)])]
    write_jsonl(run / "scores.jsonl", rows)
    (run / "expected_ids.json").write_text(json.dumps([str(i) for i in range(4)]))
    (run / "run.json").write_text(json.dumps({"expected_ids_ref": "expected_ids.json",
        "expected_ids_sha256": sha256_file(run / "expected_ids.json"), "expected_sample_count": 4,
        "scores_sha256": sha256_file(run / "scores.jsonl"), "feature_cache_key": "fixture",
        "artifact_bundle_id": "fixture", "fallback_rate_max": 0.0}))
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
    (run / "expected_ids.json").write_text(json.dumps(["x"]))
    (run / "run.json").write_text(json.dumps({"expected_ids_ref": "expected_ids.json",
        "expected_ids_sha256": sha256_file(run / "expected_ids.json"), "expected_sample_count": 1,
        "scores_sha256": sha256_file(run / "scores.jsonl"), "feature_cache_key": "fixture",
        "artifact_bundle_id": "fixture", "fallback_rate_max": 0.0}))
    with pytest.raises(DataError, match="forbidden"):
        seal_scores(run)


def test_checkpoint_evaluation_accepts_arbitrary_weight_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "anything.pt"
    checkpoint.write_bytes(b"arbitrary-parameter-file")
    protocol = tmp_path / "eval.txt"
    protocol.write_text("spk1 utt1 - - bonafide\nspk2 utt2 - A07 spoof\n")
    audio = tmp_path / "flac"
    audio.mkdir()
    architecture = {"model_id": "aasist_source", "repository_ref": "/repo",
                    "class_index_map": {"spoof": 0, "bonafide": 1}}
    monkeypatch.setattr("eptta.evaluation.checkpoint.inspect_author_repository",
                        lambda model_id, repository: architecture)
    cfg = {"paths": {"source_repos": {"aasist": "/repo", "ssl_aasist": None},
                     "generic_ssl_initialization": None}}
    job = compile_checkpoint_evaluation("aasist_source", checkpoint, protocol, audio,
                                        tmp_path / "result", cfg, evaluation_tag="diagnostic")
    assert job["checkpoint_ref"] == str(checkpoint.resolve())
    assert job["expected_sample_count"] == 2
    assert job["evaluation_tag"] == "diagnostic"
    assert "training_status" not in job and "finalized" not in json.dumps(job).lower()


def test_checkpoint_state_formats_and_ddp_prefix():
    torch = pytest.importorskip("torch")
    worker = _checkpoint_worker_module()
    tensor = torch.tensor([1.0])
    state, kind = worker.checkpoint_state({"model_state": {"module.weight": tensor}})
    assert list(state) == ["weight"] and kind == "eptta_model_state_module_prefix_removed"
    state, kind = worker.checkpoint_state({"state_dict": {"weight": tensor}})
    assert list(state) == ["weight"] and kind == "state_dict_field"
    state, kind = worker.checkpoint_state({"weight": tensor})
    assert list(state) == ["weight"] and kind == "raw_state_dict"


def test_legacy_worker_metrics_match_core_metrics():
    worker = _checkpoint_worker_module()
    scores = [-2.0, -1.0, 0.5, 2.0]
    labels = [0, 0, 1, 1]
    assert worker.binary_metrics(scores, labels, 0.0) == binary_metrics(scores, labels, 0.0)


def test_checkpoint_evaluation_protocol_rejects_duplicate_ids(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"x")
    protocol = tmp_path / "eval.txt"
    protocol.write_text("spk1 duplicate - - bonafide\nspk2 duplicate - A07 spoof\n")
    audio = tmp_path / "flac"
    audio.mkdir()
    monkeypatch.setattr("eptta.evaluation.checkpoint.inspect_author_repository",
                        lambda model_id, repository: {})
    cfg = {"paths": {"source_repos": {"aasist": "/repo"}}}
    with pytest.raises(DataError, match="duplicate"):
        compile_checkpoint_evaluation("aasist_source", checkpoint, protocol, audio,
                                      tmp_path / "result", cfg)
