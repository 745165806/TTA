import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from eptta.errors import ContractError
from eptta.models.contracts import FrozenModelBundle
from eptta.training.r4 import _eer_recompute_status


ROOT = Path(__file__).resolve().parents[2]


def valid_bundle():
    return FrozenModelBundle(
        "0.1.0", "aasist_source", "baseline", "a" * 64, "run", "b" * 64, "c" * 64,
        "d" * 64, {"scope": "native_initialization"},
        {"migration": {"exact_resume_claim": False},
         "training_endpoint": {"last_epoch": 79, "global_step": 42320,
                               "completed_epoch_count": 80, "scheduler_horizon_epochs": 100}},
        "e" * 64, {"bonafide": 1, "spoof": 0}, "linear_head.pt", 160, "FINALIZED",
        "full", "trained_in_project", "selection.json", "parity.json",
        {"embedding_point": "native_out_layer_input", "freq_aug": False},
        {"formula": "native_logits[spoof]-native_logits[bonafide]", "direction": "larger_is_spoof",
         "output_type": "logit_difference", "unit": "dimensionless"},
        {"atol": 1e-6, "rtol": 1e-5},
        {"status": "PASS", "fit_count": 128, "source_val_count": 5654}, "f" * 64)


@pytest.mark.parametrize("field,value,match", [
    ("task_training_provenance",
     {"migration": {"exact_resume_claim": True},
      "training_endpoint": {"last_epoch": 79, "completed_epoch_count": 80,
                            "scheduler_horizon_epochs": 100}}, "nonexact"),
    ("task_training_provenance",
     {"migration": {"exact_resume_claim": False},
      "training_endpoint": {"last_epoch": 99, "completed_epoch_count": 100,
                            "scheduler_horizon_epochs": 100}}, "epoch79"),
    ("score_contract", {"formula": "native_logits[bonafide]-native_logits[spoof]",
                        "direction": "larger_is_bonafide", "output_type": "logit_difference",
                        "unit": "dimensionless"}, "score"),
    ("model_contract", {"embedding_point": "wrong_layer", "freq_aug": False}, "embedding"),
    ("model_contract", {"embedding_point": "native_out_layer_input", "freq_aug": True}, "embedding"),
    ("numerical_contract", {"atol": 1e-3, "rtol": 1e-3}, "tolerance"),
    ("r4_validation", {"status": "PASS", "fit_count": 127, "source_val_count": 5654}, "incomplete"),
])
def test_r4_bundle_rejects_identity_wrapper_and_gate_counterexamples(field, value, match):
    with pytest.raises(ContractError, match=match):
        replace(valid_bundle(), **{field: value})


def _worker_module():
    path = ROOT / "workers/baseline_bridge.py"
    spec = importlib.util.spec_from_file_location("r4_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r4_worker_rejects_annotations_duplicate_and_missing_uids(tmp_path):
    worker = _worker_module()
    manifest = tmp_path / "fit.jsonl"
    base = {"schema_version": "0.1.0", "sample_id": "a", "root_key": "root",
            "audio_relpath": "a.wav", "input_sha256": None, "split_role": "fit"}
    manifest.write_text(json.dumps({**base, "canonical_label": 0}) + "\n")
    with pytest.raises(ValueError, match="annotated"):
        worker._read_unlabeled_manifest(manifest, worker.sha256_file(manifest), ["a"], "fit",
                                        {"root": str(tmp_path)})
    manifest.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        worker._read_unlabeled_manifest(manifest, worker.sha256_file(manifest), ["a"], "fit",
                                        {"root": str(tmp_path)})
    manifest.write_text(json.dumps(base) + "\n")
    with pytest.raises(ValueError, match="coverage"):
        worker._read_unlabeled_manifest(manifest, worker.sha256_file(manifest), ["missing"], "fit",
                                        {"root": str(tmp_path)})


def test_frozen_head_allows_embedding_gradient_without_parameter_gradients():
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(160, 1)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    embedding = torch.randn(3, 160, requires_grad=True)
    head(embedding).sum().backward()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
    assert all(parameter.grad is None for parameter in head.parameters())


def test_r4_eer_recompute_gate_rejects_historical_or_export_drift():
    assert _eer_recompute_status(0.1, 0.1, 0.1)[0] == "PASS"
    assert _eer_recompute_status(0.2, 0.1, 0.1)[0] == "FAIL"
    assert _eer_recompute_status(0.1, 0.1, 0.2)[0] == "FAIL"


def test_extraction_worker_applies_locked_tf32_mode():
    torch = pytest.importorskip("torch")
    worker = _worker_module()
    old_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    old_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        actual = worker._apply_numerical_mode(
            {"dtype": "float32", "tf32_matmul": False, "tf32_cudnn": True,
             "block_units": 256})
        assert actual == {"tf32_matmul": False, "tf32_cudnn": True}
        with pytest.raises(ValueError, match="TF32"):
            worker._apply_numerical_mode(
                {"dtype": "float32", "tf32_matmul": False, "block_units": 256})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
