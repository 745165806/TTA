import json
from pathlib import Path

import pytest

from eptta.data.io import sha256_file
from eptta.data.source_manifests import (publish_label_free_manifest, seal_source_manifests,
                                         validate_source_snapshot)
from eptta.data.approval import approve
from eptta.errors import ContractError, DataError
from eptta.models.author import canonical_to_native, class_weights_native, inspect_author_repository
from eptta.training.artifacts import finalize_training, require_exportable
from eptta.training.selection import equal_error_rate, select_source_checkpoint
from eptta.training.recipe import resolve_training_recipe
from eptta.training.dispatch import compile_source_job


ROOT = Path(__file__).resolve().parents[2]


def review():
    return {"accepted": True, "reviewer": "fixture-reviewer", "approved_at": "2026-09-14T00:00:00Z",
            "report_ref": "fixture-review.md", "sample_evidence_ref": "fixture-samples.json"}


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def _snapshot(tmp_path):
    canonical = tmp_path / "canonical.jsonl"
    _write_jsonl(canonical, [{"sample_id": name} for name in ("fit-0", "fit-1", "val-0", "val-1")])
    hashes = {}
    for role, prefix in (("fit", "fit"), ("source_val", "val")):
        path = tmp_path / "manifests" / (role + ".jsonl")
        _write_jsonl(path, [{"schema_version": "0.1.0", "sample_id": prefix + "-" + str(label),
                             "root_key": "fixture", "audio_relpath": prefix + "-" + str(label) + ".wav",
                             "input_sha256": "a" * 64, "split_role": role,
                             "canonical_label": label} for label in (0, 1)])
        hashes[path.name] = sha256_file(path)
    metadata = {"schema_version": "0.1.0", "status": "LOCKED", "snapshot_id": "snapshot-fixture",
                "canonical_ref": "canonical.jsonl", "canonical_sha256": sha256_file(canonical),
                "role_manifest_hashes": hashes, "fixture_only": False}
    (tmp_path / "snapshot.json").write_text(json.dumps(metadata))
    return tmp_path


def test_source_snapshot_seal_is_role_and_hash_bound(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot")
    result = seal_source_manifests(snapshot, tmp_path / "sealed")
    assert result["status"] == "LOCKED"
    assert set(result["roles"]) == {"fit", "source_val"}
    row_path = snapshot / "manifests/fit.jsonl"
    row_path.write_text(row_path.read_text() + row_path.read_text())
    with pytest.raises(DataError, match="changed"):
        validate_source_snapshot(snapshot)


def test_source_snapshot_can_publish_a_label_free_encoding_view(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot")
    result = publish_label_free_manifest(snapshot, "fit", tmp_path / "inference")
    assert result["labels_in_manifest"] is False
    text = (tmp_path / "inference/fit.jsonl").read_text()
    assert "canonical_label" not in text and '"split_role":"fit"' in text


def test_native_mapping_and_semantic_weights_are_reversed_from_canonical():
    mapping = {"spoof": 0, "bonafide": 1}
    assert canonical_to_native([0, 1], mapping) == [1, 0]
    assert class_weights_native({"bonafide": 0.9, "spoof": 0.1}, mapping) == [0.1, 0.9]


def test_source_val_eer_and_selection_contract():
    assert equal_error_rate([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]) == 0.0
    with pytest.raises(DataError, match="one class"):
        equal_error_rate([0.0, 1.0], [1, 1])
    chosen = select_source_checkpoint([
        {"epoch": 2, "source_val_eer": 0.1, "checkpoint_ref": "b", "checkpoint_sha256": "b" * 64},
        {"epoch": 1, "source_val_eer": 0.1, "checkpoint_ref": "a", "checkpoint_sha256": "a" * 64}])
    assert chosen["epoch"] == 1


def test_author_sources_are_pinned_when_local_repositories_exist():
    roots = {"aasist_source": Path("/media/dell/data/fakeAudioDection/aasist"),
             "ssl_aasist_source": Path("/media/dell/data/fakeAudioDection/SSL_Anti-spoofing")}
    for model_id, path in roots.items():
        if path.is_dir():
            result = inspect_author_repository(model_id, path)
            assert result["embedding_dim"] == 160
            assert result["class_index_map"] == {"spoof": 0, "bonafide": 1}


def test_smoke_and_external_runs_cannot_finalize_or_export(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"schema_version": "0.1.0", "status": "TRAINED",
                                               "phase": "smoke", "execution_channel": "production",
                                               "task_weight_origin": "trained_in_project"}))
    with pytest.raises(ContractError, match="smoke"):
        finalize_training(run, tmp_path / "finalized.json")
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"external")
    finalized = tmp_path / "external.json"
    finalized.write_text(json.dumps({"status": "FINALIZED", "training_phase": "full",
                                      "task_weight_origin": "external", "selected_checkpoint_ref": str(checkpoint),
                                      "selected_checkpoint_sha256": sha256_file(checkpoint)}))
    with pytest.raises(ContractError, match="external"):
        require_exportable(finalized)


def test_full_run_finalization_uses_earliest_minimum_eer(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/best.pt"
    checkpoint.write_bytes(b"in-project-checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    provenance = {"schema_version": "0.1.0", "epoch": 1, "global_step": 10,
                  "recipe_hash": "a" * 64, "fit_snapshot_hash": "b" * 64,
                  "source_val_snapshot_hash": "c" * 64, "task_weight_origin": "trained_in_project",
                  "training_seed": 13}
    (run / "checkpoints/best.pt.json").write_text(json.dumps(provenance))
    _write_jsonl(run / "metrics.jsonl", [{"epoch": 1, "source_val_eer": .1,
                                           "checkpoint_ref": "checkpoints/best.pt",
                                           "checkpoint_sha256": checkpoint_hash}])
    metadata = {"schema_version": "0.1.0", "status": "TRAINED", "phase": "full",
                "execution_channel": "production", "task_weight_origin": "trained_in_project",
                "training_run_id": "run", "model_id": "aasist_source", "recipe_ref": "recipe.json",
                "recipe_hash": "a" * 64, "fit_snapshot_hash": "b" * 64,
                "source_val_snapshot_hash": "c" * 64, "training_seed": 13,
                "metrics_ref": "metrics.jsonl", "metrics_sha256": sha256_file(run / "metrics.jsonl"),
                "architecture": {"repo_commit": "d" * 40},
                "class_index_map": {"spoof": 0, "bonafide": 1}, "initialization": None}
    (run / "run.json").write_text(json.dumps(metadata))
    result = finalize_training(run, tmp_path / "finalized.json")
    assert result["status"] == "FINALIZED" and result["selected_epoch"] == 1


def test_s08_weighted_microbatch_ddp_gradient_matches_global_batch():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
    weights = torch.tensor([.1, .9], dtype=torch.float64)
    model = torch.nn.Linear(5, 2, bias=False).double()
    initial = model.weight.detach().clone()
    full_loss = torch.nn.functional.cross_entropy(model(x), y, weight=weights)
    full_gradient, = torch.autograd.grad(full_loss, model.weight)
    rank_gradients = []
    for indices in (torch.arange(0, 12, 2), torch.arange(1, 12, 2)):
        local = torch.nn.functional.cross_entropy(model(x[indices]), y[indices], weight=weights,
                                                   reduction="none").sum()
        gradient, = torch.autograd.grad(local, model.weight)
        rank_gradients.append(gradient)
    ddp_average = torch.stack(rank_gradients).mean(0)
    corrected = ddp_average * 2 / weights[y].sum()
    torch.testing.assert_close(corrected, full_gradient)


def test_recipe_compilation_keeps_only_source_roots_and_roles(tmp_path):
    repository = Path("/media/dell/data/fakeAudioDection/aasist")
    if not repository.is_dir():
        pytest.skip("pinned author repository is not available in this checkout")
    snapshot = _snapshot(tmp_path / "snapshot")
    preprocess_proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval": None,
        "payload": {"decode": "soundfile_float32_mono_mean_require_16khz",
                    "train_unit": "repeat_or_crop_first_64600", "eval_unit": "repeat_or_crop_first_64600",
                    "source_probe": "identity", "target_probe": "probe_default_after_baseline_unit",
                    "quality_policy": "snapshot_prevalidated_fail_runtime"}}
    preprocess = approve(preprocess_proposal, review(), "preprocess")
    preprocess_path = tmp_path / "preprocess.lock.json"
    preprocess_path.write_text(json.dumps(preprocess))
    proposal = resolve_training_recipe("aasist_source", snapshot, preprocess_path,
        ROOT / "configs/training/aasist.yaml", repository,
        {"fixture": str(tmp_path), "target_test": "/forbidden/target"})
    locked = approve(proposal, review(), "recipe")
    recipe_path = tmp_path / "recipe.lock.json"
    recipe_path.write_text(json.dumps(locked))
    job = compile_source_job(recipe_path, "smoke", tmp_path / "run")
    assert job["execution"]["data_roots"] == {"fixture": str(tmp_path)}
    assert "target_test" not in json.dumps(job)
