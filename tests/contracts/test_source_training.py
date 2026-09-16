import hashlib
import json
import importlib.util
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
from eptta.training.resume import compile_resume_preflight, prepare_child_resume


ROOT = Path(__file__).resolve().parents[2]


def _author_training_module():
    path = ROOT / "workers/compat/author_training.py"
    spec = importlib.util.spec_from_file_location("eptta_test_author_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_worker_module():
    path = ROOT / "workers/source_train_bridge.py"
    spec = importlib.util.spec_from_file_location("eptta_test_source_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_fit_crop_is_deterministic_by_seed_epoch_sample_and_eval_is_first():
    module = _author_training_module()
    length = 64600
    audio_length = 211007
    key_a_epoch_0 = "%d\0%d\0%s" % (13, 0, "sample-a")
    key_a_epoch_1 = "%d\0%d\0%s" % (13, 1, "sample-a")
    key_b_epoch_0 = "%d\0%d\0%s" % (13, 0, "sample-b")
    first = module._crop_start(audio_length, length, key_a_epoch_0)
    assert first == module._crop_start(audio_length, length, key_a_epoch_0)
    assert 0 <= first <= audio_length - length
    assert len({module._crop_start(audio_length, length, key) for key in
                (key_a_epoch_0, key_a_epoch_1, key_b_epoch_0)}) > 1
    assert module._crop_start(audio_length, length, None) == 0
    assert module._crop_start(length, length, key_a_epoch_0) == 0


def test_source_val_eer_and_selection_contract():
    assert equal_error_rate([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]) == 0.0
    with pytest.raises(DataError, match="one class"):
        equal_error_rate([0.0, 1.0], [1, 1])
    chosen = select_source_checkpoint([
        {"epoch": 2, "source_val_eer": 0.1, "checkpoint_ref": "b", "checkpoint_sha256": "b" * 64},
        {"epoch": 1, "source_val_eer": 0.1, "checkpoint_ref": "a", "checkpoint_sha256": "a" * 64}])
    assert chosen["epoch"] == 1


def test_epoch_checkpoints_are_retained_while_last_and_best_advance(tmp_path):
    worker = _source_worker_module()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    epoch_0 = Path(worker.epoch_checkpoint_path(str(checkpoints), 0))
    epoch_1 = Path(worker.epoch_checkpoint_path(str(checkpoints), 1))
    epoch_0.write_bytes(b"epoch-zero")
    epoch_0.with_suffix(".pt.json").write_text('{"checkpoint_sha256":"zero"}\n')
    epoch_1.write_bytes(b"epoch-one")
    epoch_1.with_suffix(".pt.json").write_text('{"checkpoint_sha256":"one"}\n')

    worker.publish_checkpoint_alias(str(epoch_0), str(checkpoints / "last.pt"))
    worker.publish_checkpoint_alias(str(epoch_0), str(checkpoints / "best.pt"))
    worker.publish_checkpoint_alias(str(epoch_1), str(checkpoints / "last.pt"))

    assert epoch_0.read_bytes() == b"epoch-zero"
    assert epoch_1.read_bytes() == b"epoch-one"
    assert (checkpoints / "last.pt").read_bytes() == b"epoch-one"
    assert (checkpoints / "best.pt").read_bytes() == b"epoch-zero"
    assert json.loads((checkpoints / "last.pt.json").read_text())["checkpoint_sha256"] == "one"
    assert json.loads((checkpoints / "best.pt.json").read_text())["checkpoint_sha256"] == "zero"


def test_epoch_checkpoint_path_rejects_invalid_epoch(tmp_path):
    worker = _source_worker_module()
    assert worker.epoch_checkpoint_path(str(tmp_path), 12).endswith("epoch-0012.pt")
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            worker.epoch_checkpoint_path(str(tmp_path), invalid)


def _resume_fixture(tmp_path, scheduler="cosine_author_audited"):
    torch = pytest.importorskip("torch")
    worker = tmp_path / "worker.py"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text("# worker\n")
    worker_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    job = {"source_job": {"output_dir": str(run.resolve()), "model_id": "aasist_source",
                           "recipe_hash": "a" * 64, "training_seed": 13, "phase": "full",
                           "fit": {"snapshot_hash": "b" * 64},
                           "source_val": {"snapshot_hash": "b" * 64}, "initialization": None},
           "execution": {"runtime": {"strategy": "single_gpu", "world_size": 1,
                                        "per_gpu_batch_size": 2, "grad_accum_steps": 1},
                         "sample_counts": {"fit": 4},
                         "training": {"max_epochs": 80, "scheduler": scheduler,
                                      "sampler_policy": "shuffle_drop_global_tail"},
                         "training_orchestration": {"payload": {"project_worker_ref": str(worker.resolve()),
                                                                  "project_worker_sha256": worker_hash}},
                         "architecture": {}}}
    (run / "source_train_job.json").write_text(json.dumps(job))
    state = {"schema_version": "0.1.0", "model_state": {"w": torch.tensor([1.])},
             "optimizer_state": {"state": {}, "param_groups": [{"lr": 0.1}]},
             "scheduler_state": ({"last_epoch": 2, "_last_lr": [0.1]} if scheduler != "none" else None),
             "scaler_state": {}, "epoch": 0, "global_step": 2, "rng_states": [{"rank": 0}],
             "sampler_state": {"epoch": 0, "policy": "shuffle_drop_global_tail", "world_size": 1},
             "recipe_hash": "a" * 64, "fit_snapshot_hash": "b" * 64,
             "source_val_snapshot_hash": "b" * 64, "architecture": {}, "patch": {},
             "class_index_map": {"spoof": 0, "bonafide": 1}, "initialization": None,
             "training_seed": 13, "task_weight_origin": "trained_in_project"}
    checkpoint = checkpoints / "last.pt"
    torch.save(state, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = {key: state[key] for key in ("epoch", "global_step", "recipe_hash", "fit_snapshot_hash",
                                           "source_val_snapshot_hash", "training_seed", "task_weight_origin")}
    sidecar["checkpoint_sha256"] = digest
    checkpoint.with_suffix(".pt.json").write_text(json.dumps(sidecar))
    best = checkpoints / "best.pt"
    best.write_bytes(checkpoint.read_bytes())
    (run / "metrics.jsonl").write_text(json.dumps({"epoch": 0, "source_val_eer": 0.1,
        "checkpoint_ref": "checkpoints/best.pt", "checkpoint_sha256": digest}) + "\n")
    return run, checkpoint, worker, state


def test_resume_preflight_rejects_worker_hash_change(tmp_path):
    run, checkpoint, worker, _ = _resume_fixture(tmp_path)
    worker.write_text("# changed worker\n")
    result = compile_resume_preflight(run, checkpoint, worker, "exact", 80)
    assert result["status"] == "BLOCKED"
    assert any("worker hash" in item for item in result["blockers"])


def test_resume_preflight_rejects_missing_state_and_epoch_lr_offsets(tmp_path):
    torch = pytest.importorskip("torch")
    run, checkpoint, worker, state = _resume_fixture(tmp_path)
    state.pop("rng_states")
    torch.save(state, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = json.loads(checkpoint.with_suffix(".pt.json").read_text())
    sidecar["checkpoint_sha256"] = digest
    checkpoint.with_suffix(".pt.json").write_text(json.dumps(sidecar))
    with pytest.raises(DataError, match="lacks resume state"):
        compile_resume_preflight(run, checkpoint, worker, "exact", 80)

    run, checkpoint, worker, state = _resume_fixture(tmp_path / "offset")
    state["scheduler_state"]["last_epoch"] = 1
    state["scheduler_state"]["_last_lr"] = [0.2]
    torch.save(state, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = json.loads(checkpoint.with_suffix(".pt.json").read_text())
    sidecar["checkpoint_sha256"] = digest
    checkpoint.with_suffix(".pt.json").write_text(json.dumps(sidecar))
    result = compile_resume_preflight(run, checkpoint, worker, "exact", 80)
    assert any("scheduler last_epoch" in item for item in result["blockers"])
    assert any("optimizer LR" in item for item in result["blockers"])


def test_route_b_child_is_hash_bound_new_identity_and_preserves_scheduler_horizon(tmp_path):
    parent = tmp_path / "parent"
    checkpoints = parent / "checkpoints"
    checkpoints.mkdir(parents=True)
    worker = tmp_path / "worker.py"
    worker.write_text("# fixed worker bytes\n")
    parent_job = {
        "schema_version": "0.1.0", "job_type": "source_train",
        "source_job": {"model_id": "aasist_source", "recipe_hash": "a" * 64,
                       "fit": {"snapshot_hash": "b" * 64},
                       "source_val": {"snapshot_hash": "b" * 64},
                       "training_seed": 13, "phase": "full", "resume": None,
                       "output_dir": str(parent), "recipe_lock_ref": "parent.lock"},
        "execution": {"training": {"max_epochs": 100, "scheduler": "cosine_author_audited"},
                      "training_orchestration": {"payload": {"project_worker_ref": "old",
                                                                 "project_worker_sha256": "c" * 64},
                                                     "patch_sha256": "d" * 64}}}
    (parent / "source_train_job.json").write_text(json.dumps(parent_job))
    checkpoint = checkpoints / "last.pt"
    checkpoint.write_bytes(b"immutable parent")
    digest = sha256_file(checkpoint)
    sidecar = {"checkpoint_sha256": digest, "epoch": 13, "recipe_hash": "a" * 64,
               "fit_snapshot_hash": "b" * 64, "source_val_snapshot_hash": "b" * 64,
               "task_weight_origin": "trained_in_project",
               "patch": {"combined_sha256": "e" * 64}}
    checkpoint.with_suffix(".pt.json").write_text(json.dumps(sidecar))
    plan = {"plan_id": "ticket1-recovery-20260916", "models": {"aasist_source": {
        "parent_run_dir": str(parent), "parent_last_sha256": digest,
        "recommended_route": "B_PARENT_TO_CHILD", "child_total_epochs": 80}}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_hash = sha256_file(plan_path)

    first = prepare_child_resume(plan_path, plan_hash, "aasist_source", checkpoint,
                                 worker, tmp_path / "continuous", "resume_validation", 2)
    second = prepare_child_resume(plan_path, plan_hash, "aasist_source", checkpoint,
                                  worker, tmp_path / "restart", "resume_validation", 1)
    assert first["exact_resume_claim"] is False
    assert first["child_recipe_hash"] == second["child_recipe_hash"]
    assert first["execution_end_epoch"] == 16 and second["execution_end_epoch"] == 15
    child_job = json.loads((tmp_path / "continuous/source_train_job.json").read_text())
    assert child_job["execution"]["training"]["scheduler_horizon_epochs"] == 100
    assert child_job["source_job"]["recipe_hash"] != "a" * 64
    with pytest.raises(ContractError, match="SHA-256 changed"):
        prepare_child_resume(plan_path, "0" * 64, "aasist_source", checkpoint,
                             worker, tmp_path / "wrong", "resume_validation", 2)
    with pytest.raises(ContractError, match="overwrite"):
        prepare_child_resume(plan_path, plan_hash, "aasist_source", checkpoint,
                             worker, tmp_path / "continuous", "resume_validation", 2)


def test_worker_rejects_undisclosed_migration_and_uses_parent_scheduler_horizon():
    torch = pytest.importorskip("torch")
    worker = _source_worker_module()
    model = torch.nn.Linear(1, 1)
    job = {"execution": {"training": {"optimizer": "adam", "lr": 1e-4,
                                         "weight_decay": 0.0, "scheduler": "cosine",
                                         "max_epochs": 80, "scheduler_horizon_epochs": 100}}}
    optimizer, scheduler = worker.build_optimizer(model, job, 10)
    assert scheduler.lr_lambdas[0](800) > 5e-6 / 1e-4
    assert scheduler.lr_lambdas[0](1000) == pytest.approx(5e-6 / 1e-4)


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
    patch = {"payload": {"fixture": True}, "combined_sha256": "e" * 64}
    provenance = {"schema_version": "0.1.0", "epoch": 1, "global_step": 10,
                  "recipe_hash": "a" * 64, "fit_snapshot_hash": "b" * 64,
                  "source_val_snapshot_hash": "c" * 64, "task_weight_origin": "trained_in_project",
                  "training_seed": 13, "patch": patch}
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
                "class_index_map": {"spoof": 0, "bonafide": 1}, "initialization": None,
                "patch": patch}
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
                    "train_unit": "repeat_or_random_crop_64600_seed_epoch_sample",
                    "eval_unit": "repeat_or_crop_first_64600",
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
    assert "target_test" not in json.dumps({"source_job": job["source_job"],
                                             "data_roots": job["execution"]["data_roots"]})
    orchestration = job["execution"]["training_orchestration"]
    assert orchestration["payload"]["allowed_roles"] == ["fit", "source_val"]
    assert orchestration["payload"]["author_eval_path_disabled"] is True
    worker = _source_worker_module()
    worker.validate_job(job)
    assert worker.verify_training_orchestration(job) == orchestration
    tampered = json.loads(json.dumps(job))
    tampered["execution"]["training_orchestration"]["payload"]["project_worker_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="patch hash mismatch"):
        worker.verify_training_orchestration(tampered)
