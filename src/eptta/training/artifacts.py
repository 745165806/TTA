"""Finalize full source runs and guard frozen export eligibility."""
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.training.selection import select_source_checkpoint
import json
import subprocess
import sys


def finalize_training(run_ref, output):
    run_dir = Path(run_ref)
    run = read_json(run_dir / "run.json")
    if run.get("schema_version") != "0.1.0" or run.get("status") != "TRAINED":
        raise ContractError("only a completed training run can be finalized")
    if run.get("phase") != "full" or run.get("execution_channel") != "production":
        raise ContractError("smoke/synthetic runs cannot be finalized")
    if run.get("task_weight_origin") != "trained_in_project":
        raise ContractError("external task weights cannot be finalized")
    metrics_path = run_dir / run.get("metrics_ref", "metrics.jsonl")
    if sha256_file(metrics_path) != run.get("metrics_sha256"):
        raise DataError("training metrics changed after completion")
    selected = select_source_checkpoint(list(iter_jsonl(metrics_path)))
    checkpoint = run_dir / selected["checkpoint_ref"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise DataError("selected checkpoint is missing or changed")
    sidecar = read_json(str(checkpoint) + ".json")
    required_equal = {
        "recipe_hash": run["recipe_hash"], "fit_snapshot_hash": run["fit_snapshot_hash"],
        "source_val_snapshot_hash": run["source_val_snapshot_hash"],
        "task_weight_origin": "trained_in_project", "training_seed": run["training_seed"]}
    if any(sidecar.get(key) != value for key, value in required_equal.items()):
        raise ContractError("selected checkpoint provenance disagrees with its run")
    if not run.get("patch") or sidecar.get("patch") != run["patch"]:
        raise ContractError("selected checkpoint patch provenance disagrees with its run")
    result = {"schema_version": "0.1.0", "status": "FINALIZED", "training_phase": "full",
              "training_run_id": run["training_run_id"], "model_id": run["model_id"],
              "selected_epoch": selected["epoch"], "source_val_eer": selected["source_val_eer"],
              "selection_rule": "minimum_source_val_eer_then_earliest_epoch",
              "selected_checkpoint_ref": str(checkpoint.resolve()),
              "selected_checkpoint_sha256": selected["checkpoint_sha256"],
              "recipe_ref": run["recipe_ref"], "recipe_hash": run["recipe_hash"],
              "fit_snapshot_hash": run["fit_snapshot_hash"],
              "source_val_snapshot_hash": run["source_val_snapshot_hash"],
              "architecture": run["architecture"], "class_index_map": run["class_index_map"],
              "patch": run["patch"],
              "embedding_dim": 160, "initialization": run.get("initialization"),
              "task_weight_origin": "trained_in_project", "metrics_sha256": run["metrics_sha256"],
              "finalized_id": "training-final-" + content_hash({"run": run["training_run_id"],
                                  "checkpoint": selected["checkpoint_sha256"]})[:20]}
    write_json_new(output, result)
    return result


def require_exportable(finalized_ref):
    value = read_json(finalized_ref)
    if value.get("status") != "FINALIZED" or value.get("training_phase") != "full":
        raise ContractError("frozen export requires FINALIZED full training")
    if value.get("task_weight_origin") != "trained_in_project":
        raise ContractError("frozen export rejects external/synthetic task weights")
    checkpoint = Path(value.get("selected_checkpoint_ref", ""))
    if not checkpoint.is_file() or sha256_file(checkpoint) != value.get("selected_checkpoint_sha256"):
        raise DataError("finalized checkpoint is missing or changed")
    return value


def launch_frozen_export(finalized_ref, output, worker_ref=None, python_executable=None):
    finalized = require_exportable(finalized_ref)
    checkpoint = Path(finalized["selected_checkpoint_ref"])
    run_dir = checkpoint.parent.parent
    source_job = read_json(run_dir / "source_train_job.json")
    execution = source_job["execution"]
    source_val = source_job["source_job"]["source_val"]
    first = next(iter_jsonl(source_val["manifest_ref"]), None)
    if first is None:
        raise DataError("source_val fixture is empty")
    try:
        fixture = Path(execution["data_roots"][first["root_key"]]) / first["audio_relpath"]
    except KeyError as exc:
        raise DataError("source fixture root is not bound") from exc
    job = {"schema_version": "0.1.0", "job_type": "frozen_export",
           "model_id": finalized["model_id"], "architecture": finalized["architecture"],
           "initialization": finalized.get("initialization"),
           "selected_checkpoint_ref": str(checkpoint),
           "selected_checkpoint_sha256": finalized["selected_checkpoint_sha256"],
           "training_patch": finalized["patch"],
           "fixture_audio_ref": str(fixture), "output_dir": str(Path(output).resolve()),
           "bundle_fields": {
               "training_run_id": finalized["training_run_id"],
               "fit_snapshot_hash": finalized["fit_snapshot_hash"],
               "source_val_snapshot_hash": finalized["source_val_snapshot_hash"],
               "recipe_hash": finalized["recipe_hash"], "class_index_map": finalized["class_index_map"],
               "embedding_dim": finalized["embedding_dim"], "init_provenance": finalized.get("initialization") or
                                  {"scope": "native_initialization"},
               "source_val_selection_ref": str(Path(finalized_ref).resolve()),
               "task_training_provenance": {"finalized_id": finalized["finalized_id"],
                                               "task_weight_origin": "trained_in_project",
                                               "source_val_selection_sha256": sha256_file(finalized_ref),
                                               "training_patch_sha256": finalized["patch"]["combined_sha256"]},
               "eval_preprocess_hash": source_job["execution"]["preprocess_hash"]}}
    destination = Path(output)
    if destination.exists():
        raise ContractError("frozen export output exists; overwrite is forbidden")
    job_path = run_dir / ("frozen_export_job_" + finalized["finalized_id"] + ".json")
    write_json_new(job_path, job)
    worker = Path(worker_ref) if worker_ref else Path(__file__).parents[3] / "workers/baseline_bridge.py"
    completed = subprocess.run([python_executable or sys.executable, str(worker), "export", "--job", str(job_path)],
                               check=False)
    if completed.returncode:
        raise ContractError("frozen export worker failed with exit code %d" % completed.returncode)
    bundle_path = destination / "bundle.json"
    bundle = read_json(bundle_path)
    from eptta.models.contracts import FrozenModelBundle
    FrozenModelBundle(**bundle)
    return bundle
