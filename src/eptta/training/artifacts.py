"""Finalize full source runs and guard frozen export eligibility."""
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError, NotImplementedStage
from eptta.training.selection import select_source_checkpoint


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


def finalize_administrative_child(plan_ref, expected_plan_sha256, validation_report, output):
    """Finalize the approved SSL parent selection under a disclosed child identity."""
    plan_path = Path(plan_ref).resolve()
    if sha256_file(plan_path) != expected_plan_sha256:
        raise ContractError("approved recovery plan SHA-256 changed")
    validation = read_json(validation_report)
    if (validation.get("status") != "PASS" or
            validation.get("plan_sha256") != expected_plan_sha256 or
            sorted(validation.get("models_passed", [])) != ["aasist_source", "ssl_aasist_source"]):
        raise ContractError("administrative finalization requires both models to pass bound validation")
    plan = read_json(plan_path)
    if plan.get("plan_id") != "ticket1-recovery-20260916":
        raise ContractError("recovery plan id is not approved")
    model_plan = plan.get("models", {}).get("ssl_aasist_source", {})
    if model_plan.get("recommended_route") != "B_PARENT_TO_CHILD_ADMINISTRATIVE_FINALIZATION":
        raise ContractError("SSL administrative migration is not approved by this plan")
    if model_plan.get("additional_training_epochs") != 0:
        raise ContractError("administrative migration must not request training")
    run_dir = Path(model_plan["parent_run_dir"]).resolve()
    if (run_dir / "run.json").exists():
        raise ContractError("approved parent unexpectedly has a completion record")
    job = read_json(run_dir / "source_train_job.json")
    source = job["source_job"]
    if source.get("model_id") != "ssl_aasist_source" or source.get("phase") != "full":
        raise ContractError("approved SSL parent identity changed")
    metrics_path = run_dir / "metrics.jsonl"
    selected = select_source_checkpoint(list(iter_jsonl(metrics_path)))
    checkpoint = run_dir / selected["checkpoint_ref"]
    if sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise DataError("selected parent checkpoint is missing or changed")
    if selected["checkpoint_sha256"] != model_plan.get("parent_best_sha256"):
        raise ContractError("parent best selection differs from the approved plan")
    last = run_dir / "checkpoints/last.pt"
    if sha256_file(last) != model_plan.get("parent_last_sha256"):
        raise ContractError("parent last checkpoint differs from the approved plan")
    sidecar = read_json(str(checkpoint) + ".json")
    if selected["epoch"] != 5 or sidecar.get("epoch") != 5:
        raise ContractError("approved earliest SSL best epoch changed")
    for key, expected in (("recipe_hash", source["recipe_hash"]),
                          ("fit_snapshot_hash", source["fit"]["snapshot_hash"]),
                          ("source_val_snapshot_hash", source["source_val"]["snapshot_hash"]),
                          ("task_weight_origin", "trained_in_project")):
        if sidecar.get(key) != expected:
            raise ContractError("selected parent checkpoint provenance changed: " + key)
    child_id = "source-child-" + content_hash({"plan": expected_plan_sha256,
                                                "parent_best": selected["checkpoint_sha256"]})[:20]
    result = {
        "schema_version": "0.1.0", "status": "FINALIZED", "training_phase": "full",
        "training_run_id": child_id, "model_id": "ssl_aasist_source",
        "selected_epoch": selected["epoch"], "source_val_eer": selected["source_val_eer"],
        "selection_rule": "parent_minimum_source_val_eer_then_earliest_epoch",
        "selected_checkpoint_ref": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "recipe_ref": source["recipe_lock_ref"], "recipe_hash": source["recipe_hash"],
        "fit_snapshot_hash": source["fit"]["snapshot_hash"],
        "source_val_snapshot_hash": source["source_val"]["snapshot_hash"],
        "architecture": job["execution"]["architecture"],
        "class_index_map": job["execution"]["class_index_map"], "patch": sidecar["patch"],
        "embedding_dim": 160, "initialization": source.get("initialization"),
        "task_weight_origin": "trained_in_project", "metrics_sha256": sha256_file(metrics_path),
        "finalized_id": "training-final-" + content_hash({"run": child_id,
                                                            "checkpoint": selected["checkpoint_sha256"]})[:20],
        "migration": {"kind": "administrative_parent_to_child_no_training",
                      "plan_id": plan["plan_id"], "plan_sha256": expected_plan_sha256,
                      "parent_run_ref": str(run_dir), "parent_stopped_epoch": 90,
                      "parent_locked_max_epochs": 100, "child_default_epochs": 80,
                      "exact_resume_claim": False, "additional_training_epochs": 0,
                      "stop_reason": "UNKNOWN_NO_STOP_RECORD"},
    }
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
    """Reject the superseded single-fixture exporter before it writes anything.

    The production frozen-bundle contract now requires the independent R4
    validation evidence produced by ``export-frozen-r4``.  Keeping the legacy
    worker callable would create an old-format directory and only fail after
    publication, so this compatibility entrypoint is deliberately fail-closed.
    """
    raise NotImplementedStage(
        "legacy export-frozen cannot satisfy the R4 frozen bundle contract; "
        "use the reviewed export-frozen-r4 entrypoint"
    )
