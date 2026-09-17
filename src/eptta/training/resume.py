"""Read-only source-training resume preflight.

The preflight deliberately does not launch a worker or mutate a training run.  It
distinguishes an exact continuation from an explicitly approved parent/child
migration and makes scheduler/epoch boundaries machine-checkable.
"""
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from eptta.config.schema import check
from eptta.data.io import iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError, ResourceError


REQUIRED_CHECKPOINT_KEYS = frozenset({
    "schema_version", "model_state", "optimizer_state", "scheduler_state",
    "scaler_state", "epoch", "global_step", "rng_states", "sampler_state",
    "recipe_hash", "fit_snapshot_hash", "source_val_snapshot_hash",
    "architecture", "patch", "class_index_map", "initialization",
    "training_seed", "task_weight_origin",
})


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_child_resume(plan_ref, expected_plan_sha256, model_id, checkpoint_ref,
                         worker_ref, output_dir, purpose, validation_epochs=None,
                         validation_report=None):
    """Publish, but never execute, an approved parent-to-child resume job.

    The child has a new recipe/run identity.  The immediate input checkpoint may
    be either the stopped parent or an earlier checkpoint from an isolated child
    validation branch; both are bound by bytes, recipe and patch identity.
    """
    if purpose not in ("resume_validation", "formal_continuation"):
        raise ContractError("child resume purpose must be resume_validation or formal_continuation")
    if purpose == "resume_validation":
        if type(validation_epochs) is not int or validation_epochs not in (1, 2):
            raise ContractError("resume validation must execute exactly one or two complete epochs")
    elif validation_epochs is not None:
        raise ContractError("formal continuation cannot set validation_epochs")
    if purpose == "formal_continuation":
        if not validation_report:
            raise ContractError("formal continuation requires a validation PASS report")
        validation = read_json(validation_report)
        passed = (validation.get("status") == "PASS" and
                  sorted(validation.get("models_passed", [])) == ["aasist_source", "ssl_aasist_source"])
        approved_nonexact = (validation.get("status") == "AUTHORIZED_NONEXACT" and
                             validation.get("model_id") == "aasist_source" and
                             validation.get("decision") == "continue_aasist_parent_epoch13_to_child_epoch79" and
                             validation.get("exact_resume_claim") is False and
                             validation.get("validation_status") == "FAIL" and
                             float(validation.get("formal_gpu_hour_cap", 0)) <= 7.3)
        if validation.get("plan_sha256") != expected_plan_sha256 or not (passed or approved_nonexact):
            raise ContractError("formal continuation requires bound PASS or explicit non-exact authorization")
    plan_path = Path(plan_ref).resolve()
    worker = Path(worker_ref).resolve()
    checkpoint = Path(checkpoint_ref).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ContractError("child output exists; overwrite is forbidden")
    for path, label in ((plan_path, "plan"), (worker, "worker"), (checkpoint, "checkpoint"),
                        (Path(str(checkpoint) + ".json"), "checkpoint sidecar")):
        if not path.is_file():
            raise ResourceError("child resume %s is missing: %s" % (label, path))
    if sha256_file(plan_path) != expected_plan_sha256:
        raise ContractError("approved recovery plan SHA-256 changed")
    plan = read_json(plan_path)
    if plan.get("plan_id") != "ticket1-recovery-20260916":
        raise ContractError("recovery plan id is not approved")
    model_plan = plan.get("models", {}).get(model_id)
    if not isinstance(model_plan, dict) or not str(model_plan.get("recommended_route", "")).startswith("B_PARENT_TO_CHILD"):
        raise ContractError("model is not approved for route B")
    if model_id == "ssl_aasist_source" and purpose == "formal_continuation":
        raise ContractError("approved SSL route B permits administrative finalization, not more training")

    parent_run = Path(model_plan["parent_run_dir"]).resolve()
    parent_job_path = parent_run / "source_train_job.json"
    if not parent_job_path.is_file():
        raise ResourceError("approved parent source job is missing")
    job = deepcopy(read_json(parent_job_path))
    source = job["source_job"]
    parent_identity = _identity(job)
    if source.get("model_id") != model_id:
        raise ContractError("approved parent model identity changed")
    sidecar = read_json(str(checkpoint) + ".json")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != sidecar.get("checkpoint_sha256"):
        raise DataError("resume checkpoint hash disagrees with sidecar")
    if checkpoint.parent.parent == parent_run:
        approved_last = model_plan.get("parent_last_sha256")
        if checkpoint_sha != approved_last:
            raise ContractError("initial route-B input is not the approved parent last checkpoint")
    elif purpose != "resume_validation":
        raise ContractError("formal continuation must start from the approved stopped parent")

    target_epochs = int(model_plan.get("child_total_epochs", 80))
    if model_id == "ssl_aasist_source":
        target_epochs = 80
    parent_horizon = int(job["execution"]["training"]["max_epochs"])
    if parent_horizon != 100:
        raise ContractError("approved route B requires the observed 100-epoch parent horizon")
    execution_end_epoch = target_epochs
    if purpose == "resume_validation":
        execution_end_epoch = int(sidecar["epoch"]) + 1 + validation_epochs
    job["execution"]["training"]["max_epochs"] = execution_end_epoch
    job["execution"]["training"]["scheduler_horizon_epochs"] = parent_horizon
    orchestration = job["execution"]["training_orchestration"]
    payload = orchestration["payload"]
    payload["project_worker_ref"] = str(worker)
    payload["project_worker_sha256"] = sha256_file(worker)
    orchestration["patch_sha256"] = _canonical_sha256(payload)
    identity = {
        "kind": "approved_parent_to_child", "plan_id": plan["plan_id"],
        "plan_sha256": expected_plan_sha256, "model_id": model_id,
        "parent_run_ref": str(parent_run), "parent_recipe_hash": source["recipe_hash"],
        "target_max_epochs": target_epochs, "scheduler_horizon_epochs": parent_horizon,
        "purpose": purpose,
    }
    child_recipe_hash = _canonical_sha256(identity)
    source["output_dir"] = str(output)
    source["recipe_lock_ref"] = str(plan_path)
    source["recipe_hash"] = child_recipe_hash
    source["phase"] = "full" if purpose == "formal_continuation" else "resume_validation"
    source["resume"] = {
        "artifact_ref": str(checkpoint), "training_run_id": parent_identity,
        "recipe_hash": sidecar["recipe_hash"],
        "fit_snapshot_hash": sidecar["fit_snapshot_hash"],
        "source_val_snapshot_hash": sidecar["source_val_snapshot_hash"],
        "task_weight_origin": sidecar["task_weight_origin"],
    }
    job["migration"] = {
        "kind": "approved_parent_to_child", "plan_id": plan["plan_id"],
        "plan_sha256": expected_plan_sha256, "parent_run_ref": str(parent_run),
        "resume_checkpoint_sha256": checkpoint_sha,
        "resume_recipe_hash": sidecar["recipe_hash"],
        "resume_patch_sha256": sidecar["patch"]["combined_sha256"],
        "exact_resume_claim": False, "purpose": purpose,
    }
    output.mkdir(parents=True)
    write_json_new(output / "source_train_job.json", job)
    if purpose == "formal_continuation":
        parent_metrics = list(iter_jsonl(parent_run / "metrics.jsonl"))
        if not parent_metrics:
            raise DataError("formal child cannot inherit an empty parent selection history")
        parent_best = min(parent_metrics, key=lambda item: (float(item["source_val_eer"]), int(item["epoch"])))
        parent_best_path = parent_run / parent_best["checkpoint_ref"]
        if not parent_best_path.is_file() or sha256_file(parent_best_path) != parent_best["checkpoint_sha256"]:
            raise DataError("formal child parent best checkpoint is missing or changed")
        seeded = dict(parent_best)
        seeded["checkpoint_ref"] = str(parent_best_path.resolve())
        seeded["selection_origin"] = "approved_parent_run"
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(seeded, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    manifest = {
        "schema_version": "0.1.0", "status": "PREPARED", "execution_started": False,
        "exact_resume_claim": False, "plan_id": plan["plan_id"],
        "plan_sha256": expected_plan_sha256, "model_id": model_id,
        "purpose": purpose, "parent_run_ref": str(parent_run),
        "resume_checkpoint_ref": str(checkpoint), "resume_checkpoint_sha256": checkpoint_sha,
        "child_run_dir": str(output), "child_recipe_hash": child_recipe_hash,
        "target_max_epochs": target_epochs, "scheduler_horizon_epochs": parent_horizon,
        "execution_end_epoch": execution_end_epoch, "validation_epochs": validation_epochs,
        "source_train_job_sha256": sha256_file(output / "source_train_job.json"),
    }
    if validation_report:
        manifest["authorization_ref"] = str(Path(validation_report).resolve())
        manifest["authorization_sha256"] = sha256_file(validation_report)
        manifest["authorization_status"] = read_json(validation_report).get("status")
    write_json_new(output / "migration_manifest.json", manifest)
    return manifest


def compare_resume_validation(model_id, continuous_checkpoint, restarted_checkpoint,
                              continuous_log, restarted_log, output):
    """Require exact state equality at the continuous/restarted epoch boundary."""
    import torch
    left_path = Path(continuous_checkpoint).resolve()
    right_path = Path(restarted_checkpoint).resolve()
    for path in (left_path, right_path, Path(continuous_log), Path(restarted_log)):
        if not path.is_file():
            raise ResourceError("resume comparison input is missing: %s" % path)
    left = _load_checkpoint(left_path)
    right = _load_checkpoint(right_path)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise DataError("resume comparison checkpoints must be mappings")
    for name, checkpoint in (("continuous", left), ("restarted", right)):
        missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
        if missing:
            raise DataError("%s checkpoint misses exact-state fields: %s" %
                            (name, ", ".join(sorted(missing))))

    def mismatch(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return None if (a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)) else path
        if type(a) is not type(b):
            return path + ".type"
        if isinstance(a, dict):
            if set(a) != set(b):
                return path + ".keys"
            for key in sorted(a):
                found = mismatch(a[key], b[key], path + "." + str(key))
                if found:
                    return found
            return None
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return path + ".length"
            for index, (one, two) in enumerate(zip(a, b)):
                found = mismatch(one, two, "%s[%d]" % (path, index))
                if found:
                    return found
            return None
        try:
            import numpy
            if isinstance(a, numpy.ndarray):
                return None if numpy.array_equal(a, b) else path
        except ImportError:
            pass
        return None if a == b else path

    fields = ("model_state", "optimizer_state", "scheduler_state", "scaler_state",
              "rng_states", "epoch", "global_step", "sampler_state", "recipe_hash",
              "fit_snapshot_hash", "source_val_snapshot_hash", "architecture", "patch",
              "class_index_map", "initialization", "training_seed", "task_weight_origin")
    differences = {}
    for field in fields:
        found = mismatch(left.get(field), right.get(field), field)
        if found:
            differences[field] = found
    left_rows = list(iter_jsonl(continuous_log))
    right_rows = list(iter_jsonl(restarted_log))
    if not left_rows or not right_rows:
        raise DataError("resume comparison logs must be nonempty")
    log_difference = mismatch(left_rows[-1], right_rows[-1], "last_train_log_record")
    if log_difference:
        differences["train_log"] = log_difference
    result = {
        "schema_version": "0.1.0", "status": "PASS" if not differences else "FAIL",
        "model_id": model_id, "comparison": "continuous_vs_process_restart_exact_epoch_boundary",
        "continuous_checkpoint_ref": str(left_path),
        "continuous_checkpoint_sha256": sha256_file(left_path),
        "restarted_checkpoint_ref": str(right_path),
        "restarted_checkpoint_sha256": sha256_file(right_path),
        "epoch": left.get("epoch"), "global_step": left.get("global_step"),
        "exact_fields": list(fields), "last_train_log_exact": not bool(log_difference),
        "sample_and_augmentation_identity": "exact_state_and_epoch_seeded_sampler_contract",
        "differences": differences,
    }
    write_json_new(output, result)
    return result


def _load_checkpoint(path):
    try:
        import torch
    except ImportError as exc:
        raise ResourceError("resume preflight requires CPU PyTorch to inspect state") from exc
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # PyTorch versions before mmap/weights_only keyword support.
        return torch.load(str(path), map_location="cpu")


def _expected_steps(job):
    runtime = job["execution"]["runtime"]
    if runtime["strategy"] != "single_gpu" or runtime.get("world_size") != 1:
        return None
    samples = int(job["execution"]["sample_counts"]["fit"])
    batches = int(math.ceil(float(samples) / int(runtime["per_gpu_batch_size"])))
    return int(math.ceil(float(batches) / int(runtime["grad_accum_steps"])))


def _identity(job):
    source = job["source_job"]
    material = (source["recipe_hash"] + source["fit"]["snapshot_hash"] +
                source["source_val"]["snapshot_hash"] + str(source["training_seed"]))
    return "source-run-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def compile_resume_preflight(run_ref, checkpoint_ref, worker_ref, route, target_max_epochs, output=None):
    """Inspect a stopped epoch-boundary checkpoint without starting CUDA/training."""
    if route not in ("exact", "child"):
        raise ContractError("resume route must be exact or child")
    if type(target_max_epochs) is not int or target_max_epochs < 1:
        raise ContractError("target_max_epochs must be a positive integer")
    run = Path(run_ref).resolve()
    checkpoint = Path(checkpoint_ref).resolve()
    worker = Path(worker_ref).resolve()
    for path, label in ((run / "source_train_job.json", "source job"),
                        (checkpoint, "checkpoint"), (Path(str(checkpoint) + ".json"), "checkpoint sidecar"),
                        (worker, "worker")):
        if not path.is_file():
            raise ResourceError(f"resume {label} is missing: {path}")
    if (run / "run.json").exists():
        raise ContractError("completed run.json exists; stopped-run preflight is not applicable")

    job = read_json(run / "source_train_job.json")
    source = job.get("source_job", {})
    execution = job.get("execution", {})
    if source.get("output_dir") != str(run):
        raise ContractError("run path disagrees with the immutable source job")
    sidecar = read_json(str(checkpoint) + ".json")
    digest = sha256_file(checkpoint)
    if digest != sidecar.get("checkpoint_sha256"):
        raise DataError("checkpoint hash disagrees with its sidecar")
    state = _load_checkpoint(checkpoint)
    if not isinstance(state, dict):
        raise DataError("checkpoint payload must be a mapping")
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(state))
    if missing:
        raise DataError("checkpoint lacks resume state: %s" % missing)

    blockers = []
    warnings = []
    required_equal = {
        "recipe_hash": source.get("recipe_hash"),
        "fit_snapshot_hash": source.get("fit", {}).get("snapshot_hash"),
        "source_val_snapshot_hash": source.get("source_val", {}).get("snapshot_hash"),
        "training_seed": source.get("training_seed"),
        "task_weight_origin": "trained_in_project",
    }
    for key, expected in required_equal.items():
        if state.get(key) != expected or sidecar.get(key) != expected:
            blockers.append(f"{key} does not match source job/checkpoint/sidecar")
    if state.get("epoch") != sidecar.get("epoch") or state.get("global_step") != sidecar.get("global_step"):
        blockers.append("epoch/global_step disagree between checkpoint and sidecar")

    runtime = execution.get("runtime", {})
    sampler = state.get("sampler_state")
    if not isinstance(sampler, dict):
        blockers.append("sampler_state is not a mapping")
    else:
        if sampler.get("epoch") != state.get("epoch"):
            blockers.append("sampler epoch does not equal completed checkpoint epoch")
        if sampler.get("world_size") != runtime.get("world_size"):
            blockers.append("sampler/DDP world size changed")
        if sampler.get("policy") != execution.get("training", {}).get("sampler_policy"):
            blockers.append("sampler policy changed")
    rng = state.get("rng_states")
    if not isinstance(rng, list) or len(rng) != runtime.get("world_size") or any(item is None for item in rng):
        blockers.append("per-rank RNG state is incomplete")

    steps_per_epoch = _expected_steps(job)
    completed_epochs = int(state["epoch"]) + 1
    if steps_per_epoch is not None:
        expected_global_step = completed_epochs * steps_per_epoch
        if state["global_step"] != expected_global_step:
            blockers.append("global_step is not on the declared epoch boundary")
    else:
        expected_global_step = None
        warnings.append("DDP/non-single-GPU global-step arithmetic requires a production-loader check")

    training = execution.get("training", {})
    scheduler_name = training.get("scheduler")
    scheduler_state = state.get("scheduler_state")
    if scheduler_name in (None, "none"):
        if scheduler_state is not None:
            blockers.append("checkpoint has scheduler state but recipe declares no scheduler")
    elif not isinstance(scheduler_state, dict):
        blockers.append("scheduled recipe lacks scheduler state")
    else:
        if scheduler_state.get("last_epoch") != state.get("global_step"):
            blockers.append("scheduler last_epoch/global_step offset detected")
        groups = state.get("optimizer_state", {}).get("param_groups", [])
        last_lrs = scheduler_state.get("_last_lr", [])
        if len(groups) != len(last_lrs) or any(float(g.get("lr")) != float(lr) for g, lr in zip(groups, last_lrs)):
            blockers.append("optimizer LR disagrees with scheduler state")

    metrics_path = run / "metrics.jsonl"
    best = None
    if not metrics_path.is_file():
        blockers.append("source-val selection history metrics.jsonl is missing")
    else:
        rows = list(iter_jsonl(metrics_path))
        if not rows:
            blockers.append("source-val selection history is empty")
        else:
            best = min(rows, key=lambda item: (float(item["source_val_eer"]), int(item["epoch"])))
            best_path = run / best["checkpoint_ref"]
            if not best_path.is_file() or sha256_file(best_path) != best["checkpoint_sha256"]:
                blockers.append("recorded best checkpoint is missing or changed")

    orchestration_payload = execution.get("training_orchestration", {}).get("payload", {})
    bound_worker_hash = orchestration_payload.get("project_worker_sha256")
    bound_worker_ref = orchestration_payload.get("project_worker_ref")
    actual_worker_hash = sha256_file(worker)
    locked_max_epochs = int(training.get("max_epochs"))
    start_epoch = completed_epochs
    if route == "exact":
        if actual_worker_hash != bound_worker_hash:
            blockers.append("exact resume worker hash differs from the locked run")
        if not bound_worker_ref or worker != Path(bound_worker_ref).resolve():
            blockers.append("exact resume worker import path differs from the locked run")
        if target_max_epochs != locked_max_epochs:
            blockers.append("exact resume cannot change the locked max_epochs")
    else:
        if actual_worker_hash == bound_worker_hash:
            warnings.append("child route was requested although worker bytes match the parent")
        if target_max_epochs <= start_epoch:
            blockers.append("parent checkpoint is already at or beyond the child training budget")
        if scheduler_name not in (None, "none") and target_max_epochs != locked_max_epochs:
            warnings.append("child must preserve the parent scheduler horizon to avoid an LR discontinuity")

    status = "PASS" if not blockers and route == "exact" else ("APPROVAL_REQUIRED" if not blockers else "BLOCKED")
    report = {
        "schema_version": "0.1.0", "status": status, "read_only": True,
        "route": route, "execution_ready": status == "PASS",
        "run_dir": str(run), "run_id_recorded": None,
        "deterministic_completion_run_id": _identity(job),
        "model_id": source.get("model_id"), "training_seed": source.get("training_seed"),
        "phase": source.get("phase"), "source_job_sha256": sha256_file(run / "source_train_job.json"),
        "checkpoint_ref": str(checkpoint), "checkpoint_sha256": digest,
        "checkpoint_epoch": state["epoch"], "resume_start_epoch": start_epoch,
        "global_step": state["global_step"], "steps_per_epoch": steps_per_epoch,
        "expected_global_step": expected_global_step,
        "locked_max_epochs": locked_max_epochs, "target_max_epochs": target_max_epochs,
        "recipe_hash": source.get("recipe_hash"),
        "snapshot_hashes": {"fit": source.get("fit", {}).get("snapshot_hash"),
                            "source_val": source.get("source_val", {}).get("snapshot_hash")},
        "architecture": execution.get("architecture"), "initialization": source.get("initialization"),
        "training_patch_sha256": state.get("patch", {}).get("combined_sha256"),
        "optimizer_state_present": isinstance(state.get("optimizer_state"), dict),
        "optimizer_lrs": [group.get("lr") for group in state.get("optimizer_state", {}).get("param_groups", [])],
        "scheduler": scheduler_name, "scheduler_state_present": scheduler_state is not None,
        "scheduler_position": ({"last_epoch": scheduler_state.get("last_epoch"),
                                "step_count": scheduler_state.get("_step_count"),
                                "last_lrs": scheduler_state.get("_last_lr")}
                               if isinstance(scheduler_state, dict) else None),
        "scaler_state_present": isinstance(state.get("scaler_state"), dict),
        "rng_rank_count": len(rng) if isinstance(rng, list) else None,
        "sampler_state": sampler, "best_selection_record": best,
        "bound_worker_ref": bound_worker_ref, "bound_worker_sha256": bound_worker_hash,
        "candidate_worker_ref": str(worker),
        "candidate_worker_sha256": actual_worker_hash,
        "stop_reason": "UNKNOWN_NO_STOP_RECORD" if not (run / "failure.json").is_file() else read_json(run / "failure.json"),
        "train_log_sha256": sha256_file(run / "train_log.jsonl") if (run / "train_log.jsonl").is_file() else None,
        "metrics_sha256": sha256_file(metrics_path) if metrics_path.is_file() else None,
        "blockers": blockers, "warnings": warnings,
    }
    check(report, "resume_preflight")
    if output:
        write_json_new(output, report)
    return report
