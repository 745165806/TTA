"""AASIST R4 candidate export, independent real-data parity, and gated publication."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.contracts import FrozenModelBundle
from eptta.training.artifacts import require_exportable
from eptta.training.selection import equal_error_rate, select_source_checkpoint


ATOL = 1e-6
RTOL = 1e-5
EER_ATOL = 1e-12
GPU_HOUR_CAP = 0.5


def _jsonl_new(path, rows):
    destination = Path(path)
    if destination.exists():
        raise ContractError(f"output exists; overwrite is forbidden: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _hash_order(value, seed=13):
    return hashlib.sha256(f"r4-fit128-v1\0{seed}\0{value}".encode()).hexdigest()


def _eer_recompute_status(historical, reference, exported):
    differences = {"reference_vs_export_eer_abs": abs(reference - exported),
                   "historical_vs_reference_abs": abs(historical - reference)}
    return ("PASS" if all(value <= EER_ATOL for value in differences.values()) else "FAIL",
            differences)


def _alternating_extremes(rows):
    rows = sorted(rows, key=lambda row: (row["audio_bytes"], _hash_order(row["sample_id"])))
    result = []
    lo, hi = 0, len(rows) - 1
    while lo <= hi:
        result.append(rows[lo])
        lo += 1
        if lo <= hi:
            result.append(rows[hi])
            hi -= 1
    return result


def _round_robin_select(groups, count):
    queues = {key: _alternating_extremes(value) for key, value in groups.items()}
    keys = sorted(queues)
    result, seen, index = [], set(), 0
    while len(result) < count and any(queues.values()):
        key = keys[index % len(keys)]
        index += 1
        if not queues[key]:
            continue
        chosen_index = next((i for i, row in enumerate(queues[key])
                             if row.get("speaker_id") not in seen), 0)
        row = queues[key].pop(chosen_index)
        result.append(row)
        seen.add(row.get("speaker_id"))
    if len(result) != count:
        raise DataError(f"cannot select {count} rows from fixed R4 strata")
    return result


def _prepare_manifests(source_job, run_root):
    fit_manifest = Path(source_job["source_job"]["fit"]["manifest_ref"])
    val_manifest = Path(source_job["source_job"]["source_val"]["manifest_ref"])
    expected_hashes = source_job["execution"]["manifest_hashes"]
    if sha256_file(fit_manifest) != expected_hashes["fit"] or sha256_file(val_manifest) != expected_hashes["source_val"]:
        raise DataError("locked source manifest changed")
    fit_rows = list(iter_jsonl(fit_manifest))
    val_rows = list(iter_jsonl(val_manifest))
    fit_ids = {row["sample_id"] for row in fit_rows}
    snapshot_root = fit_manifest.parent.parent
    snapshot = read_json(snapshot_root / "snapshot.json")
    role_counts = snapshot.get("role_counts", {})
    if (snapshot.get("status") != "LOCKED" or role_counts.get("fit") != len(fit_rows) or
            role_counts.get("source_val") != len(val_rows)):
        raise ContractError("source manifests do not match the locked snapshot role counts")
    metadata = {}
    for row in iter_jsonl(snapshot_root / snapshot["canonical_ref"]):
        if row["sample_id"] in fit_ids:
            metadata[row["sample_id"]] = row
    if len(metadata) != len(fit_rows):
        raise DataError("fit canonical metadata coverage mismatch")
    roots = source_job["execution"]["data_roots"]
    strata = {0: defaultdict(list), 1: defaultdict(list)}
    for row in fit_rows:
        item = metadata[row["sample_id"]]
        path = Path(roots[row["root_key"]]) / row["audio_relpath"]
        if not path.is_file():
            raise DataError(f"R4 source audio is missing: {row['sample_id']}")
        enriched = {**row, "speaker_id": item.get("speaker_id"),
                    "attack_id": item.get("generator_id"), "audio_bytes": path.stat().st_size}
        label = int(row["canonical_label"])
        stratum = enriched["attack_id"] if label == 1 else enriched["speaker_id"]
        strata[label][stratum].append(enriched)
    policy = source_job["execution"].get("parity_policy", {})
    budget = policy.get("per_class_budget", 64)
    if type(budget) is not int or budget < 1:
        raise ContractError("parity per-class budget must be a positive integer")
    if any(sum(len(rows) for rows in strata[label].values()) < budget for label in (0, 1)):
        raise ContractError("source snapshot cannot meet the approved parity budget without repeats")
    selected = _round_robin_select(strata[0], budget) + _round_robin_select(strata[1], budget)
    selected.sort(key=lambda row: _hash_order(row["sample_id"]))

    def unlabeled(row):
        return {key: row[key] for key in ("schema_version", "sample_id", "root_key", "audio_relpath",
                                          "input_sha256", "split_role")}

    contracts = Path(run_root) / "contracts"
    fit_view = contracts / "fit128.unlabeled.jsonl"
    val_view = contracts / "source_val.unlabeled.jsonl"
    _jsonl_new(fit_view, [unlabeled(row) for row in selected])
    _jsonl_new(val_view, [unlabeled(row) for row in val_rows])
    expected_attacks = policy.get("expected_attack_ids")
    actual_attacks = sorted({row["attack_id"] for row in selected
                             if row["canonical_label"] == 1 and row["attack_id"] is not None})
    if expected_attacks is not None and actual_attacks != expected_attacks:
        raise ContractError("parity sample does not cover the approved attack/group set")
    coverage_claim = ("observed_generator_groups" if actual_attacks else
                      "metadata_unavailable_weaker_coverage")
    fit_evidence = {"schema_version": "0.2.0", "status": "LOCKED", "rule":
                    "balanced canonical classes; approved budget; deterministic seed13 hash order; round-robin audited groups; no scores or repeated IDs",
                    "count": 2 * budget, "class_counts": {"bonafide": budget, "spoof": budget},
                    "parity_policy": {"per_class_budget": budget,
                                      "expected_attack_ids": expected_attacks},
                    "coverage_claim": coverage_claim,
                    "attack_ids": actual_attacks,
                    "speaker_count": len({row["speaker_id"] for row in selected}),
                    "audio_bytes_min": min(row["audio_bytes"] for row in selected),
                    "audio_bytes_max": max(row["audio_bytes"] for row in selected),
                    "uids": [{"sample_id": row["sample_id"], "canonical_label": row["canonical_label"],
                              "speaker_id": row["speaker_id"], "attack_id": row["attack_id"],
                              "audio_bytes": row["audio_bytes"]} for row in selected],
                    "unlabeled_manifest_ref": str(fit_view.resolve()),
                    "unlabeled_manifest_sha256": sha256_file(fit_view)}
    write_json_new(contracts / "fit128_uids.json", fit_evidence)
    labels = {row["sample_id"]: int(row["canonical_label"]) for row in val_rows}
    write_json_new(contracts / "source_val_coordinator_labels.json",
                   {"schema_version": "0.1.0", "role": "source_val", "worker_access": False,
                    "labels": labels})
    return {"snapshot": snapshot, "roots": roots, "fit_view": fit_view, "val_view": val_view,
            "fit_ids": [row["sample_id"] for row in selected],
            "val_ids": [row["sample_id"] for row in val_rows], "val_labels": labels,
            "fit_evidence": fit_evidence}


def _identity(finalized_ref):
    finalized = require_exportable(finalized_ref)
    if finalized.get("model_id") != "aasist_source":
        raise ContractError("R4 currently supports only the independently verified AASIST source path")
    checkpoint = Path(finalized["selected_checkpoint_ref"])
    run_dir = checkpoint.parent.parent
    run = read_json(run_dir / "run.json")
    job = read_json(run_dir / "source_train_job.json")
    migration_path = run_dir / "migration_manifest.json"
    migration = read_json(migration_path) if migration_path.is_file() else None
    last = read_json(run_dir / "checkpoints/last.pt.json")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_hash = sha256_file(metrics_path)
    if metrics_hash != finalized.get("metrics_sha256") or metrics_hash != run.get("metrics_sha256"):
        raise ContractError("finalized selection metrics changed after training")
    selected_sidecar = read_json(str(checkpoint) + ".json")
    if (selected_sidecar.get("epoch") != finalized["selected_epoch"] or
            selected_sidecar.get("checkpoint_sha256") not in (None, finalized["selected_checkpoint_sha256"])):
        raise ContractError("selected checkpoint sidecar disagrees with finalized identity")
    metrics = list(iter_jsonl(metrics_path))
    chosen = select_source_checkpoint(metrics)
    if (chosen["epoch"] != finalized["selected_epoch"] or
            chosen["checkpoint_sha256"] != finalized["selected_checkpoint_sha256"] or
            run.get("training_run_id") != finalized.get("training_run_id")):
        raise ContractError("finalized selection disagrees with the immutable run evidence")
    training = job["execution"]["training"]
    max_epochs = training.get("max_epochs")
    endpoint_ok = (type(max_epochs) is int and max_epochs > 0 and last.get("epoch") + 1 == max_epochs)
    approved_stop = finalized.get("approved_stop")
    if not endpoint_ok and not isinstance(approved_stop, dict):
        raise ContractError("training completion lacks the locked recipe endpoint or an approved stop rule")
    if migration is not None:
        if run.get("migration", {}).get("exact_resume_claim") is not False:
            raise ContractError("historical nonexact migration disclosure changed")
        if migration.get("execution_end_epoch") != last.get("epoch") + 1:
            raise ContractError("migration endpoint disagrees with the completed training log")
    return finalized, run_dir, run, job, migration, last


def compile_r4_preview(finalized_ref, output, worker_ref=None, python_executable=None, gpu_id=0):
    finalized, run_dir, run, source_job, migration, last = _identity(finalized_ref)
    worker = Path(worker_ref) if worker_ref else Path(__file__).parents[3] / "workers/baseline_bridge.py"
    parity_budget = source_job["execution"].get("parity_policy", {}).get("per_class_budget", 64)
    source_val_count = sum(1 for _ in iter_jsonl(source_job["source_job"]["source_val"]["manifest_ref"]))
    return {"schema_version": "0.1.0", "status": "PREVIEW", "side_effects": False,
            "model_id": "aasist_source", "training_run_id": finalized["training_run_id"],
            "finalized_id": finalized["finalized_id"], "selected_epoch": finalized["selected_epoch"],
            "selected_checkpoint_sha256": finalized["selected_checkpoint_sha256"],
            "training_endpoint": {"last_epoch": last["epoch"], "global_step": last["global_step"],
                                  "completed_epoch_count": last["epoch"] + 1,
                                  "scheduler_horizon_epochs": source_job["execution"]["training"]["scheduler_horizon_epochs"]},
            "migration": run.get("migration"), "output": str(Path(output).resolve()),
            "worker_ref": str(worker.resolve()), "worker_sha256": sha256_file(worker),
            "python_executable": python_executable or sys.executable, "gpu_id": gpu_id,
            "gpu_hour_cap": GPU_HOUR_CAP, "atol": ATOL, "rtol": RTOL,
            "roles": {"fit": parity_budget * 2, "source_val": source_val_count},
            "forbidden": ["training", "target_scoring", "R5_resource_computation", "SSL_operations"]}


def _worker_job(kind, finalized, source_job, paths, worker, candidate, validation):
    return {"schema_version": "0.1.0", "job_type": kind, "model_id": "aasist_source",
            "architecture": finalized["architecture"], "initialization": finalized.get("initialization"),
            "selected_checkpoint_ref": finalized["selected_checkpoint_ref"],
            "selected_checkpoint_sha256": finalized["selected_checkpoint_sha256"],
            "training_patch": finalized["patch"], "candidate_dir": str(candidate.resolve()),
            "validation_dir": str(validation.resolve()), "data_roots": paths["roots"],
            "fit_manifest_ref": str(paths["fit_view"].resolve()),
            "fit_manifest_sha256": sha256_file(paths["fit_view"]),
            "source_val_manifest_ref": str(paths["val_view"].resolve()),
            "source_val_manifest_sha256": sha256_file(paths["val_view"]),
            "fit_ids": paths["fit_ids"], "source_val_ids": paths["val_ids"],
            "class_index_map": finalized["class_index_map"], "embedding_dim": 160,
            "batch_sizes": [1, 2, 7, 48], "atol": ATOL, "rtol": RTOL,
            "worker_sha256": sha256_file(worker), "gpu_hour_cap": GPU_HOUR_CAP}


def _run_worker(python_executable, worker, command, job_path, gpu_id):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completed = subprocess.run([python_executable, str(worker), command, "--job", str(job_path)],
                               env=environment, check=False)
    if completed.returncode:
        raise ContractError(f"R4 worker {command} failed with exit code {completed.returncode}")


def launch_r4_export(finalized_ref, output, worker_ref=None, python_executable=None, gpu_id=0):
    run_root = Path(output).resolve()
    if run_root.exists():
        raise ContractError("R4 output exists; overwrite is forbidden")
    finalized, run_dir, run, source_job, migration, last = _identity(finalized_ref)
    worker = Path(worker_ref) if worker_ref else Path(__file__).parents[3] / "workers/baseline_bridge.py"
    python_executable = python_executable or sys.executable
    run_root.mkdir(parents=True)
    paths = _prepare_manifests(source_job, run_root)
    candidate, validation = run_root / "candidate", run_root / "validation"
    candidate_job = _worker_job("r4_export_candidate", finalized, source_job, paths, worker, candidate, validation)
    verify_job = {**candidate_job, "job_type": "r4_verify"}
    candidate_job_path, verify_job_path = run_root / "r4_candidate_job.json", run_root / "r4_verify_job.json"
    write_json_new(candidate_job_path, candidate_job)
    write_json_new(verify_job_path, verify_job)
    write_json_new(run_root / "r4_plan.json", compile_r4_preview(finalized_ref, output, worker, python_executable, gpu_id))
    _run_worker(python_executable, worker, "r4-export-candidate", candidate_job_path, gpu_id)
    _run_worker(python_executable, worker, "r4-verify", verify_job_path, gpu_id)
    parity = read_json(validation / "parity.json")
    if parity.get("status") != "PASS":
        raise ContractError("R4 parity did not pass; candidate remains diagnostic only")
    source_rows = [row for row in iter_jsonl(validation / "per_sample.jsonl")
                   if row.get("phase") == "source_val_full"]
    source_val_count = len(paths["val_ids"])
    if len(source_rows) != source_val_count or len({row["sample_id"] for row in source_rows}) != source_val_count:
        raise DataError("source_val parity output coverage is incomplete")
    source_by_id = {row["sample_id"]: row for row in source_rows}
    ordered = [source_by_id[sample_id] for sample_id in paths["val_ids"]]
    labels = [paths["val_labels"][sample_id] for sample_id in paths["val_ids"]]
    reference_eer = equal_error_rate([row["reference_score"] for row in ordered], labels)
    export_eer = equal_error_rate([row["export_score"] for row in ordered], labels)
    historical = float(finalized["source_val_eer"])
    recompute_status, eer_differences = _eer_recompute_status(historical, reference_eer, export_eer)
    recompute = {"schema_version": "0.1.0", "status": recompute_status,
                 "kind": "R4_RECOMPUTE_NO_HISTORICAL_PER_SAMPLE_FILE",
                 "score_formula": "native_logits[spoof]-native_logits[bonafide]",
                 "score_direction": "larger_is_spoof", "eer_unit": "ratio_0_to_1",
                 "historical_raw_eer": historical, "historical_percent": historical * 100.0,
                 "reference_recomputed_eer": reference_eer, "export_recomputed_eer": export_eer,
                 **eer_differences, "eer_atol": EER_ATOL,
                 "source_val_count": source_val_count,
                 "source_val_ids_sha256": hashlib.sha256("\n".join(paths["val_ids"]).encode()).hexdigest(),
                 "full_coverage_no_drop_last": True, "checkpoint_reselection": False}
    write_json_new(run_root / "source_val_recompute.json", recompute)
    if recompute_status != "PASS":
        raise ContractError("R4 source_val EER recompute differs from historical/reference evidence")

    published = run_root / "published_bundle"
    with AtomicDirectory(published) as temporary:
        for name in ("detector_state.pt", "linear_head.pt", "baseline_bridge.py", "author_training.py"):
            shutil.copy2(candidate / name, temporary / name)
        shutil.copy2(validation / "parity.json", temporary / "parity.json")
        shutil.copy2(validation / "per_sample.jsonl", temporary / "parity_per_sample.jsonl")
        shutil.copy2(run_root / "source_val_recompute.json", temporary / "source_val_recompute.json")
        shutil.copy2(run_root / "contracts/fit128_uids.json", temporary / "fit128_uids.json")
        endpoint = {"last_epoch": last["epoch"], "global_step": last["global_step"],
                    "completed_epoch_count": last["epoch"] + 1,
                    "scheduler_horizon_epochs": source_job["execution"]["training"]["scheduler_horizon_epochs"]}
        provenance = {"finalized_id": finalized["finalized_id"], "task_weight_origin": "trained_in_project",
                      "source_val_selection_sha256": sha256_file(finalized_ref),
                      "training_patch_sha256": finalized["patch"]["combined_sha256"],
                      "training_code_sha256": finalized["patch"]["combined_sha256"],
                      "training_seed": run["training_seed"], "training_endpoint": endpoint,
                      "selection": {"rule": finalized["selection_rule"],
                                    "selected_epoch": finalized["selected_epoch"],
                                    "allowed_roles": ["source_val"],
                                    "candidate_epochs": sorted({row["epoch"] for row in iter_jsonl(
                                        run_dir / "metrics.jsonl")})}}
        if migration is not None:
            provenance["migration"] = run["migration"]
            provenance["parent_lineage"] = {
                "parent_run_ref": run["migration"]["parent_run_ref"],
                "resume_checkpoint_sha256": run["migration"]["resume_checkpoint_sha256"]}
        else:
            provenance["completion_evidence"] = "locked_recipe_endpoint_or_approved_stop"
        bundle = {"schema_version": "0.1.0", "model_id": "aasist_source",
                  "baseline_id": "baseline-" + content_hash({"checkpoint": finalized["selected_checkpoint_sha256"],
                                                               "head": sha256_file(temporary / "linear_head.pt"),
                                                               "r4": sha256_file(temporary / "parity.json")})[:20],
                  "selected_checkpoint_sha256": finalized["selected_checkpoint_sha256"],
                  "training_run_id": finalized["training_run_id"],
                  "fit_snapshot_hash": finalized["fit_snapshot_hash"],
                  "source_val_snapshot_hash": finalized["source_val_snapshot_hash"],
                  "recipe_hash": finalized["recipe_hash"], "init_provenance": {"scope": "native_initialization"},
                  "task_training_provenance": provenance,
                  "eval_preprocess_hash": source_job["execution"]["preprocess_hash"],
                  "class_index_map": finalized["class_index_map"], "head_ref": "linear_head.pt",
                  "embedding_dim": 160, "training_status": "FINALIZED", "training_phase": "full",
                  "task_weight_origin": "trained_in_project",
                  "source_val_selection_ref": str(Path(finalized_ref).resolve()),
                  "parity_report_ref": "parity.json",
                  "model_contract": {"architecture": finalized["architecture"],
                                     "actual_import_path": finalized["architecture"]["repository_ref"] + "/" + finalized["architecture"]["entrypoint"],
                                     "embedding_point": "native_out_layer_input", "embedding_dim": 160,
                                     "native_logits_shape": ["B", 2], "freq_aug": False,
                                     "preprocess": {"sample_rate_hz": 16000, "channels": "mono_mean",
                                                    "sample_count": 64600, "crop": "first", "short": "repeat",
                                                    "normalization": "none", "eval_augmentation": False}},
                  "score_contract": {"formula": "native_logits[spoof]-native_logits[bonafide]",
                                     "direction": "larger_is_spoof", "output_type": "logit_difference",
                                     "unit": "dimensionless"},
                  "numerical_contract": {"device": "CUDA", "dtype": "float32", "amp": False,
                                         "atol": ATOL, "rtol": RTOL,
                                         "tf32_matmul": parity["tf32_matmul"], "tf32_cudnn": parity["tf32_cudnn"]},
                  "r4_validation": {"status": "PASS", "fit_count": paths["fit_evidence"]["count"],
                                    "source_val_count": source_val_count,
                                    "parity_policy": paths["fit_evidence"]["parity_policy"],
                                    "fit_uids_sha256": sha256_file(temporary / "fit128_uids.json"),
                                    "parity_sha256": sha256_file(temporary / "parity.json"),
                                    "per_sample_sha256": sha256_file(temporary / "parity_per_sample.jsonl"),
                                    "source_val_recompute_sha256": sha256_file(temporary / "source_val_recompute.json")},
                  "export_code_sha256": sha256_file(worker)}
        FrozenModelBundle(**bundle)
        write_json_new(temporary / "bundle.json", bundle)
        files = {path.name: sha256_file(path) for path in temporary.iterdir() if path.is_file()}
        write_json_new(temporary / "export_manifest.json",
                       {"schema_version": "0.1.0", "status": "LOCKED", "immutable": True,
                        "r5_eligible": True, "files": files})
    from eptta.models.frozen import verify_frozen_export
    verified = verify_frozen_export(published / "bundle.json")
    return {"schema_version": "0.1.0", "status": "PASS", "r5_eligible": True,
            "bundle_id": verified[0]["baseline_id"], "bundle_ref": str((published / "bundle.json").resolve()),
            "bundle_sha256": sha256_file(published / "bundle.json"),
            "export_manifest_sha256": sha256_file(published / "export_manifest.json"),
            "parity_report_sha256": sha256_file(published / "parity.json"),
            "gpu_hours": parity["gpu_hours"], "gpu_hour_cap": GPU_HOUR_CAP}
