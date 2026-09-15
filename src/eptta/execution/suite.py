"""Run registered cache-based methods without opening evaluation labels."""
import json
from dataclasses import replace
from pathlib import Path
import time

from eptta.baselines.dispatch import ComparisonGroup, run_method
from eptta.baselines.registry import get_method_contract
from eptta.cache.reader import FeatureCache
from eptta.config.schema import read_document
from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, sha256_file, write_json_new
from eptta.errors import ContractError
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


def run_suite(plan_ref, suite_id, phase, output, frozen_spec_ref=None):
    import torch
    plan = read_document(plan_ref)
    required = {"schema_version", "status", "suite_id", "feature_cache_ref", "resources_ref",
                "frozen_bundle_ref", "methods"}
    if set(plan) != required or plan.get("status") != "LOCKED" or plan.get("suite_id") != suite_id:
        raise ContractError("adaptation plan must be a strict matching LOCKED v0.1.0 document")
    if phase not in ("select", "confirmatory"):
        raise ContractError("invalid suite phase")
    if phase == "confirmatory":
        if not frozen_spec_ref:
            raise ContractError("confirmatory suite requires a preregistered frozen spec")
        frozen_spec = read_document(frozen_spec_ref)
        frozen_required = {"schema_version", "status", "suite_id", "phase", "selected_on_role",
                           "plan_ref", "plan_sha256", "methods"}
        if (set(frozen_spec) != frozen_required or frozen_spec.get("schema_version") != "0.1.0" or
                frozen_spec.get("status") != "LOCKED" or frozen_spec.get("suite_id") != suite_id or
                frozen_spec.get("phase") != "confirmatory" or
                frozen_spec.get("selected_on_role") != "select" or
                Path(frozen_spec.get("plan_ref", "")).resolve() != Path(plan_ref).resolve() or
                frozen_spec.get("plan_sha256") != sha256_file(plan_ref) or
                frozen_spec.get("methods") != plan["methods"]):
            raise ContractError("confirmatory plan/methods are not identical to the preregistered lock")
    bundle, _export, _parity, _selection = verify_frozen_export(plan["frozen_bundle_ref"])
    resources, extras, resource_meta = load_frozen_resources(plan["resources_ref"], bundle)
    cache = FeatureCache(plan["feature_cache_ref"])
    if cache.index["identity"]["baseline_id"] != bundle["baseline_id"] or cache.index["identity"][
            "selected_checkpoint_sha256"] != bundle["selected_checkpoint_sha256"]:
        raise ContractError("feature cache does not share the suite frozen detector")
    group = ComparisonGroup("suite-" + content_hash({"suite": suite_id, "bundle": bundle["baseline_id"]})[:20],
                            bundle["baseline_id"], bundle["selected_checkpoint_sha256"])
    group.validate_run(bundle)
    from eptta.adaptation.types import EPConfig, TargetViews
    features = cache.load_by_id()
    methods = plan["methods"]
    if not methods or len({item.get("method_id") for item in methods}) != len(methods):
        raise ContractError("suite methods must be nonempty and unique")
    with AtomicDirectory(output) as temporary:
        method_runs = []
        for item in methods:
            if set(item) != {"method_id", "config", "params"}:
                raise ContractError("method entries require only method_id/config/params")
            cfg = EPConfig(**item["config"])
            params = dict(item["params"])
            method_contract = get_method_contract(item["method_id"])
            unavailable = None
            if method_contract["comparison_track"] == "published_port":
                unavailable = ("BLOCKED_AUDIT",
                               "published audio port has not passed its author/parity audit")
            elif method_contract["implementation_status"] not in (
                    "IMPLEMENTED_SYNTHETIC_VERIFIED", "IMPLEMENTED_UNVERIFIED"):
                unavailable = ("NOT_IMPLEMENTED_STAGE", "method implementation is unavailable")
            method_resources = resources
            if item["method_id"] == "ep_feature_pca_U":
                if extras.get("U_feature_pca") is None:
                    unavailable = ("NOT_RUN_MISSING_RESOURCE", "U_feature_pca is unavailable")
                else:
                    method_resources = replace(resources, U=extras["U_feature_pca"],
                                               artifact_bundle_id=resource_meta["artifact_bundle_id"] + ":feature_pca")
            if item["method_id"] == "ep_random_U":
                index = params.pop("random_index", None)
                if index not in (0, 1, 2):
                    raise ContractError("ep_random_U requires preregistered random_index 0/1/2")
                random_u = extras.get("U_random_%d" % index)
                if random_u is None:
                    unavailable = ("NOT_RUN_MISSING_RESOURCE", "preregistered random U is unavailable")
                else:
                    method_resources = replace(resources, U=random_u,
                                               artifact_bundle_id=resource_meta["artifact_bundle_id"] + ":random%d" % index)
            if item["method_id"] == "ep_keep_fisher":
                if extras.get("fisher") is None:
                    unavailable = ("NOT_RUN_MISSING_RESOURCE", "source Fisher resource is unavailable")
                else:
                    params["fisher"] = extras["fisher"]
            if item["method_id"] == "fixed_source_adapter":
                if extras.get("fixed_R") is None:
                    unavailable = ("NOT_RUN_MISSING_RESOURCE", "fixed source adapter is unavailable")
                else:
                    params["fixed_R"] = extras["fixed_R"]
            method_dir = temporary / item["method_id"]
            method_dir.mkdir()
            if unavailable is not None:
                reason_code, reason = unavailable
                run = {"schema_version": "0.1.0", "status": "NOT_RUN", "suite_id": suite_id,
                       "phase": phase, "method_id": item["method_id"],
                       "baseline_id": bundle["baseline_id"],
                       "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
                       "feature_cache_key": cache.index["cache_key"],
                       "artifact_bundle_id": resource_meta["artifact_bundle_id"],
                       "expected_sample_count": len(features), "sample_count": 0,
                       "coverage_fraction": 0.0, "reason_code": reason_code, "reason": reason,
                       "target_labels_read": False, "reset_policy": "per_sample"}
                write_json_new(method_dir / "run.json", run)
                method_runs.append({"method_id": item["method_id"], "status": "NOT_RUN",
                                    "reason_code": reason_code, "run_ref": item["method_id"]})
                continue
            score_path = method_dir / "scores.jsonl"
            runtime_path = method_dir / "runtime.jsonl"
            elapsed = 0.0
            steps_completed = 0
            objective_evaluations = 0
            with score_path.open("x", encoding="utf-8") as stream, runtime_path.open(
                    "x", encoding="utf-8") as runtime_stream:
                for sample_id in sorted(features):
                    tensor = torch.from_numpy(features[sample_id])
                    target = TargetViews(sample_id, tensor, cache.index["cache_key"])
                    started = time.perf_counter()
                    result = run_method(item["method_id"], target, method_resources, cfg, params)
                    sample_seconds = time.perf_counter() - started
                    elapsed += sample_seconds
                    steps_completed += result["steps_completed"]
                    objective_evaluations += result["objective_evaluations"]
                    r_value = result.get("R")
                    r_fro = float(torch.linalg.vector_norm(r_value)) if r_value is not None else None
                    row = {"schema_version": "0.1.0", "sample_id": sample_id,
                           "score": result["score"], "score_before": result["score_before"],
                           "status": "ok", "method_id": item["method_id"],
                           "steps_completed": result["steps_completed"],
                           "objective_evaluations": result["objective_evaluations"],
                           "r_fro": r_fro, "runtime_ref": "runtime.jsonl"}
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                    runtime_row = {"schema_version": "0.1.0", "sample_id": sample_id,
                                   "method_id": item["method_id"], "scope": "cache_adaptation_only",
                                   "elapsed_seconds": sample_seconds,
                                   "steps_completed": result["steps_completed"],
                                   "objective_evaluations": result["objective_evaluations"]}
                    runtime_stream.write(json.dumps(runtime_row, sort_keys=True, separators=(",", ":"),
                                                    allow_nan=False) + "\n")
            run = {"schema_version": "0.1.0", "status": "SCORED_UNSEALED", "suite_id": suite_id,
                   "phase": phase, "method_id": item["method_id"], "baseline_id": bundle["baseline_id"],
                   "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
                   "feature_cache_key": cache.index["cache_key"],
                   "artifact_bundle_id": resource_meta["artifact_bundle_id"],
                   "expected_sample_count": len(features), "sample_count": len(features),
                   "coverage_fraction": 1.0, "scores_ref": "scores.jsonl",
                   "scores_sha256": sha256_file(score_path), "target_labels_read": False,
                   "reset_policy": "per_sample",
                   "cost": {"scope": "cache_adaptation_only", "elapsed_seconds": elapsed,
                            "steps_completed": steps_completed,
                            "objective_evaluations": objective_evaluations,
                            "runtime_ref": "runtime.jsonl",
                            "runtime_sha256": sha256_file(runtime_path),
                            "feature_extraction_included": False}}
            write_json_new(method_dir / "run.json", run)
            method_runs.append({"method_id": item["method_id"], "status": "SCORED_UNSEALED",
                                "run_ref": item["method_id"],
                                "scores_sha256": run["scores_sha256"]})
        suite_status = "PARTIAL" if any(item["status"] == "NOT_RUN" for item in method_runs) else "SCORED_UNSEALED"
        index = {"schema_version": "0.1.0", "status": suite_status, "suite_id": suite_id,
                 "phase": phase, "comparison_group_id": group.comparison_group_id,
                 "baseline_id": bundle["baseline_id"], "methods": method_runs, "target_labels_read": False}
        write_json_new(temporary / "suite.json", index)
    return index
