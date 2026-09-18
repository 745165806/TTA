"""Run registered label-free cache methods with per-sample reset."""
import json
import time
from dataclasses import replace
from pathlib import Path

from eptta.adaptation.adapter import adaptation_diagnostics, validate_method_setup
from eptta.baselines.dispatch import run_method
from eptta.baselines.registry import get_method_contract
from eptta.cache.reader import FeatureCache
from eptta.config.schema import read_document
from eptta.data.io import AtomicDirectory, iter_jsonl, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


_FIELDS = {"schema_version", "status", "suite_id", "feature_cache_ref", "resources_ref",
           "frozen_bundle_ref", "input_manifest_ref", "input_role", "scope_id", "methods"}
_OPTIONAL = {"fallback_rate_max"}


def _validate_plan(plan, suite_id, phase):
    if (set(plan) - (_FIELDS | _OPTIONAL) or _FIELDS - set(plan) or
            plan.get("schema_version") != "0.3.0" or plan.get("status") != "READY" or
            plan.get("suite_id") != suite_id):
        raise ContractError("adaptation plan must be a strict matching READY document")
    allowed = {"select"} if phase == "select" else ({"audit", "cal1"} if phase == "audit" else
                                                       {"control_test", "target_test"})
    if phase not in ("select", "confirmatory", "audit") or plan["input_role"] not in allowed:
        raise ContractError("suite phase cannot consume this input role")
    if not isinstance(plan["scope_id"], str) or not plan["scope_id"]:
        raise ContractError("suite requires a nonempty scope_id")


def _manifest_ids(plan, cache):
    path = Path(plan["input_manifest_ref"])
    if not path.is_file():
        raise DataError("suite input manifest is missing")
    ids = []
    required = {"schema_version", "sample_id", "root_key", "audio_relpath", "split_role"}
    allowed = required | {"sample_index"}
    for number, row in enumerate(iter_jsonl(path), 1):
        if not required.issubset(row) or set(row) - allowed or row.get("split_role") != plan["input_role"]:
            raise DataError("suite manifest row %d contains labels/metadata or a mismatched role" % number)
        ids.append(row["sample_id"])
    if not ids or len(ids) != len(set(ids)):
        raise DataError("suite manifest IDs must be unique and nonempty")
    cache.verify_expected_ids(ids)
    identity = cache.index.get("identity", {})
    if cache.index.get("format") == "sharded_npy_v2":
        if (identity.get("split_role") != plan["input_role"] or
                Path(identity.get("manifest_ref", "")).resolve() != path.resolve()):
            raise ContractError("cache declared manifest/role differs from the suite")
    return ids


def _prepare_method(item, resources, extras, resource_meta):
    from eptta.adaptation.types import EPConfig
    if set(item) != {"method_id", "config", "params"}:
        raise ContractError("method entries require only method_id/config/params")
    try:
        cfg = EPConfig(**item["config"])
        contract = get_method_contract(item["method_id"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ContractError("invalid method configuration: %s" % item.get("method_id")) from exc
    params, method_resources = dict(item["params"]), resources
    unavailable = None
    if contract["comparison_track"] == "published_port":
        unavailable = ("NOT_RUN_UNVERIFIED_PORT", "published audio port has not passed model parity")
    elif contract["implementation_status"] not in ("IMPLEMENTED_SYNTHETIC_VERIFIED", "IMPLEMENTED_UNVERIFIED"):
        unavailable = ("NOT_IMPLEMENTED", "method implementation is unavailable")
    if item["method_id"] == "ep_feature_pca_U":
        if extras.get("U_feature_pca") is None:
            unavailable = ("NOT_RUN_MISSING_RESOURCE", "U_feature_pca is unavailable")
        else:
            method_resources = replace(resources, U=extras["U_feature_pca"],
                                       artifact_bundle_id=resource_meta["artifact_bundle_id"] + ":feature_pca")
    if item["method_id"] == "ep_random_U":
        index = params.pop("random_index", None)
        if index not in (0, 1, 2):
            raise ContractError("ep_random_U requires random_index 0/1/2")
        random_u = extras.get("U_random_%d" % index)
        if random_u is None:
            unavailable = ("NOT_RUN_MISSING_RESOURCE", "configured random U is unavailable")
        else:
            method_resources = replace(resources, U=random_u,
                                       artifact_bundle_id=resource_meta["artifact_bundle_id"] + ":random%d" % index)
    if item["method_id"] == "ep_keep_fisher":
        if extras.get("fisher") is None:
            raise ContractError("ep_keep_fisher requires Fisher")
        params["fisher"] = extras["fisher"]
    if item["method_id"] == "fixed_source_adapter":
        if extras.get("fixed_R") is None:
            raise ContractError("fixed_source_adapter requires fixed_R")
        params["fixed_R"] = extras["fixed_R"]
    if unavailable is None:
        try:
            validate_method_setup(item["method_id"], method_resources, cfg, params)
        except ValueError as exc:
            raise ContractError("invalid method/resource setup for %s: %s" %
                                (item["method_id"], exc)) from exc
    return cfg, params, method_resources, unavailable


def _not_run(method_dir, suite_id, phase, method_id, bundle, cache, resource_meta, count, unavailable):
    reason_code, reason = unavailable
    run = {"schema_version": "0.3.0", "status": "NOT_RUN", "suite_id": suite_id,
           "phase": phase, "method_id": method_id, "baseline_id": bundle["baseline_id"],
           "checkpoint_ref": bundle.get("checkpoint_ref") or bundle.get("selected_checkpoint_ref"),
           "cache_id": cache.cache_id, "artifact_bundle_id": resource_meta["artifact_bundle_id"],
           "expected_sample_count": count, "sample_count": 0, "coverage_fraction": 0.0,
           "reason_code": reason_code, "reason": reason, "target_labels_read": False,
           "reset_policy": "per_sample"}
    write_json_new(method_dir / "run.json", run)
    return {"method_id": method_id, "status": "NOT_RUN", "reason_code": reason_code,
            "run_ref": method_id}


def _score_method(method_dir, item, cfg, params, resources, cache, features, expected_ids,
                  suite_id, phase, bundle, resource_meta, fallback_limit):
    import torch
    from eptta.adaptation.types import TargetViews
    score_path, runtime_path = method_dir / "scores.jsonl", method_dir / "runtime.jsonl"
    diagnostics_path, events_path = method_dir / "diagnostics.jsonl", method_dir / "events.jsonl"
    elapsed = steps = evaluations = fallbacks = 0
    with score_path.open("x", encoding="utf-8") as scores, runtime_path.open("x", encoding="utf-8") as runtimes, \
            diagnostics_path.open("x", encoding="utf-8") as diagnostics, events_path.open("x", encoding="utf-8") as events:
        for sample_id in sorted(expected_ids):
            target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache.cache_id)
            started = time.perf_counter()
            result = run_method(item["method_id"], target, resources, cfg, params)
            try:
                diag = adaptation_diagnostics(result, target, resources, cfg)
            except FloatingPointError as exc:
                score = float(target.features[0] @ resources.w + resources.b)
                result = {"method_id": item["method_id"], "sample_id": sample_id,
                          "score": score, "score_before": score, "status": "fallback_numeric",
                          "steps_completed": result["steps_completed"],
                          "objective_evaluations": result["objective_evaluations"],
                          "error_type": type(exc).__name__, "error_message": str(exc)}
                diag = adaptation_diagnostics(result, target, resources, cfg)
            seconds = time.perf_counter() - started
            elapsed += seconds
            steps += result["steps_completed"]
            evaluations += result["objective_evaluations"]
            status = result.get("status", "ok")
            if status == "fallback_numeric":
                fallbacks += 1
                events.write(json.dumps({"schema_version": "0.3.0", "sample_id": sample_id,
                                         "method_id": item["method_id"], "status": status,
                                         "exception_type": result["error_type"],
                                         "exception_message": result["error_message"],
                                         "completed_steps": result["steps_completed"]},
                                        sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            scores.write(json.dumps({"schema_version": "0.3.0", "sample_id": sample_id,
                                     "score": result["score"], "score_before": result["score_before"],
                                     "status": status, "method_id": item["method_id"],
                                     "steps_completed": result["steps_completed"],
                                     "objective_evaluations": result["objective_evaluations"],
                                     "r_fro": diag.get("final_R_norm"), "runtime_ref": "runtime.jsonl"},
                                    sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            diagnostics.write(json.dumps({"schema_version": "0.3.0", "sample_id": sample_id,
                                          "method_id": item["method_id"], "status": status, **diag},
                                         sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            runtimes.write(json.dumps({"schema_version": "0.3.0", "sample_id": sample_id,
                                       "method_id": item["method_id"], "scope": "cache_adaptation_only",
                                       "elapsed_seconds": seconds,
                                       "steps_completed": result["steps_completed"],
                                       "objective_evaluations": result["objective_evaluations"]},
                                      sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    write_json_new(method_dir / "expected_ids.json", sorted(expected_ids))
    fallback_rate = fallbacks / len(expected_ids)
    valid = fallback_rate <= fallback_limit
    run = {"schema_version": "0.3.0", "status": "SCORED", "suite_id": suite_id,
           "phase": phase, "method_id": item["method_id"], "baseline_id": bundle["baseline_id"],
           "checkpoint_ref": bundle.get("checkpoint_ref") or bundle.get("selected_checkpoint_ref"),
           "cache_id": cache.cache_id, "artifact_bundle_id": resource_meta["artifact_bundle_id"],
           "expected_sample_count": len(expected_ids), "sample_count": len(expected_ids),
           "coverage_fraction": 1.0, "expected_ids_ref": "expected_ids.json",
           "scores_ref": "scores.jsonl", "diagnostics_ref": "diagnostics.jsonl",
           "events_ref": "events.jsonl", "fallback_count": fallbacks,
           "fallback_rate": fallback_rate, "fallback_rate_max": fallback_limit,
           "valid_for_comparison": valid, "target_labels_read": False,
           "reset_policy": "per_sample", "cost": {"scope": "cache_adaptation_only",
           "elapsed_seconds": elapsed, "steps_completed": steps,
           "objective_evaluations": evaluations, "runtime_ref": "runtime.jsonl",
           "feature_extraction_included": False}}
    write_json_new(method_dir / "run.json", run)
    return {"method_id": item["method_id"], "status": "SCORED", "run_ref": item["method_id"],
            "fallback_count": fallbacks, "valid_for_comparison": valid}


def run_suite(plan_ref, suite_id, phase, output, frozen_spec_ref=None):
    if frozen_spec_ref is not None:
        raise ContractError("score locks are no longer part of the research workflow")
    plan = read_document(plan_ref)
    _validate_plan(plan, suite_id, phase)
    bundle, _export, _parity, _selection = verify_frozen_export(plan["frozen_bundle_ref"])
    resources, extras, resource_meta = load_frozen_resources(plan["resources_ref"], bundle)
    cache = FeatureCache(plan["feature_cache_ref"])
    identity = cache.index.get("identity", {})
    if identity.get("source_run_id") and bundle.get("source_run_id") and \
            identity["source_run_id"] != bundle["source_run_id"]:
        raise ContractError("feature cache and frozen detector name different source runs")
    if identity.get("checkpoint_ref") and bundle.get("checkpoint_ref") and \
            Path(identity["checkpoint_ref"]).resolve() != Path(bundle["checkpoint_ref"]).resolve():
        raise ContractError("feature cache and frozen detector name different epoch checkpoints")
    if cache.index.get("format") == "sharded_npy_v2" and \
            identity.get("preprocess") != bundle.get("preprocess"):
        raise ContractError("feature cache and frozen detector name different preprocessing")
    if identity.get("baseline_id") and identity["baseline_id"] != bundle["baseline_id"]:
        raise ContractError("legacy cache does not share the frozen detector")
    expected_ids = _manifest_ids(plan, cache)
    features = cache.load_by_id()
    methods = plan["methods"]
    if not methods or len({item.get("method_id") for item in methods}) != len(methods):
        raise ContractError("suite methods must be nonempty and unique")
    prepared = [_prepare_method(item, resources, extras, resource_meta) for item in methods]
    fallback_limit = plan.get("fallback_rate_max", 0.01)
    if type(fallback_limit) not in (int, float) or isinstance(fallback_limit, bool) or not 0 <= fallback_limit <= 1:
        raise ContractError("fallback_rate_max must be in [0,1]")
    with AtomicDirectory(output) as temporary:
        runs = []
        for item, (cfg, params, method_resources, unavailable) in zip(methods, prepared):
            method_dir = temporary / item["method_id"]
            method_dir.mkdir()
            if unavailable:
                runs.append(_not_run(method_dir, suite_id, phase, item["method_id"], bundle, cache,
                                     resource_meta, len(expected_ids), unavailable))
            else:
                runs.append(_score_method(method_dir, item, cfg, params, method_resources, cache, features,
                                          expected_ids, suite_id, phase, bundle, resource_meta, fallback_limit))
        status = "PARTIAL" if any(row["status"] == "NOT_RUN" for row in runs) else "SCORED"
        index = {"schema_version": "0.3.0", "status": status, "suite_id": suite_id,
                 "phase": phase, "comparison_group_id": suite_id,
                 "baseline_id": bundle["baseline_id"], "methods": runs,
                 "target_labels_read": False,
                 "valid_for_comparison": all(row.get("valid_for_comparison", True) for row in runs
                                             if row["status"] != "NOT_RUN")}
        write_json_new(temporary / "suite.json", index)
    return index
