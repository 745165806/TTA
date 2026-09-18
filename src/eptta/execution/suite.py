"""Run registered cache methods without opening evaluation labels."""
import json
from dataclasses import replace
from pathlib import Path
import time

from eptta.adaptation.adapter import adaptation_diagnostics, validate_method_setup
from eptta.baselines.dispatch import ComparisonGroup, run_method
from eptta.baselines.registry import get_method_contract
from eptta.cache.reader import FeatureCache
from eptta.config.schema import read_document
from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


_LEGACY_FIELDS = {"schema_version", "status", "suite_id", "feature_cache_ref", "resources_ref",
                  "frozen_bundle_ref", "methods"}
_V2_FIELDS = _LEGACY_FIELDS | {"input_manifest_ref", "input_manifest_sha256", "input_role", "scope_id"}
_V2_OPTIONAL = {"fallback_rate_max", "input_extraction_plan_ref", "input_extraction_plan_sha256"}


def _validate_plan(plan, suite_id, phase):
    fields = set(plan)
    legacy = fields == _LEGACY_FIELDS and plan.get("schema_version") == "0.1.0"
    current = _V2_FIELDS.issubset(fields) and not fields - (_V2_FIELDS | _V2_OPTIONAL) and plan.get(
        "schema_version") == "0.2.0"
    if not (legacy or current) or plan.get("status") != "LOCKED" or plan.get("suite_id") != suite_id:
        raise ContractError("adaptation plan must be a strict matching LOCKED v0.1.0/v0.2.0 document")
    if phase not in ("select", "confirmatory", "audit"):
        raise ContractError("invalid suite phase")
    if current:
        allowed = {"select"} if phase == "select" else ({"audit"} if phase == "audit" else
                                                          {"control_test", "target_test"})
        if plan["input_role"] not in allowed:
            raise ContractError("suite phase cannot consume this approved input role")
        if type(plan["scope_id"]) is not str or not plan["scope_id"]:
            raise ContractError("suite requires a nonempty approved scope_id")
    return current


def _manifest_ids(plan, cache, current):
    if not current:
        return sorted(cache.load_by_id())
    path = Path(plan["input_manifest_ref"])
    if not path.is_file() or sha256_file(path) != plan["input_manifest_sha256"]:
        raise DataError("suite input manifest is missing or changed")
    if cache.index["identity"].get("input_manifest_sha256") != plan["input_manifest_sha256"]:
        raise ContractError("cache identity does not match the approved suite manifest")
    ids = []
    allowed = {"schema_version", "sample_id", "root_key", "audio_relpath", "input_sha256", "split_role"}
    for number, row in enumerate(iter_jsonl(path), 1):
        if set(row) != allowed or row.get("split_role") != plan["input_role"]:
            raise DataError("suite manifest row %d contains labels or a mismatched role" % number)
        ids.append(row["sample_id"])
    if not ids or len(ids) != len(set(ids)):
        raise DataError("suite manifest IDs must be unique and nonempty")
    cache.verify_expected_ids(ids)
    index_path = path.parent / "inference_manifest.json"
    if index_path.is_file():
        index = read_document(index_path)
        if (index.get("status") != "LOCKED" or index.get("role") != plan["input_role"] or
                index.get("manifest_sha256") != plan["input_manifest_sha256"] or
                not index.get("source_role_manifest_sha256") or index.get("labels_in_manifest") is not False or
                index.get("immutable") is not True):
            raise ContractError("suite manifest is not traceable to an approved snapshot role view")
    else:
        extraction_ref = plan.get("input_extraction_plan_ref")
        extraction_sha = plan.get("input_extraction_plan_sha256")
        if not extraction_ref or sha256_file(extraction_ref) != extraction_sha:
            raise DataError("legacy cache is not traceable to its locked extraction plan")
        extraction = read_document(extraction_ref)
        if (extraction.get("status") != "LOCKED" or extraction.get("input_role") != plan["input_role"] or
                Path(extraction.get("manifest_ref", "")).resolve() != path.resolve() or
                extraction.get("manifest_sha256") != plan["input_manifest_sha256"]):
            raise ContractError("legacy extraction plan does not approve this manifest role/identity")
    return ids


def _validate_frozen_spec(frozen_spec_ref, plan_ref, plan, suite_id):
    if not frozen_spec_ref:
        raise ContractError("confirmatory suite requires a preregistered frozen spec")
    spec = read_document(frozen_spec_ref)
    required = {"schema_version", "status", "suite_id", "phase", "selected_on_role",
                "plan_ref", "plan_sha256", "methods"}
    if (set(spec) != required or spec.get("status") != "LOCKED" or spec.get("suite_id") != suite_id or
            spec.get("phase") != "confirmatory" or spec.get("selected_on_role") != "select" or
            Path(spec.get("plan_ref", "")).resolve() != Path(plan_ref).resolve() or
            spec.get("plan_sha256") != sha256_file(plan_ref) or spec.get("methods") != plan["methods"]):
        raise ContractError("confirmatory plan/methods are not identical to the preregistered lock")


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
        unavailable = ("BLOCKED_AUDIT", "published audio port has not passed its author/parity audit")
    elif contract["implementation_status"] not in ("IMPLEMENTED_SYNTHETIC_VERIFIED", "IMPLEMENTED_UNVERIFIED"):
        unavailable = ("NOT_IMPLEMENTED_STAGE", "method implementation is unavailable")
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
            raise ContractError("ep_keep_fisher requires the locked Fisher resource")
        params["fisher"] = extras["fisher"]
    if item["method_id"] == "fixed_source_adapter":
        if extras.get("fixed_R") is None:
            raise ContractError("fixed_source_adapter requires the locked fixed_R resource")
        params["fixed_R"] = extras["fixed_R"]
    if unavailable is None:
        try:
            validate_method_setup(item["method_id"], method_resources, cfg, params)
        except ValueError as exc:
            raise ContractError("invalid method/resource setup for %s: %s" % (item["method_id"], exc)) from exc
    return cfg, params, method_resources, unavailable


def _not_run(method_dir, suite_id, phase, method_id, bundle, cache, resource_meta, count, unavailable):
    reason_code, reason = unavailable
    run = {"schema_version": "0.2.0", "status": "NOT_RUN", "suite_id": suite_id, "phase": phase,
           "method_id": method_id, "baseline_id": bundle["baseline_id"],
           "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
           "feature_cache_key": cache.index["cache_key"], "artifact_bundle_id": resource_meta["artifact_bundle_id"],
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
            target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache.index["cache_key"])
            started = time.perf_counter()
            result = run_method(item["method_id"], target, resources, cfg, params)
            try:
                diag = adaptation_diagnostics(result, target, resources, cfg)
            except FloatingPointError as exc:
                result = {"method_id": item["method_id"], "sample_id": sample_id,
                          "score": float(target.features[0] @ resources.w + resources.b),
                          "score_before": float(target.features[0] @ resources.w + resources.b),
                          "status": "fallback_numeric", "steps_completed": result["steps_completed"],
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
                event = {"schema_version": "0.2.0", "sample_id": sample_id, "method_id": item["method_id"],
                         "status": status, "exception_type": result["error_type"],
                         "exception_message": result["error_message"],
                         "completed_steps": result["steps_completed"]}
                events.write(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            score_row = {"schema_version": "0.2.0", "sample_id": sample_id, "score": result["score"],
                         "score_before": result["score_before"], "status": status,
                         "method_id": item["method_id"], "steps_completed": result["steps_completed"],
                         "objective_evaluations": result["objective_evaluations"],
                         "r_fro": diag.get("final_R_norm"), "runtime_ref": "runtime.jsonl"}
            scores.write(json.dumps(score_row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            diag_row = {"schema_version": "0.2.0", "sample_id": sample_id,
                        "method_id": item["method_id"], "status": status, **diag}
            diagnostics.write(json.dumps(diag_row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            runtime = {"schema_version": "0.2.0", "sample_id": sample_id, "method_id": item["method_id"],
                       "scope": "cache_adaptation_only", "elapsed_seconds": seconds,
                       "steps_completed": result["steps_completed"],
                       "objective_evaluations": result["objective_evaluations"]}
            runtimes.write(json.dumps(runtime, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    write_json_new(method_dir / "expected_ids.json", sorted(expected_ids))
    fallback_rate = fallbacks / len(expected_ids)
    valid = fallback_rate <= fallback_limit
    run = {"schema_version": "0.2.0", "status": "SCORED_UNSEALED", "suite_id": suite_id,
           "phase": phase, "method_id": item["method_id"], "baseline_id": bundle["baseline_id"],
           "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
           "feature_cache_key": cache.index["cache_key"], "artifact_bundle_id": resource_meta["artifact_bundle_id"],
           "expected_sample_count": len(expected_ids), "sample_count": len(expected_ids), "coverage_fraction": 1.0,
           "expected_ids_ref": "expected_ids.json", "expected_ids_sha256": sha256_file(method_dir / "expected_ids.json"),
           "scores_ref": "scores.jsonl", "scores_sha256": sha256_file(score_path),
           "diagnostics_ref": "diagnostics.jsonl", "diagnostics_sha256": sha256_file(diagnostics_path),
           "events_ref": "events.jsonl", "events_sha256": sha256_file(events_path),
           "fallback_count": fallbacks, "fallback_rate": fallback_rate,
           "fallback_rate_max": fallback_limit, "valid_for_comparison": valid,
           "target_labels_read": False, "reset_policy": "per_sample",
           "cost": {"scope": "cache_adaptation_only", "elapsed_seconds": elapsed,
                    "steps_completed": steps, "objective_evaluations": evaluations,
                    "runtime_ref": "runtime.jsonl", "runtime_sha256": sha256_file(runtime_path),
                    "feature_extraction_included": False}}
    write_json_new(method_dir / "run.json", run)
    return {"method_id": item["method_id"], "status": "SCORED_UNSEALED", "run_ref": item["method_id"],
            "scores_sha256": run["scores_sha256"], "fallback_count": fallbacks,
            "valid_for_comparison": valid}


def run_suite(plan_ref, suite_id, phase, output, frozen_spec_ref=None):
    plan = read_document(plan_ref)
    current = _validate_plan(plan, suite_id, phase)
    if phase == "confirmatory":
        _validate_frozen_spec(frozen_spec_ref, plan_ref, plan, suite_id)
    bundle, _export, _parity, _selection = verify_frozen_export(plan["frozen_bundle_ref"])
    resources, extras, resource_meta = load_frozen_resources(plan["resources_ref"], bundle)
    cache = FeatureCache(plan["feature_cache_ref"])
    if (cache.index["identity"]["baseline_id"] != bundle["baseline_id"] or
            cache.index["identity"]["selected_checkpoint_sha256"] != bundle["selected_checkpoint_sha256"]):
        raise ContractError("feature cache does not share the suite frozen detector")
    expected_ids = _manifest_ids(plan, cache, current)
    features = cache.load_by_id()
    methods = plan["methods"]
    if not methods or len({item.get("method_id") for item in methods}) != len(methods):
        raise ContractError("suite methods must be nonempty and unique")
    prepared = [_prepare_method(item, resources, extras, resource_meta) for item in methods]
    group = ComparisonGroup("suite-" + content_hash({"suite": suite_id, "bundle": bundle["baseline_id"]})[:20],
                            bundle["baseline_id"], bundle["selected_checkpoint_sha256"])
    group.validate_run(bundle)
    fallback_limit = plan.get("fallback_rate_max", 0.01)
    if type(fallback_limit) not in (int, float) or not 0 <= fallback_limit <= 1:
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
        status = "PARTIAL" if any(row["status"] == "NOT_RUN" for row in runs) else "SCORED_UNSEALED"
        index = {"schema_version": "0.2.0", "status": status, "suite_id": suite_id, "phase": phase,
                 "comparison_group_id": group.comparison_group_id, "baseline_id": bundle["baseline_id"],
                 "methods": runs, "target_labels_read": False,
                 "valid_for_comparison": all(row.get("valid_for_comparison", True) for row in runs
                                             if row["status"] != "NOT_RUN")}
        write_json_new(temporary / "suite.json", index)
    return index


def select_methods(metrics_root, policy_ref, output):
    """Select only from preregistered source-select evaluations."""
    policy = read_document(policy_ref)
    if policy.get("status") != "LOCKED" or type(policy.get("candidates")) is not list:
        raise ContractError("selection policy must be LOCKED and list candidates")
    root = Path(metrics_root).resolve()
    rows = []
    frozen_eer = None
    for candidate in policy["candidates"]:
        required = {"candidate_id", "family", "method_id", "suite_id", "suite_plan_ref",
                    "suite_plan_sha256", "evaluation_ref", "coverage_min", "fallback_rate_max",
                    "rank_order", "candidate_config_sha256"}
        if not required.issubset(candidate):
            raise ContractError("selection candidate is missing preregistered evidence fields")
        evaluation_path = (root / candidate["evaluation_ref"]).resolve()
        try:
            evaluation_path.relative_to(root)
        except ValueError as exc:
            raise ContractError("selection evaluation escapes metrics root") from exc
        plan_path = Path(candidate["suite_plan_ref"])
        if sha256_file(plan_path) != candidate["suite_plan_sha256"]:
            raise DataError("candidate suite plan changed after selection lock")
        plan = read_document(plan_path)
        if (plan.get("suite_id") != candidate["suite_id"] or plan.get("input_role") != "select" or
                len(plan.get("methods", [])) != 1 or
                plan["methods"][0].get("method_id") != candidate["method_id"]):
            raise ContractError("selection candidate is not a single source-select suite")
        if content_hash(plan["methods"][0]) != candidate["candidate_config_sha256"]:
            raise ContractError("selection candidate configuration hash differs from policy")
        if not evaluation_path.is_file():
            rows.append({**candidate, "status": "NOT_RUN", "reason": "evaluation_missing",
                         "valid": False, "eer": None})
            continue
        evaluation = read_document(evaluation_path)
        run_path = evaluation_path.parent / "run.json"
        seal_path = evaluation_path.parent / "score_seal.json"
        if not run_path.is_file() or not seal_path.is_file():
            raise ContractError("selection requires sealed source-select scores")
        run, seal = read_document(run_path), read_document(seal_path)
        if run.get("phase") != "select" or run.get("suite_id") != candidate["suite_id"]:
            raise ContractError("selection cannot consume target/confirmatory evaluations")
        if (run.get("baseline_id") != policy.get("baseline_id") or
                run.get("artifact_bundle_id") != policy.get("artifact_bundle_id")):
            raise ContractError("selection candidate bundle/resource identity differs")
        metrics = evaluation.get("metrics", {})
        eer = metrics.get("eer")
        valid = (seal.get("status") == "SEALED" and
                 evaluation.get("score_sha256") == seal.get("score_sha256") and
                 run.get("coverage_fraction", 0) >= candidate["coverage_min"] and
                 run.get("fallback_rate", 1) <= candidate["fallback_rate_max"] and
                 type(eer) in (int, float))
        row = {**candidate, "status": "EVALUATED", "valid": bool(valid), "eer": eer,
               "evaluation_sha256": sha256_file(evaluation_path),
               "method": plan["methods"][0]}
        rows.append(row)
        if candidate["method_id"] == "frozen" and valid:
            frozen_eer = eer
    selected = {}
    for family in sorted({row["family"] for row in rows}):
        valid_rows = [row for row in rows if row["family"] == family and row["valid"]]
        if valid_rows:
            selected[family] = min(valid_rows, key=lambda row: (float(row["eer"]),
                                    tuple(row["rank_order"]), row["candidate_id"]))
    ep_families = set(policy.get("required_ep_families", ["ep_tta"]))
    if ep_families and not ep_families.intersection(selected):
        raise ContractError("no valid EP candidate passed the preregistered coverage/fallback gates")
    result = {"schema_version": "0.2.0", "status": "SELECTED", "policy_ref": str(Path(policy_ref).resolve()),
              "policy_sha256": sha256_file(policy_ref), "baseline_id": policy.get("baseline_id"),
              "artifact_bundle_id": policy.get("artifact_bundle_id"), "candidates": rows,
              "selected": {family: {**row, "delta_eer_vs_frozen": None if frozen_eer is None else row["eer"] - frozen_eer,
                                     "improves_frozen": None if frozen_eer is None else row["eer"] < frozen_eer}
                           for family, row in selected.items()}}
    write_json_new(output, result)
    return result


def freeze_selection(plan_ref, selection_ref, output):
    """Bind selected source configurations to approved target manifests without reading scores."""
    plan, selection = read_document(plan_ref), read_document(selection_ref)
    if plan.get("status") != "LOCKED" or selection.get("status") != "SELECTED":
        raise ContractError("freeze requires a LOCKED target plan and SELECTED source result")
    if selection.get("policy_sha256") != plan.get("selection_policy_sha256"):
        raise ContractError("freeze plan does not bind this selection policy")
    targets = plan.get("targets")
    if type(targets) is not list or not targets:
        raise ContractError("freeze plan must list approved targets")
    methods = [row["method"] for _family, row in sorted(selection["selected"].items())]
    if not methods or len({row["method_id"] for row in methods}) != len(methods):
        raise ContractError("frozen target suite methods must be nonempty and unique")
    with AtomicDirectory(output) as temporary:
        tsv_rows = []
        for target in targets:
            required = {"target_id", "role", "scope_id", "labels_ref", "extraction_plan_ref",
                        "feature_cache_ref", "input_manifest_ref", "input_manifest_sha256"}
            if not required.issubset(target) or target["role"] not in ("control_test", "target_test"):
                raise ContractError("target freeze entry is incomplete or has an invalid role")
            target_dir = temporary / target["target_id"]
            target_dir.mkdir()
            suite_id = target.get("suite_id", "target-" + target["target_id"])
            suite = {"schema_version": "0.2.0", "status": "LOCKED", "suite_id": suite_id,
                     "feature_cache_ref": target["feature_cache_ref"], "resources_ref": plan["resources_ref"],
                     "frozen_bundle_ref": plan["frozen_bundle_ref"],
                     "input_manifest_ref": target["input_manifest_ref"],
                     "input_manifest_sha256": target["input_manifest_sha256"],
                     "input_role": target["role"], "scope_id": target["scope_id"],
                     "fallback_rate_max": plan.get("fallback_rate_max", 0.01), "methods": methods}
            suite_ref = target_dir / "suite.json"
            write_json_new(suite_ref, suite)
            suite_final = Path(output).resolve() / target["target_id"] / "suite.json"
            frozen = {"schema_version": "0.2.0", "status": "LOCKED", "suite_id": suite_id,
                      "phase": "confirmatory", "selected_on_role": "select",
                      "plan_ref": str(suite_final), "plan_sha256": sha256_file(suite_ref),
                      "methods": methods}
            frozen_ref = target_dir / "frozen_spec.json"
            write_json_new(frozen_ref, frozen)
            frozen_final = Path(output).resolve() / target["target_id"] / "frozen_spec.json"
            method_ref = target_dir / "method_ids.txt"
            method_ref.write_text("".join(row["method_id"] + "\n" for row in methods), encoding="utf-8")
            tsv_rows.append("\t".join((target["target_id"], target["role"], target["labels_ref"],
                                       target["extraction_plan_ref"], suite_id, str(suite_final),
                                       str(frozen_final))))
        (temporary / "targets.tsv").write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
        index = {"schema_version": "0.2.0", "status": "FROZEN", "target_count": len(targets),
                 "selection_sha256": sha256_file(selection_ref), "labels_read": False}
        write_json_new(temporary / "freeze.json", index)
    return index
