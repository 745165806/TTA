"""Run registered cache-based methods without opening evaluation labels."""
import json
from dataclasses import replace
from pathlib import Path

from eptta.baselines.dispatch import ComparisonGroup, run_method
from eptta.cache.reader import FeatureCache
from eptta.config.schema import read_document
from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, sha256_file, write_json_new
from eptta.errors import ContractError
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
        if frozen_spec.get("status") != "LOCKED" or frozen_spec.get("methods") != plan["methods"]:
            raise ContractError("confirmatory method list/parameters differ from frozen spec")
    bundle = read_document(plan["frozen_bundle_ref"])
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
            method_resources = resources
            if item["method_id"] == "ep_feature_pca_U":
                method_resources = replace(resources, U=extras.get("U_feature_pca"),
                                           artifact_bundle_id=resource_meta["artifact_bundle_id"] + ":feature_pca")
            if item["method_id"] == "ep_random_U":
                index = params.pop("random_index", None)
                if index not in (0, 1, 2):
                    raise ContractError("ep_random_U requires preregistered random_index 0/1/2")
                method_resources = replace(resources, U=extras.get("U_random_%d" % index),
                                           artifact_bundle_id=resource_meta["artifact_bundle_id"] + ":random%d" % index)
            if item["method_id"] == "ep_keep_fisher":
                params["fisher"] = extras.get("fisher")
            if item["method_id"] == "fixed_source_adapter":
                params["fixed_R"] = extras.get("fixed_R")
            method_dir = temporary / item["method_id"]
            method_dir.mkdir()
            score_path = method_dir / "scores.jsonl"
            with score_path.open("x", encoding="utf-8") as stream:
                for sample_id in sorted(features):
                    tensor = torch.from_numpy(features[sample_id])
                    target = TargetViews(sample_id, tensor, cache.index["cache_key"])
                    result = run_method(item["method_id"], target, method_resources, cfg, params)
                    row = {"schema_version": "0.1.0", "sample_id": sample_id,
                           "score": result["score"], "score_before": result["score_before"],
                           "status": "ok", "method_id": item["method_id"]}
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            run = {"schema_version": "0.1.0", "status": "SCORED_UNSEALED", "suite_id": suite_id,
                   "phase": phase, "method_id": item["method_id"], "baseline_id": bundle["baseline_id"],
                   "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
                   "feature_cache_key": cache.index["cache_key"],
                   "artifact_bundle_id": resource_meta["artifact_bundle_id"],
                   "sample_count": len(features), "scores_ref": "scores.jsonl",
                   "scores_sha256": sha256_file(score_path), "target_labels_read": False}
            write_json_new(method_dir / "run.json", run)
            method_runs.append({"method_id": item["method_id"], "run_ref": item["method_id"],
                                "scores_sha256": run["scores_sha256"]})
        index = {"schema_version": "0.1.0", "status": "SCORED_UNSEALED", "suite_id": suite_id,
                 "phase": phase, "comparison_group_id": group.comparison_group_id,
                 "baseline_id": bundle["baseline_id"], "methods": method_runs, "target_labels_read": False}
        write_json_new(temporary / "suite.json", index)
    return index
