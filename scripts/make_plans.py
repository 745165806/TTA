#!/usr/bin/env python3
"""Prepare reviewable EP-TTA plans.  This script never trains, scores, or auto-approves."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
from pathlib import Path

from eptta.cache.reader import FeatureCache
from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, read_json, sha256_file, write_json_new
from eptta.data.source_manifests import publish_label_free_manifest
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


def _config(path):
    value = read_json(path)
    if value.get("schema_version") not in ("0.1.0", "0.2.0"):
        raise ContractError("experiment.json must declare schema_version 0.1.0/0.2.0")
    return value


def _existing(path, expected=None, kind="resource"):
    value = Path(path).resolve()
    if not value.exists():
        raise DataError("BLOCKED_RESOURCE: %s does not exist: %s" % (kind, value))
    if expected:
        identity_file = value
        if value.is_dir():
            candidates = [value / name for name in ("snapshot.json", "resources.json", "index.json")
                          if (value / name).is_file()]
            if len(candidates) != 1:
                raise DataError("BLOCKED_RESOURCE: %s directory has no unique identity document" % kind)
            identity_file = candidates[0]
        if not identity_file.is_file() or sha256_file(identity_file) != expected:
            raise DataError("BLOCKED_RESOURCE: %s hash differs from experiment.json" % kind)
    return value


def _write_json(path, value):
    write_json_new(path, value)
    return path


def _review(root, stage):
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.name != "review.json":
            files[str(path.relative_to(root))] = sha256_file(path)
    review = {"schema_version": "0.2.0", "status": "PROPOSED", "stage": stage,
              "approval_required": True, "execution_ready": False, "files": files,
              "content_sha256": content_hash(files)}
    write_json_new(Path(root) / "review.json", review)
    digest = sha256_file(Path(root) / "review.json")
    print(digest)
    return {"status": "PROPOSED", "review_ref": str((Path(root) / "review.json").resolve()),
            "review_sha256": digest}


def _extraction_plan(cfg, role, manifest, manifest_sha, output_root):
    return {"schema_version": "0.2.0", "status": "PROPOSED", "purpose":
            "select" if role == "select" else ("confirmatory" if role in ("control_test", "target_test")
                                                else "source_prepare"),
            "input_role": role, "frozen_bundle_ref": cfg["frozen_bundle_ref"],
            "manifest_ref": str(Path(manifest).resolve()), "manifest_sha256": manifest_sha,
            "data_roots": cfg["data_roots"], "probe": cfg["probe"],
            "numerical_mode": cfg["numerical_mode"], "output_root": str(Path(output_root).resolve())}


def source(config_ref, output):
    cfg = _config(config_ref)
    source_cfg = cfg.get("source") or {}
    snapshot = _existing(source_cfg.get("snapshot_ref", ""), source_cfg.get("snapshot_sha256"),
                         "source snapshot")
    bundle_ref = _existing(source_cfg.get("frozen_bundle_ref", ""), source_cfg.get("frozen_bundle_sha256"),
                           "frozen bundle")
    verify_frozen_export(bundle_ref)
    destination = Path(output).resolve()
    with AtomicDirectory(destination) as temporary:
        role_meta = {}
        for role in ("fit", "cal0", "select"):
            role_dir = temporary / "manifests" / role
            meta = publish_label_free_manifest(snapshot, role, role_dir, status="PROPOSED")
            manifest_actual = role_dir / meta["manifest_ref"]
            manifest_final = destination / "manifests" / role / meta["manifest_ref"]
            role_meta[role] = {**meta, "manifest_ref": str(manifest_final)}
            extract = _extraction_plan(source_cfg, role, manifest_final, meta["manifest_sha256"],
                                       Path(source_cfg["cache_root"]) / role)
            _write_json(temporary / "extract" / (role + ".json"), extract)
        artifacts = dict(source_cfg.get("source_artifacts", {}))
        required = {"fit_labels_ref", "fit_groups_ref", "calibration_labels_ref", "rank", "alpha_cal",
                    "anchor_per_class", "seed", "random_seeds", "treatment_families", "samples_per_group",
                    "pair_seed", "margin_bins", "margin_epsilon", "minimum_cal0_bonafide", "fixed_adapter"}
        if not required.issubset(artifacts):
            raise ContractError("BLOCKED_CONTRACT: source.source_artifacts lacks reviewed parameters")
        for name in ("fit_labels_ref", "fit_groups_ref", "calibration_labels_ref"):
            _existing(artifacts[name], artifacts.get(name.replace("_ref", "_sha256")), name)
        artifacts.update({"schema_version": "0.2.0", "status": "PROPOSED",
                          "frozen_bundle_ref": str(bundle_ref), "fit_role": "fit",
                          "fit_manifest_sha256": role_meta["fit"]["manifest_sha256"],
                          "calibration_role": "cal0",
                          "calibration_manifest_sha256": role_meta["cal0"]["manifest_sha256"],
                          "calibration_cache_ref": str(Path(source_cfg["cache_root"]) / "cal0" /
                                                       "shard-00000-of-00001"),
                          "source_snapshot_hash": read_json(snapshot / "snapshot.json" if snapshot.is_dir()
                                                            else snapshot)["canonical_sha256"]})
        artifacts.setdefault("fit_labels_sha256", sha256_file(artifacts["fit_labels_ref"]))
        artifacts.setdefault("fit_groups_sha256", sha256_file(artifacts["fit_groups_ref"]))
        artifacts.setdefault("calibration_labels_sha256", sha256_file(artifacts["calibration_labels_ref"]))
        _write_json(temporary / "source-artifacts.json", artifacts)
    return _review(destination, "source")


def _method(candidate_id, method_id, steps, lr, rho, gamma, keep, params=None, family=None,
            claim="primary"):
    return {"candidate_id": candidate_id, "family": family or method_id, "method_id": method_id,
            "config": {"steps": steps, "lr": lr, "rho": rho, "gamma": gamma,
                       "lambda_keep": keep}, "params": params or {}, "claim": claim}


def _candidates(cfg, rank):
    rho, gamma, keep = cfg["rho"], cfg["gamma"], cfg["lambda_keep"]
    base_k, base_lr = cfg["steps"], cfg["lr"]
    rows = [_method("frozen", "frozen", 0, base_lr, rho, gamma, keep),
            _method("multiview_mean", "multiview_mean", 0, base_lr, rho, gamma, keep),
            _method("ep_full", "ep_tta", base_k, base_lr, rho, gamma, keep, family="ep_tta"),
            _method("ep_no_keep", "ep_no_keep", base_k, base_lr, rho, gamma, 0.0),
            _method("fixed_R", "fixed_source_adapter", 0, base_lr, rho, gamma, keep),
            _method("scalar", "ep_scalar_adaptive", 0, base_lr, rho, gamma, keep,
                    {"grid_size": 17})]
    for fraction in (0.0, 0.25, 0.5, 1.0):
        rows.append(_method("static_%03d" % round(fraction * 100), "static_subspace", 0, base_lr,
                            rho, gamma, keep, {"amount": fraction * rho / math.sqrt(rank)},
                            family="static_subspace"))
    for steps in (1, 3, 5):
        for multiplier in (0.3, 1.0, 3.0):
            lr = base_lr * multiplier
            rows.append(_method("ep_K%d_lr_%s" % (steps, format(lr, ".8g")), "ep_tta", steps, lr,
                                rho, gamma, keep, family="ep_tta"))
    for index in range(3):
        rows.append(_method("random_U_%d" % index, "ep_random_U", base_k, base_lr, rho, gamma, keep,
                            {"random_index": index}, claim="mechanism_exploration"))
    for method_id in ("ep_feature_pca_U", "ep_keep_l2", "ep_keep_logit", "ep_keep_fisher",
                      "entropy_same_adapter_no_keep", "entropy_same_adapter",
                      "memo_same_adapter_no_keep", "memo_same_adapter_keep", "source_ce_only"):
        rows.append(_method(method_id, method_id, base_k, base_lr, rho, gamma, keep,
                            claim="mechanism_exploration"))
    for method_id in ("tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full", "eata_audio_ep", "t2a_audio_ep"):
        rows.append(_method(method_id, method_id, base_k, base_lr, rho, gamma, keep,
                            claim="published_port_NOT_RUN_until_audit"))
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise AssertionError("candidate IDs are not unique")
    return rows


def select(config_ref, output):
    cfg = _config(config_ref)
    item = cfg.get("select") or {}
    bundle_path = _existing(item.get("frozen_bundle_ref", ""), item.get("frozen_bundle_sha256"),
                            "select frozen bundle")
    bundle, _manifest, _parity, _selection = verify_frozen_export(bundle_path)
    resources_ref = _existing(item.get("resources_ref", ""), kind="source resources")
    resources, _extras, resource_meta = load_frozen_resources(resources_ref, bundle)
    cache = FeatureCache(_existing(item.get("feature_cache_ref", ""), kind="select feature cache"))
    manifest = _existing(item.get("input_manifest_ref", ""), item.get("input_manifest_sha256"),
                         "select label-free manifest")
    if cache.index["identity"]["input_manifest_sha256"] != sha256_file(manifest):
        raise ContractError("select cache does not bind the original label-free manifest")
    candidates = _candidates(item["adaptation"], resources.U.shape[1])
    destination = Path(output).resolve()
    with AtomicDirectory(destination) as temporary:
        policy_rows, tsv = [], []
        for candidate in candidates:
            suite_id = candidate["candidate_id"]
            suite = {"schema_version": "0.2.0", "status": "PROPOSED", "suite_id": suite_id,
                     "feature_cache_ref": str(Path(item["feature_cache_ref"]).resolve()),
                     "resources_ref": str(resources_ref), "frozen_bundle_ref": str(bundle_path),
                     "input_manifest_ref": str(manifest), "input_manifest_sha256": sha256_file(manifest),
                     "input_role": "select", "scope_id": item["scope_id"],
                     "fallback_rate_max": item.get("fallback_rate_max", 0.01),
                     "methods": [{key: candidate[key] for key in ("method_id", "config", "params")}]}
            if item.get("input_extraction_plan_ref"):
                extraction = _existing(item["input_extraction_plan_ref"],
                                       item.get("input_extraction_plan_sha256"),
                                       "select extraction plan")
                suite["input_extraction_plan_ref"] = str(extraction)
                suite["input_extraction_plan_sha256"] = sha256_file(extraction)
            suite_path = _write_json(temporary / "suites" / (suite_id + ".json"), suite)
            suite_final = destination / "suites" / (suite_id + ".json")
            evaluation_rel = "select_runs/%s/%s/evaluation.json" % (suite_id, candidate["method_id"])
            ranking = [candidate["config"]["steps"], candidate["config"]["lr"], candidate["candidate_id"]]
            policy_rows.append({"candidate_id": candidate["candidate_id"], "family": candidate["family"],
                                "method_id": candidate["method_id"], "suite_id": suite_id,
                                "suite_plan_ref": str(suite_final),
                                "suite_plan_sha256": sha256_file(suite_path),
                                "candidate_config_sha256": content_hash(suite["methods"][0]),
                                "evaluation_ref": evaluation_rel, "coverage_min": 1.0,
                                "fallback_rate_max": item.get("fallback_rate_max", 0.01),
                                "rank_order": ranking, "claim": candidate["claim"]})
            tsv.append("\t".join((suite_id, candidate["method_id"], str(suite_final))))
        policy = {"schema_version": "0.2.0", "status": "PROPOSED",
                  "baseline_id": bundle["baseline_id"],
                  "artifact_bundle_id": resource_meta["artifact_bundle_id"],
                  "selection_role": "select", "primary_metric": "eer_raw_ratio",
                  "tie_break": ["fewer_K", "smaller_lr", "candidate_id"],
                  "required_ep_families": ["ep_tta"], "candidates": policy_rows}
        _write_json(temporary / "selection-policy.json", policy)
        (temporary / "select_jobs.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    return _review(destination, "select")


def target(config_ref, selection_ref, output):
    cfg, selection_value = _config(config_ref), read_json(selection_ref)
    if selection_value.get("status") != "SELECTED":
        raise ContractError("target planning requires a completed source-only selection")
    target_cfg = cfg.get("target") or {}
    targets = target_cfg.get("snapshots")
    if type(targets) is not list or not targets:
        raise ContractError("BLOCKED_RESOURCE: experiment target.snapshots is empty")
    destination = Path(output).resolve()
    with AtomicDirectory(destination) as temporary:
        frozen_targets = []
        for item in targets:
            if item.get("source_group_overlap") not in (False, []):
                raise ContractError("BLOCKED_CONTRACT: target has unresolved source-group overlap")
            snapshot = _existing(item["snapshot_ref"], item.get("snapshot_sha256"), item["target_id"])
            role = item["role"]
            manifest_dir = temporary / "manifests" / item["target_id"]
            meta = publish_label_free_manifest(snapshot, role, manifest_dir, status="PROPOSED")
            manifest = destination / "manifests" / item["target_id"] / meta["manifest_ref"]
            scope = {key: item.get(key) for key in ("dataset_id", "release", "subset", "label_policy")}
            if any(value in (None, "") for value in scope.values()):
                raise ContractError("BLOCKED_CONTRACT: target scope lacks release/subset/label_policy")
            scope_id = "%s/%s/%s/label-%s" % (
                scope["dataset_id"], scope["release"], scope["subset"],
                content_hash(scope["label_policy"])[:16])
            extraction = _extraction_plan(target_cfg, role, manifest, meta["manifest_sha256"],
                                          Path(target_cfg["cache_root"]) / item["target_id"])
            _write_json(temporary / "extract" / (item["target_id"] + ".json"), extraction)
            extraction_ref = destination / "extract" / (item["target_id"] + ".json")
            frozen_targets.append({"target_id": item["target_id"], "role": role, "scope_id": scope_id,
                                   "dataset_id": item["dataset_id"], "release": item["release"],
                                   "subset": item["subset"], "label_policy": item["label_policy"],
                                   "labels_ref": item["labels_ref"],
                                   "extraction_plan_ref": str(extraction_ref),
                                   "feature_cache_ref": str(Path(target_cfg["cache_root"]) / item["target_id"]),
                                   "input_manifest_ref": str(manifest),
                                   "input_manifest_sha256": meta["manifest_sha256"]})
        freeze = {"schema_version": "0.2.0", "status": "PROPOSED",
                  "selection_policy_sha256": selection_value["policy_sha256"],
                  "selection_ref": str(Path(selection_ref).resolve()),
                  "selection_sha256": sha256_file(selection_ref),
                  "frozen_bundle_ref": target_cfg["frozen_bundle_ref"],
                  "resources_ref": target_cfg["resources_ref"],
                  "fallback_rate_max": target_cfg.get("fallback_rate_max", 0.01),
                  "targets": frozen_targets, "target_scores_read": False}
        _write_json(temporary / "freeze-plan.json", freeze)
    return _review(destination, "target")


def _replace(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, old, new) for key, item in value.items()}
    return value


def _refresh_ref_hashes(value):
    if isinstance(value, list):
        return [_refresh_ref_hashes(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = {key: _refresh_ref_hashes(item) for key, item in value.items()}
    for key, reference in list(value.items()):
        if not key.endswith("_ref") or not isinstance(reference, str):
            continue
        hash_key = key[:-4] + "_sha256"
        path = Path(reference)
        if hash_key in value and path.is_file():
            value[hash_key] = sha256_file(path)
    return value


def lock(manifest_ref, expected_sha256, reviewer):
    manifest = Path(manifest_ref).resolve()
    if sha256_file(manifest) != expected_sha256:
        raise ContractError("review.json changed; regenerate and review the proposal")
    review = read_json(manifest)
    if review.get("status") != "PROPOSED" or not reviewer.strip():
        raise ContractError("only a PROPOSED review may be locked by a named reviewer")
    if review.get("content_sha256") is not None and review["content_sha256"] != content_hash(review["files"]):
        raise ContractError("review file table digest is invalid")
    source_root, locked = manifest.parent, manifest.parent / "locked"
    if locked.exists():
        raise ContractError("locked stage exists; overwrite is forbidden")
    locked.mkdir()
    for relative, expected in review["files"].items():
        source_path = source_root / relative
        if sha256_file(source_path) != expected:
            raise DataError("proposal file changed after review: %s" % relative)
        target_path = locked / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    old, new = str(source_root), str(locked)
    for path in list(locked.rglob("*.tsv")) + list(locked.rglob("*.txt")):
        text = path.read_text(encoding="utf-8").replace(old, new)
        path.write_text(text, encoding="utf-8")
    for path in sorted(locked.rglob("*.json")):
        value = _replace(read_json(path), old, new)
        if isinstance(value, dict) and value.get("status") == "PROPOSED":
            value["status"] = "LOCKED"
            if "immutable" in value:
                value["immutable"] = True
        path.unlink()
        write_json_new(path, value)
    # Leaf plans change when PROPOSED becomes LOCKED.  Recompute all internal
    # reference hashes after rebinding, deepest paths first, until stable.
    json_paths = sorted(locked.rglob("*.json"), key=lambda item: len(item.parts), reverse=True)
    for _pass in range(3):
        for path in json_paths:
            value = _refresh_ref_hashes(read_json(path))
            path.unlink()
            write_json_new(path, value)
    locked_files = {str(path.relative_to(locked)): sha256_file(path)
                    for path in sorted(locked.rglob("*")) if path.is_file()}
    lock_value = {"schema_version": "0.2.0", "status": "LOCKED", "stage": review["stage"],
                  "proposal_review_ref": str(manifest), "proposal_review_sha256": expected_sha256,
                  "reviewer": reviewer,
                  "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "files": locked_files}
    write_json_new(locked / "lock.json", lock_value)
    return lock_value


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("source", "select"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--out", required=True)
    target_parser = sub.add_parser("target")
    target_parser.add_argument("--config", required=True)
    target_parser.add_argument("--selection", required=True)
    target_parser.add_argument("--out", required=True)
    lock_parser = sub.add_parser("lock")
    lock_parser.add_argument("--manifest", required=True)
    lock_parser.add_argument("--sha256", required=True)
    lock_parser.add_argument("--reviewer", required=True)
    args = parser.parse_args(argv)
    if args.command == "source":
        result = source(args.config, args.out)
    elif args.command == "select":
        result = select(args.config, args.out)
    elif args.command == "target":
        result = target(args.config, args.selection, args.out)
    else:
        result = lock(args.manifest, args.sha256, args.reviewer)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
