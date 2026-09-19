"""Small research workflow over the existing EP-TTA numerical implementation.

This module owns orchestration and ordinary file formats only.  Feature extraction,
source-resource construction, model training, episodic adaptation, and metrics stay
in their original modules.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from eptta.config.schema import read_document
from eptta.data.io import AtomicDirectory, iter_jsonl, write_json_new
from eptta.data.permissions import safe_relative
from eptta.data.roles import ROLES, SOURCE_TRAIN_ROLES
from eptta.data.splits import stable_group_roles
from eptta.errors import ContractError, DataError, EPTTAError, ResourceError


COMMANDS = ("prepare-data", "train-source", "prepare-source", "run-tta", "evaluate", "report")
COMMON = {"schema_version", "command", "paths"}
TOP_LEVEL = {
    "prepare-data": COMMON | {"dataset", "columns", "label_map", "split", "output"},
    "train-source": COMMON | {"recipe", "phase", "output", "resume", "stop_after_epoch"},
    "prepare-source": COMMON | {"frozen_bundle", "training_run", "frozen_output",
                                "gpu_id", "roles", "artifact_plan", "fit_cache",
                                "resources_output", "selection", "output"},
    "run-tta": COMMON | {"run_name", "output_root", "role", "scope_id", "input_manifest",
                          "feature_cache", "resources", "frozen_bundle", "method", "threshold",
                          "threshold_source", "fallback_rate_max", "selection_source", "seed",
                          "extraction"},
}


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        import re
        def replace(match):
            name = match.group(1)
            if not os.environ.get(name):
                raise EPTTAError("unresolved environment variable: %s" % name)
            return os.environ[name]
        result = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
        if "${" in result:
            raise EPTTAError("unresolved environment variable in configuration")
        return result
    return value


def _deep_merge(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _bind_local_paths(value, paths):
    if isinstance(value, dict):
        return {key: _bind_local_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind_local_paths(item, paths) for item in value]
    if isinstance(value, str) and value.startswith("@path:"):
        key = value[6:]
        if key not in paths or not isinstance(paths[key], str) or not paths[key]:
            raise ContractError("local paths config does not define %s" % key)
        return paths[key]
    return value


def load_config(path, command, paths_ref=None):
    config = read_document(path)
    if not isinstance(config, dict):
        raise ContractError("experiment config must be a mapping")
    if config.get("schema_version") != "0.1.0" or config.get("command") != command:
        raise ContractError("config must declare schema_version 0.1.0 and command %s" % command)
    unknown = set(config) - TOP_LEVEL[command]
    if unknown:
        raise ContractError("unknown top-level config fields: %s" % sorted(unknown))
    path_bindings = config.get("paths", {})
    if not isinstance(path_bindings, dict):
        raise ContractError("config paths must be a mapping")
    if paths_ref:
        overlay = read_document(paths_ref)
        if not isinstance(overlay, dict) or set(overlay) - {"schema_version", "paths"}:
            raise ContractError("local paths config may contain only schema_version and paths")
        if overlay.get("schema_version") != "0.1.0" or not isinstance(overlay.get("paths"), dict):
            raise ContractError("invalid local paths config")
        path_bindings = _deep_merge(path_bindings, overlay["paths"])
    config = _bind_local_paths(config, path_bindings)
    # Bind locations, but never persist the entire private path map in a run's
    # resolved configuration or metadata.
    config.pop("paths", None)
    return _expand(config)


def _require_fields(value, required, optional=(), name="object"):
    if not isinstance(value, dict):
        raise ContractError("%s must be a mapping" % name)
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing or unknown:
        raise ContractError("%s fields invalid; missing=%s unknown=%s" %
                            (name, sorted(missing), sorted(unknown)))


def _require_resolved_text(value, name):
    if not isinstance(value, str) or not value.strip() or "REPLACE_WITH_" in value:
        raise ContractError("%s must be explicitly filled; replace its REPLACE_WITH_ placeholder" % name)
    return value


def _rows(path):
    source = Path(path)
    if not source.is_file():
        raise ResourceError("manifest does not exist: %s" % source)
    if source.suffix.lower() in (".jsonl", ".json"):
        yield from iter_jsonl(source)
        return
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise DataError("CSV manifest has no header")
        for row in reader:
            yield dict(row)


def _write_csv(path, fieldnames, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _git_metadata():
    root = Path(__file__).resolve().parents[2]
    def run(*args):
        completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status), "status": status or ""}, root


def _new_run_id(prefix):
    from datetime import datetime, timezone
    return "%s-%s-%s" % (prefix, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                          secrets.token_hex(3))


def write_run_metadata(run, config, *, cache_ref=None, checkpoint_ref=None, data_ref=None):
    run = Path(run)
    git, root = _git_metadata()
    versions = {}
    for distribution in ("torch", "torchaudio", "numpy", "scipy", "PyYAML", "fairseq",
                         "librosa", "soundfile", "tensorboardX"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    meta = {"schema_version": "0.2.0", "run_id": config.get("run_id") or _new_run_id("run"),
            "code": git, "seed": config.get("seed"),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": versions},
            "checkpoint_ref": checkpoint_ref, "cache_ref": cache_ref,
            "data_ref": data_ref, "method": config.get("method"),
            "selection_source": config.get("selection_source"),
            "code_snapshot_ref": "code_snapshot" if git["dirty"] else None}
    write_json_new(run / "meta.json", meta)
    write_json_new(run / "config.yaml", config)
    if git["dirty"]:
        completed = subprocess.run(["git", "diff", "--binary", "--", ".", ":(exclude)configs/*.private.*"],
                                   cwd=root, capture_output=True, check=False)
        if completed.returncode == 0:
            (run / "code.diff").write_bytes(completed.stdout)
        snapshot = run / "code_snapshot"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        shutil.copytree(root / "src" / "eptta", snapshot / "src" / "eptta", ignore=ignore)
        shutil.copytree(root / "workers", snapshot / "workers", ignore=ignore)
        for name in ("pyproject.toml", "environment.yml"):
            if (root / name).is_file():
                shutil.copy2(root / name, snapshot / name)
    return meta


def prepare_data(config):
    dataset = config.get("dataset")
    columns = config.get("columns")
    split = config.get("split")
    _require_fields(dataset, {"dataset_id", "release", "subset", "manifest"},
                    {"audio_root", "root_key"}, "dataset")
    _require_fields(columns, {"sample_id", "audio_relpath", "label", "group_id"},
                    {"split_role", "sample_index"}, "columns")
    _require_fields(split, {"reuse_existing", "seed"}, {"ratios", "assignments"}, "split")
    if type(split["reuse_existing"]) is not bool or type(split["seed"]) is not int:
        raise ContractError("split.reuse_existing must be boolean and split.seed must be an integer")
    for field in ("dataset_id", "release", "subset", "manifest"):
        _require_resolved_text(dataset[field], "dataset.%s" % field)
    label_map = config.get("label_map")
    if not isinstance(label_map, dict) or not label_map:
        raise ContractError("label_map must explicitly map raw values to 0=bonafide/1=spoof")
    if any(type(value) is not int or value not in (0, 1) for value in label_map.values()):
        raise ContractError("label_map values must be canonical integers 0 or 1")
    ratios = split.get("ratios")
    source_rows = list(_rows(dataset["manifest"]))
    if not source_rows:
        raise DataError("input manifest is empty")
    prior_assignments = {}
    if split.get("assignments"):
        if split["reuse_existing"] is not True:
            raise ContractError("split.assignments requires reuse_existing=true")
        for number, assignment in enumerate(_rows(split["assignments"]), 2):
            sample_id = str(assignment.get("sample_id", "")).strip()
            role = str(assignment.get("split_role", "")).strip()
            group_id = str(assignment.get("group_id", assignment.get("source_group_id", ""))).strip()
            try:
                raw_assignment_index = assignment["sample_index"]
                if isinstance(raw_assignment_index, bool):
                    raise ValueError
                sample_index = int(raw_assignment_index)
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError("assignment row %d requires a persistent integer sample_index" % number) from exc
            if (not sample_id or sample_id in prior_assignments or role not in ROLES or
                    role in ("unassigned", "quarantine") or not group_id or sample_index < 0):
                raise DataError("invalid or duplicate assignment row %d" % number)
            prior_assignments[sample_id] = {"split_role": role, "group_id": group_id,
                                             "sample_index": sample_index}
        indices = [item["sample_index"] for item in prior_assignments.values()]
        if len(indices) != len(set(indices)):
            raise DataError("assignment sample_index values must be unique")
    output_rows, seen, group_roles = [], set(), {}
    missing = Counter()
    root = Path(dataset["audio_root"]).resolve() if dataset.get("audio_root") else None
    for number, raw in enumerate(source_rows, 2):
        def field(name):
            key = columns.get(name)
            value = raw.get(key) if key else None
            if value is None or str(value).strip() == "":
                missing[name] += 1
                raise DataError("manifest row %d lacks required %s" % (number, name))
            return str(value).strip()
        sample_id, relpath, group_id = field("sample_id"), field("audio_relpath"), field("group_id")
        if sample_id in seen:
            raise DataError("duplicate sample_id: %s" % sample_id)
        seen.add(sample_id)
        safe_relative(relpath)
        raw_label = field("label")
        if raw_label not in label_map:
            raise DataError("unmapped label at row %d: %r" % (number, raw_label))
        role_key = columns.get("split_role")
        role = str(raw.get(role_key, "")).strip() if role_key else ""
        prior = prior_assignments.get(sample_id)
        if prior:
            if prior["group_id"] != group_id or role and prior["split_role"] != role:
                raise DataError("manifest conflicts with saved assignment for %s" % sample_id)
            role = prior["split_role"]
        if role:
            if role not in ROLES or role in ("unassigned", "quarantine"):
                raise DataError("invalid existing split_role at row %d: %s" % (number, role))
            if split["reuse_existing"] is not True:
                raise ContractError("existing assignments require split.reuse_existing=true")
        previous = group_roles.get(group_id)
        if role and previous and previous != role:
            raise DataError("group crosses split roles: %s" % group_id)
        if role:
            group_roles[group_id] = role
        if root:
            audio = (root / relpath).resolve()
            if not audio.is_relative_to(root) or not audio.is_file():
                raise DataError("audio path missing or outside root: %s" % relpath)
        index_key = columns.get("sample_index")
        raw_index = raw.get(index_key) if index_key else None
        if raw_index not in (None, ""):
            try:
                if isinstance(raw_index, bool):
                    raise ValueError
                sample_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise DataError("manifest row %d has invalid sample_index" % number) from exc
            if sample_index < 0 or prior and sample_index != prior["sample_index"]:
                raise DataError("manifest sample_index conflicts with saved assignment")
        else:
            sample_index = prior["sample_index"] if prior else None
        output_rows.append({"sample_id": sample_id, "audio_relpath": relpath,
                            "label": label_map[raw_label], "split_role": role or None,
                            "sample_index": sample_index,
                            "group_id": group_id, "dataset_id": dataset["dataset_id"],
                            "release": dataset["release"], "subset": dataset["subset"],
                            "root_key": dataset.get("root_key", dataset["dataset_id"])})
    unassigned_groups = sorted({row["group_id"] for row in output_rows if row["split_role"] is None and
                                row["group_id"] not in group_roles})
    if unassigned_groups:
        if not ratios:
            raise ContractError("rows without assignments require explicit split.ratios")
        group_roles.update(stable_group_roles(split["seed"], unassigned_groups, ratios))
    extra_assignments = set(prior_assignments) - seen
    if extra_assignments:
        raise DataError("assignment file contains IDs absent from the manifest")
    used_indices = [row["sample_index"] for row in output_rows if row["sample_index"] is not None]
    if len(used_indices) != len(set(used_indices)):
        raise DataError("manifest sample_index values must be unique")
    next_index = max(used_indices, default=-1) + 1
    for row in sorted((row for row in output_rows if row["sample_index"] is None),
                      key=lambda item: item["sample_id"]):
        row["sample_index"] = next_index
        next_index += 1
    for row in output_rows:
        role = group_roles[row["group_id"]]
        if row["split_role"] is not None and row["split_role"] != role:
            raise DataError("group crosses split roles: %s" % row["group_id"])
        row["split_role"] = role
    destination = Path(config["output"])
    with AtomicDirectory(destination) as temporary:
        fields = list(output_rows[0])
        _write_csv(temporary / "manifest.csv", fields, output_rows)
        roles = Counter(row["split_role"] for row in output_rows)
        classes = Counter(row["label"] for row in output_rows)
        role_files = {}
        (temporary / "manifests").mkdir()
        (temporary / "inference").mkdir()
        (temporary / "labels").mkdir()
        (temporary / "groups").mkdir()
        for role in sorted(roles):
            selected = [row for row in output_rows if row["split_role"] == role]
            inference = [{"schema_version": "0.1.0", "sample_id": row["sample_id"],
                          "sample_index": row["sample_index"],
                          "root_key": row["root_key"], "audio_relpath": row["audio_relpath"],
                          "split_role": role} for row in selected]
            training = [dict(item) for item in inference]
            if role in SOURCE_TRAIN_ROLES:
                for item, row in zip(training, selected):
                    item["canonical_label"] = row["label"]
            labels = [{"schema_version": "0.1.0", "sample_id": row["sample_id"],
                       "canonical_label": row["label"]} for row in selected]
            groups = [{"schema_version": "0.1.0", "sample_id": row["sample_id"],
                       "source_group_id": row["group_id"]} for row in selected]
            manifest_path = temporary / "manifests" / (role + ".jsonl")
            inference_path = temporary / "inference" / (role + ".jsonl")
            label_path = temporary / "labels" / (role + ".jsonl")
            group_path = temporary / "groups" / (role + ".jsonl")
            _write_jsonl(manifest_path, training)
            _write_jsonl(inference_path, inference)
            _write_jsonl(label_path, labels)
            _write_jsonl(group_path, groups)
            role_files[role] = {"manifest_ref": "manifests/" + manifest_path.name,
                                "inference_ref": "inference/" + inference_path.name,
                                "labels_ref": "labels/" + label_path.name,
                                "groups_ref": "groups/" + group_path.name, "count": len(selected)}
        assignments = [{"sample_id": row["sample_id"], "sample_index": row["sample_index"],
                        "group_id": row["group_id"], "split_role": row["split_role"]}
                       for row in sorted(output_rows, key=lambda item: item["sample_index"])]
        _write_csv(temporary / "assignments.csv", list(assignments[0]), assignments)
        summary = {"schema_version": "0.2.0", "status": "READY", "dataset_id": dataset["dataset_id"],
                   "release": dataset["release"], "subset": dataset["subset"],
                   "sample_count": len(output_rows), "class_counts": dict(sorted(classes.items())),
                   "role_counts": dict(sorted(roles.items())), "missing": dict(missing),
                   "conflicts": 0, "split_seed": split["seed"],
                   "assignment_method": "reuse_existing_or_sorted_group_prng_v1",
                   "assignments_ref": "assignments.csv", "roles": role_files}
        write_json_new(temporary / "config.yaml", config)
        write_json_new(temporary / "summary.json", summary)
    return summary


def train_source(config):
    from eptta.training.dispatch import compile_source_job, launch_resume_job, launch_source_job
    _require_fields(config, {"schema_version", "command", "recipe", "phase", "output"},
                    {"resume", "paths", "stop_after_epoch"}, "train-source config")
    if config["phase"] not in ("smoke", "full"):
        raise ContractError("training phase must be smoke or full")
    if config.get("resume"):
        result = launch_resume_job(config["output"], config["resume"])
    else:
        job = compile_source_job(config["recipe"], config["phase"], config["output"])
        result = launch_source_job(job, config.get("stop_after_epoch"))
    return result


def _finish_prepare_source(config, bundle, caches, resources_path, selection_result):
    result = {"schema_version": "0.2.0", "status": "READY",
              "frozen_bundle": config["frozen_bundle"],
              "source_run_id": bundle["source_run_id"],
              "checkpoint_ref": bundle["checkpoint_ref"], "caches": caches,
              "resources": resources_path, "selection": selection_result}
    if config.get("output"):
        output = Path(config["output"])
        if output.exists():
            raise ContractError("prepare-source run output exists; choose a new run directory")
        with AtomicDirectory(output) as temporary:
            write_run_metadata(temporary, config, cache_ref=caches,
                               checkpoint_ref=bundle["checkpoint_ref"],
                               data_ref={role: item["manifest"]
                                         for role, item in config.get("roles", {}).items()})
            write_json_new(temporary / "result.json", result)
            (temporary / "log.txt").write_text("prepare-source completed\n", encoding="utf-8")
        result["run_output"] = str(output)
    return result


def prepare_source(config):
    from eptta.cache.merge import merge_feature_caches
    from eptta.execution.extract import compile_inference_job, launch_extraction
    from eptta.models.frozen import verify_frozen_export
    from eptta.offline.artifacts import build_source_resources, load_frozen_resources
    config = dict(config)
    if not config.get("frozen_bundle"):
        if not config.get("training_run") or not config.get("frozen_output"):
            raise ContractError("prepare-source requires frozen_bundle or training_run + frozen_output")
        from eptta.training.artifacts import finalize_training
        from eptta.training.r4 import launch_r4_export
        finalized = Path(config["training_run"]) / "finalized.json"
        if not finalized.exists():
            finalize_training(config["training_run"], finalized)
        launch_r4_export(finalized, config["frozen_output"], gpu_id=config.get("gpu_id", 0))
        config["frozen_bundle"] = str(Path(config["frozen_output"]) / "bundle.json")
    bundle, _manifest, _parity, _selection = verify_frozen_export(config["frozen_bundle"])
    caches = {}
    for role, item in (config.get("roles") or {}).items():
        _require_fields(item, {"manifest", "dataset_id", "purpose", "data_roots", "probe",
                               "numerical_mode", "cache"},
                        {"workers", "reuse", "cache_id"}, "prepare-source role %s" % role)
        cache_path = Path(item["cache"])
        if cache_path.exists():
            if item.get("reuse") is not True:
                raise ContractError("existing cache requires reuse=true: %s" % cache_path)
            from eptta.cache.reader import FeatureCache
            cache = FeatureCache(cache_path)
            identity = cache.index.get("identity", {})
            if cache.index.get("format") == "sharded_npy_v2" and (
                    Path(identity.get("manifest_ref", "")).resolve() != Path(item["manifest"]).resolve() or
                    identity.get("dataset_id") != item["dataset_id"] or identity.get("split_role") != role or
                    identity.get("source_run_id") != bundle["source_run_id"] or
                    Path(identity.get("checkpoint_ref", "")).resolve() != Path(bundle["checkpoint_ref"]).resolve() or
                    identity.get("preprocess") != bundle["preprocess"] or
                    identity.get("views") != item["probe"] or
                    identity.get("seed") != item["probe"]["seed"] or
                    identity.get("dtype") != item["numerical_mode"]["dtype"] or
                    identity.get("numerical_mode") != item["numerical_mode"]):
                raise DataError("existing cache declared provenance differs for %s" % role)
            if identity.get("baseline_id") and identity["baseline_id"] != bundle["baseline_id"]:
                raise DataError("historical cache does not name this frozen detector")
            cache.verify_expected_ids([row["sample_id"] for row in iter_jsonl(item["manifest"])])
            caches[role] = str(cache_path)
            continue
        workers = item.get("workers", 1)
        if type(workers) is not int or workers < 1:
            raise ContractError("workers must be a positive integer")
        plan = {"schema_version": "0.3.0", "status": "READY", "purpose": item["purpose"],
                "input_role": role, "frozen_bundle_ref": config["frozen_bundle"],
                "manifest_ref": item["manifest"], "dataset_id": item["dataset_id"],
                "data_roots": item["data_roots"], "probe": item["probe"],
                "numerical_mode": item["numerical_mode"],
                "cache_id": item.get("cache_id") or _new_run_id("cache"),
                "output_root": str(cache_path) + ".shards"}
        with tempfile.TemporaryDirectory(prefix="eptta-source-plan-") as directory:
            plan_path = Path(directory) / "plan.json"
            write_json_new(plan_path, plan)
            shards = []
            for slot in range(workers):
                job = compile_inference_job(plan_path, slot, workers)
                launch_extraction(job)
                shards.append(job["output_dir"])
        ids = [row["sample_id"] for row in iter_jsonl(item["manifest"])]
        merge_feature_caches(shards, cache_path, ids)
        caches[role] = str(cache_path)
    resources_path = config.get("resources_output")
    if resources_path:
        if Path(resources_path).exists():
            load_frozen_resources(resources_path, bundle)
        else:
            if not config.get("artifact_plan") or not config.get("fit_cache"):
                raise ContractError("resource construction requires artifact_plan and fit_cache")
            artifact = read_document(config["artifact_plan"])
            scientific = {"rank", "alpha_cal", "anchor_per_class", "seed", "random_seeds",
                          "treatment_families", "samples_per_group", "pair_seed", "margin_bins",
                          "margin_epsilon", "minimum_cal0_bonafide", "fixed_adapter",
                          "fit_labels_ref", "fit_groups_ref", "calibration_labels_ref"}
            if set(artifact) != scientific:
                raise ContractError("source resource parameters have missing/unknown fields: %s" %
                                    sorted(set(artifact) ^ scientific))
            if "fit" not in config.get("roles", {}) or "cal0" not in config.get("roles", {}):
                raise ContractError("source resource construction requires fit and cal0 roles")
            full = {"schema_version": "0.3.0", "status": "READY",
                    "frozen_bundle_ref": config["frozen_bundle"], "fit_role": "fit",
                    "fit_manifest_ref": config["roles"]["fit"]["manifest"],
                    "fit_labels_ref": artifact["fit_labels_ref"],
                    "fit_groups_ref": artifact["fit_groups_ref"], "calibration_role": "cal0",
                    "calibration_manifest_ref": config["roles"]["cal0"]["manifest"],
                    "calibration_cache_ref": caches.get("cal0", config["roles"]["cal0"]["cache"]),
                    "calibration_labels_ref": artifact["calibration_labels_ref"]}
            full.update({key: artifact[key] for key in scientific -
                         {"fit_labels_ref", "fit_groups_ref", "calibration_labels_ref"}})
            with tempfile.TemporaryDirectory(prefix="eptta-artifact-plan-") as directory:
                plan_path = Path(directory) / "plan.json"
                write_json_new(plan_path, full)
                build_source_resources(plan_path, config["fit_cache"], resources_path)
    selection_result = config.get("selection")
    if selection_result:
        if not resources_path:
            raise ContractError("source selection requires an explicit source resource bundle")
        selection = read_document(selection_result)
        if "candidates" in selection:
            required = {"schema_version", "selected_on_role", "candidates", "output"}
            optional = {"source_select"}
            if (set(selection) - required - optional or required - set(selection) or
                    selection["selected_on_role"] != "select"):
                raise ContractError("source selection plan has invalid fields/role")
            selection_output = Path(selection["output"])
            if selection_output.exists():
                existing = read_document(selection_output)
                if existing.get("selected_on_role") != "select" or not existing.get("methods"):
                    raise DataError("existing source selection is malformed")
                return _finish_prepare_source(config, bundle, caches, resources_path,
                                              str(selection_output))
            chosen = {}
            for index, candidate in enumerate(selection["candidates"]):
                _require_fields(candidate, {"family", "method"}, {"rank_order", "run"},
                                "source selection candidate")
                run_ref = candidate.get("run")
                if not run_ref:
                    source_select = selection.get("source_select")
                    _require_fields(source_select, {"manifest", "cache", "labels", "scope_id", "output_root",
                                                    "threshold", "threshold_source", "seed"},
                                    {"fallback_rate_max"}, "source_select")
                    name = "select-%s-%02d" % (candidate["family"], index)
                    run_dir = Path(source_select["output_root"]) / name
                    if not run_dir.exists():
                        run_tta({"schema_version": "0.1.0", "command": "run-tta", "run_name": name,
                                 "output_root": source_select["output_root"], "role": "select",
                                 "scope_id": source_select["scope_id"],
                                 "input_manifest": source_select["manifest"],
                                 "feature_cache": source_select["cache"], "resources": resources_path,
                                 "frozen_bundle": config["frozen_bundle"], "method": candidate["method"],
                                 "threshold": source_select["threshold"],
                                 "threshold_source": source_select["threshold_source"],
                                 "fallback_rate_max": source_select.get("fallback_rate_max", 0.01),
                                 "selection_source": None, "seed": source_select["seed"]})
                    if not (run_dir / "metrics.json").exists():
                        evaluate(run_dir, source_select["labels"])
                    run_ref = str(run_dir)
                run_dir = Path(run_ref)
                run_config = read_document(run_dir / "config.yaml")
                metrics_doc = read_document(run_dir / "metrics.json")
                if run_config.get("role") != "select" or run_config.get("method") != candidate["method"]:
                    raise ContractError("selection candidate is not the declared source-select run")
                eer = metrics_doc.get("metrics", {}).get("eer")
                if type(eer) not in (int, float):
                    raise DataError("selection candidate lacks numeric EER")
                row = {**candidate, "eer": float(eer), "metrics_ref": str((run_dir / "metrics.json").resolve())}
                key = (row["eer"], tuple(row.get("rank_order", [])), index)
                if candidate["family"] not in chosen or key < chosen[candidate["family"]][0]:
                    chosen[candidate["family"]] = (key, row)
            methods = [value[1]["method"] for _, value in sorted(chosen.items())]
            if not methods:
                raise ContractError("source selection has no candidates")
            selected = {"schema_version": "0.2.0", "status": "SELECTED",
                        "selected_on_role": "select", "methods": methods,
                        "selection_plan_ref": str(Path(selection_result).resolve()),
                        "families": {key: value[1] for key, value in sorted(chosen.items())}}
            write_json_new(selection["output"], selected)
            selection_result = selection["output"]
        elif (selection.get("selected_on_role") != "select" or
              not isinstance(selection.get("methods"), list) or not selection["methods"]):
            raise ContractError("selection must record nonempty methods selected_on_role=select")
    return _finish_prepare_source(config, bundle, caches, resources_path, selection_result)


def run_tta(config):
    from eptta.execution.suite import run_suite
    config = dict(config)
    if config.get("threshold") is None:
        resource_path = Path(config["resources"])
        metadata_path = resource_path / "resources.json" if resource_path.is_dir() else resource_path
        resources_meta = read_document(metadata_path)
        threshold = (resources_meta.get("scalars") or {}).get("tau0")
        if type(threshold) not in (int, float) or isinstance(threshold, bool):
            raise ContractError("source resources do not contain a saved cal0 tau0")
        config["threshold"] = float(threshold)
        config["threshold_source"] = str(metadata_path.resolve()) + "#scalars.tau0"
    import math
    if (type(config.get("threshold")) not in (int, float) or isinstance(config.get("threshold"), bool) or
            not math.isfinite(config["threshold"])):
        raise ContractError("run-tta threshold must be a finite source/cal0 value")
    _require_resolved_text(config.get("threshold_source"), "threshold_source")
    if type(config.get("seed")) is not int:
        raise ContractError("run-tta seed must be an integer")
    role = config.get("role")
    if role not in ("select", "audit", "cal1", "control_test", "target_test"):
        raise ContractError("run-tta role must be select/audit/cal1/control_test/target_test")
    method = config.get("method")
    _require_fields(method, {"method_id", "config", "params"}, (), "method")
    from eptta.baselines.registry import get_method_contract
    method_contract = get_method_contract(method["method_id"])
    if method_contract["route"] != "feature_cache":
        raise ContractError("NOT_IMPLEMENTED: %s requires a waveform/model-update path and cannot use "
                            "a frozen feature cache" % method["method_id"])
    _require_resolved_text(config.get("scope_id"), "scope_id")
    def contains_label_key(value):
        if isinstance(value, dict):
            return any("label" in str(key).lower() or contains_label_key(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(contains_label_key(item) for item in value)
        return False
    if contains_label_key(config):
        raise ContractError("run-tta config must not contain target label references")
    cache_path = Path(config["feature_cache"])
    extraction = config.get("extraction")
    if not cache_path.exists():
        _require_fields(extraction, {"dataset_id", "data_roots", "probe", "numerical_mode"},
                        {"workers", "cache_id"}, "run-tta extraction")
        from eptta.cache.merge import merge_feature_caches
        from eptta.execution.extract import compile_inference_job, launch_extraction
        workers = extraction.get("workers", 1)
        if type(workers) is not int or workers < 1:
            raise ContractError("extraction workers must be a positive integer")
        purpose = "select" if role == "select" else "audit" if role in ("audit", "cal1") else "confirmatory"
        plan = {"schema_version": "0.3.0", "status": "READY", "purpose": purpose,
                "input_role": role, "frozen_bundle_ref": config["frozen_bundle"],
                "manifest_ref": config["input_manifest"], "dataset_id": extraction["dataset_id"],
                "data_roots": extraction["data_roots"], "probe": extraction["probe"],
                "numerical_mode": extraction["numerical_mode"],
                "cache_id": extraction.get("cache_id") or _new_run_id("cache"),
                "output_root": str(cache_path) + ".shards"}
        with tempfile.TemporaryDirectory(prefix="eptta-target-plan-") as directory:
            plan_path = Path(directory) / "plan.json"
            write_json_new(plan_path, plan)
            shards = []
            for slot in range(workers):
                job = compile_inference_job(plan_path, slot, workers)
                launch_extraction(job)
                shards.append(job["output_dir"])
        ids = [row["sample_id"] for row in iter_jsonl(config["input_manifest"])]
        merge_feature_caches(shards, cache_path, ids)
    elif extraction:
        _require_fields(extraction, {"dataset_id", "data_roots", "probe", "numerical_mode"},
                        {"workers", "cache_id"}, "run-tta extraction")
        from eptta.cache.reader import FeatureCache
        cache = FeatureCache(cache_path)
        identity = cache.index.get("identity", {})
        if cache.index.get("format") == "sharded_npy_v2" and (
                identity.get("dataset_id") != extraction["dataset_id"] or
                identity.get("split_role") != role or
                Path(identity.get("manifest_ref", "")).resolve() != Path(config["input_manifest"]).resolve() or
                identity.get("views") != extraction["probe"] or
                identity.get("seed") != extraction["probe"]["seed"] or
                identity.get("dtype") != extraction["numerical_mode"]["dtype"] or
                identity.get("numerical_mode") != extraction["numerical_mode"]):
            raise DataError("existing target cache differs from the declared extraction settings")
    if role in ("control_test", "target_test"):
        selection_ref = config.get("selection_source")
        if not selection_ref or not Path(selection_ref).is_file():
            raise ContractError("final test scoring requires an existing source-selection record")
        selection = read_document(selection_ref)
        candidates = selection.get("methods") if selection.get("selected_on_role") == "select" else [
            row.get("method") for row in (selection.get("selected") or {}).values()]
        if not isinstance(candidates, list) or method not in candidates:
            raise ContractError("run method/config is not identical to the saved source-select choice")
    suite_id = config["run_name"]
    plan = {"schema_version": "0.3.0", "status": "READY", "suite_id": suite_id,
            "feature_cache_ref": str(cache_path), "resources_ref": config["resources"],
            "frozen_bundle_ref": config["frozen_bundle"], "input_manifest_ref": config["input_manifest"],
            "input_role": role, "scope_id": config.get("scope_id", suite_id),
            "fallback_rate_max": config.get("fallback_rate_max", 0.01), "methods": [method]}
    output = Path(config["output_root"]) / config["run_name"]
    with tempfile.TemporaryDirectory(prefix="eptta-suite-plan-") as directory:
        plan_path = Path(directory) / "suite.json"
        write_json_new(plan_path, plan)
        phase = "select" if role == "select" else "audit" if role in ("audit", "cal1") else "confirmatory"
        result = run_suite(plan_path, suite_id, phase, output)
    method_dir = output / method["method_id"]
    run_metadata = read_document(method_dir / "run.json")
    for name in ("run.json", "expected_ids.json", "runtime.jsonl", "diagnostics.jsonl", "events.jsonl"):
        source = method_dir / name
        if source.is_file():
            shutil.copy2(source, output / name)
    score_rows = list(iter_jsonl(method_dir / "scores.jsonl"))
    temporary = output / ".scores.csv.tmp"
    _write_csv(temporary, ["sample_id", "score", "score_before", "status"],
               [{key: row.get(key) for key in ("sample_id", "score", "score_before", "status")}
                for row in score_rows])
    os.replace(temporary, output / "scores.csv")
    bundle, _export, _parity, _selection = __import__(
        "eptta.models.frozen", fromlist=["verify_frozen_export"]).verify_frozen_export(config["frozen_bundle"])
    write_run_metadata(output, config, cache_ref=config["feature_cache"],
                       checkpoint_ref=bundle["checkpoint_ref"], data_ref=config["input_manifest"])
    (output / "log.txt").write_text("run-tta completed without reading labels\n", encoding="utf-8")
    return {**result, "output": str(output), "scores": str(output / "scores.csv")}


def _label_rows(path):
    for row in _rows(path):
        label = row.get("canonical_label", row.get("label"))
        try:
            label = int(label)
        except (TypeError, ValueError) as exc:
            raise DataError("evaluation label must be canonical integer 0 or 1") from exc
        if (set(row).intersection({"sample_id"}) != {"sample_id"} or
                not isinstance(row["sample_id"], str) or not row["sample_id"] or label not in (0, 1)):
            raise DataError("evaluation label row is invalid")
        yield row["sample_id"], label


def evaluate(run_ref, labels_ref):
    from eptta.evaluation.metrics import binary_metrics
    run = Path(run_ref)
    config = read_document(run / "config.yaml")
    threshold = config.get("threshold")
    if type(threshold) not in (int, float) or isinstance(threshold, bool):
        raise ContractError("evaluate requires a fixed cal0 threshold saved in the run config")
    if not config.get("threshold_source"):
        raise ContractError("run config must record the source/cal0 threshold origin")
    scores_path = run / "scores.csv"
    if not scores_path.is_file():
        raise DataError("scores.csv is missing")
    scores = {}
    with scores_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in scores:
                raise DataError("score IDs must be unique and nonempty")
            try:
                score = float(row["score"])
                before = float(row["score_before"]) if row.get("score_before") not in (None, "") else None
            except (TypeError, ValueError) as exc:
                raise DataError("score values must be numeric") from exc
            import math
            if not math.isfinite(score) or before is not None and not math.isfinite(before):
                raise DataError("scores must be finite")
            scores[sample_id] = (score, before)
    expected = read_document(run / "expected_ids.json")
    if (not isinstance(expected, list) or
            any(not isinstance(value, str) or not value for value in expected) or
            set(scores) != set(expected) or len(expected) != len(set(expected))):
        raise DataError("score IDs differ from the expected set")
    labels = dict(_label_rows(labels_ref))
    if len(labels) != len(list(_label_rows(labels_ref))):
        raise DataError("evaluation labels contain duplicate IDs")
    if set(scores) != set(labels):
        raise DataError("score and label ID sets differ")
    order = sorted(scores)
    adapted = [scores[key][0] for key in order]
    frozen = [scores[key][1] for key in order]
    if any(value is None for value in frozen):
        frozen = None
    result = {"schema_version": "0.2.0", "status": "EVALUATED", "score_direction": "larger_is_spoof",
              "label_mapping": {"0": "bonafide", "1": "spoof"}, "threshold_source": config["threshold_source"],
              "scores_ref": str(scores_path.resolve()), "labels_ref": str(Path(labels_ref).resolve()),
              "metrics": binary_metrics(adapted, [labels[key] for key in order], threshold, frozen),
              "unavailable_metrics": ["tDCF/minDCF require explicit ASV scores and protocol inputs"]}
    write_json_new(run / "metrics.json", result)
    return result


def report(runs_ref, output):
    root = Path(runs_ref)
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run = metrics_path.parent
        metrics_doc = read_document(metrics_path)
        config = read_document(run / "config.yaml")
        metrics = metrics_doc["metrics"]
        rows.append({"run_name": run.name, "role": config.get("role"),
                     "method_id": (config.get("method") or {}).get("method_id"),
                     "selection_source": config.get("selection_source"), "count": metrics.get("count"),
                     "eer": metrics.get("eer"), "auroc": metrics.get("auroc"),
                     "fpr": metrics.get("fpr"), "fnr": metrics.get("fnr"),
                     "threshold": metrics.get("threshold"), "status": metrics_doc.get("status")})
    destination = Path(output)
    if destination.exists():
        raise EPTTAError("output exists; overwrite is forbidden: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_name", "role", "method_id", "selection_source", "count", "eer", "auroc",
              "fpr", "fnr", "threshold", "status"]
    fd, temporary = tempfile.mkstemp(prefix=".%s." % destination.name, dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {"schema_version": "0.1.0", "status": "REPORTED", "run_count": len(rows),
            "output": str(destination)}
