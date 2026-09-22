#!/usr/bin/env python
"""Fail-closed preflight for the P2 multi-domain confirmatory queue."""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from baselines.locked_config import (PUBLISHED_METHODS, load_locked_document,
                                     load_locked_method)
from baselines.provenance import git_commit, require_clean_tree
from baselines.port_validation import load_validation_evidence
from baselines.splits import get_split, load_split_sample_ids
from baselines.target_waveform import TargetWaveformDataset
from eptta.cache.reader import FeatureCache

CONFIRMATORY_SPLITS = ("itw_target90", "asv2021_la", "asv2021_df", "control_test")


def _load_json(path, description):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("%s is missing or invalid: %s" % (description, path)) from exc


def validate_port_results(path):
    document = _load_json(path, "port validation artifact")
    by_method = {record.get("method_id"): record
                 for record in document.get("validations", {}).values()
                 if isinstance(record, dict)}
    invalid = [method for method in PUBLISHED_METHODS
               if by_method.get(method, {}).get("PORT_VALID") is not True]
    if invalid:
        raise ValueError("published methods are not PORT_VALID: %s" % ", ".join(invalid))


def validate_output_directory(confirmatory_dir):
    path = Path(confirmatory_dir)
    if not path.exists():
        return "NEW"
    status = path / "job_status.json"
    if not status.is_file():
        raise ValueError("confirmatory output exists without resumable job_status.json")
    _load_json(status, "confirmatory resume status")
    return "RESUME"


def validate_validation_run(validation_run, locked_config, current_commit):
    validation_run = Path(validation_run)
    summary = _load_json(validation_run / "validation_summary.json",
                         "validation summary")
    if (summary.get("validation_git_commit") != current_commit
            or summary.get("validation_status") != "PASS"
            or summary.get("published_ports_valid") is not True):
        raise ValueError("validation run does not approve the current git commit")
    ports = _load_json(validation_run / "pilot/port_validation.json",
                       "port validation artifact")
    if ports.get("validation_git_commit") != current_commit:
        raise ValueError("port validation git commit mismatch")
    if ports.get("locked_config_content") != load_locked_document(locked_config):
        raise ValueError("validation run locked config mismatch")
    validate_port_results(validation_run / "pilot/port_validation.json")
    evidence_ref = ports.get("validation_evidence_ref")
    if not evidence_ref or any(
            load_validation_evidence(evidence_ref, method, current_commit) is None
            for method in PUBLISHED_METHODS):
        raise ValueError("validation evidence is absent or stale")
    parity = _load_json(validation_run / "parity/waveform_frozen_parity.json",
                        "current direct parity")
    if (parity.get("git_commit") != current_commit
            or parity.get("all_within_project_tolerance") is not True):
        raise ValueError("current-commit direct parity is not PASS")
    return summary


def run_preflight(run_dir, validation_run, locked_config, gpu_count,
                  current_commit=None):
    if gpu_count != 4:
        raise ValueError("confirmatory requires exactly 4 GPUs, found %d" % gpu_count)
    if current_commit is None:
        require_clean_tree()
        current_commit = git_commit()
    run_dir = Path(run_dir)
    for method in (*PUBLISHED_METHODS, "norm_only_audio"):
        load_locked_method(locked_config, method)
    validate_validation_run(validation_run, locked_config, current_commit)

    split_ids = {}
    for split_id in CONFIRMATORY_SPLITS:
        split = get_split(split_id)
        dataset = TargetWaveformDataset(
            split["manifest_ref"], split["data_roots"], role=split["role"],
            format=split.get("manifest_format", "jsonl"))
        expected, _excluded = load_split_sample_ids(split_id)
        dataset_ids = {row["sample_id"] for row in dataset.rows}
        if not set(expected).issubset(dataset_ids):
            raise ValueError("%s dataset does not cover expected IDs" % split_id)
        cache_ids = set(FeatureCache(split["feature_cache_ref"]).load_by_id())
        missing = set(expected) - cache_ids
        if missing:
            raise ValueError("%s feature cache misses %d expected IDs" %
                             (split_id, len(missing)))
        split_ids[split_id] = set(expected)

    target10_ids, _ = load_split_sample_ids("target10")
    if split_ids["itw_target90"].intersection(target10_ids):
        raise ValueError("itw_target90 overlaps target10")
    output_mode = validate_output_directory(run_dir)
    if output_mode == "RESUME":
        provenance = _load_json(run_dir / "confirmatory_provenance.json",
                                "confirmatory provenance")
        if (provenance.get("git_commit") != current_commit
                or provenance.get("validation_run_ref") != str(Path(validation_run).resolve())
                or provenance.get("locked_config_content") != load_locked_document(
                    locked_config)):
            raise ValueError("CONFIG_MISMATCH: confirmatory provenance differs")
    return {"status": "PASS", "gpu_count": gpu_count, "output_mode": output_mode,
            "splits": {key: len(value) for key, value in split_ids.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--gpu-count", type=int, default=None,
                        help="test override; normally detected from torch")
    args = parser.parse_args()
    try:
        if args.gpu_count is None:
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            gpu_count = args.gpu_count
        report = run_preflight(args.run_dir, args.validation_run, args.locked_config,
                               gpu_count)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
