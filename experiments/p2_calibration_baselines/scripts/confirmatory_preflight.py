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

from baselines.locked_config import PUBLISHED_METHODS, load_locked_method
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


def run_preflight(run_dir, locked_config, gpu_count):
    if gpu_count != 4:
        raise ValueError("confirmatory requires exactly 4 GPUs, found %d" % gpu_count)
    for method in (*PUBLISHED_METHODS, "norm_only_audio"):
        load_locked_method(locked_config, method)

    run_dir = Path(run_dir)
    validate_port_results(run_dir / "pilot/port_validation.json")
    parity = _load_json(run_dir / "parity/waveform_frozen_parity.json",
                        "target10 direct parity")
    if parity.get("all_within_project_tolerance") is not True:
        raise ValueError("target10 direct parity is not PASS")

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
    output_mode = validate_output_directory(run_dir / "confirmatory")
    return {"status": "PASS", "gpu_count": gpu_count, "output_mode": output_mode,
            "splits": {key: len(value) for key, value in split_ids.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
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
        report = run_preflight(args.run_dir, args.locked_config, gpu_count)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
