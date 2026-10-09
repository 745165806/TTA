"""Dependency-free protocol contract and score validation (no target labels)."""
import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
PROTOCOLS = ("episodic", "continual", "reset32", "reset128")
MANIFEST = ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"
FIXED = {
    "schema_version": "0.1.0", "comparison_track": "protocol_tta",
    "base_method": "tent_audio_native_v1", "dataset": "in_the_wild_target10",
    "shuffle": False, "sample_order_policy": "manifest_order", "optimizer": "Adam",
    "optimizer_state_policy": "retain_until_protocol_reset",
    "lr": 0.001, "steps": 1, "weight_decay": 0.0,
    "parameter_scope": "backend_norm_affine_v1",
    "normalization_policy": "model_eval_with_source_bn_running_statistics", "seed": 2026,
    "protocols": list(PROTOCOLS),
}


def load_config(path=EXP / "config.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config != FIXED or any(type(config[k]) is not type(v) for k, v in FIXED.items()):
        raise ValueError("protocol study requires exactly the preregistered config")
    return config


def reset_info(protocol, index):
    """Zero-based sequence/episode index; samples_since_reset includes this sample."""
    if protocol not in PROTOCOLS or type(index) is not int or index < 0:
        raise ValueError("invalid protocol or sequence index")
    if protocol == "episodic":
        return True, index, 1
    if protocol == "continual":
        return index == 0, 0, index + 1
    window = 32 if protocol == "reset32" else 128
    return index % window == 0, index // window, index % window + 1


def manifest_ids():
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["records"]
    allowed = {"schema_version", "sample_id", "root_key", "audio_relpath",
               "split_role", "sample_index"}
    if not records or any(set(row) != allowed or row["split_role"] != "select" for row in records):
        raise ValueError("unsafe target10 inference manifest")
    ids = [row["sample_id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate manifest sample ID")
    return ids


NUMERIC_FIELDS = (
    "source_frozen_score", "score_before_update", "current_pre_adapt_score", "score_after",
    "entropy_before", "entropy_after", "grad_norm", "parameter_delta_norm",
    "parameter_distance_from_source", "parameter_distance_before", "runtime",
    "source_reference_runtime",
)


def read_scores(path, protocol, expected_ids):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids) or ids != expected_ids:
        raise ValueError("duplicate, incomplete, or reordered samples: %s" % path)
    for index, row in enumerate(rows):
        reset, episode, since = reset_info(protocol, index)
        if (row["protocol"] != protocol or row["sample_index"] != index or
                row["reset_applied"] is not reset or row["episode_index"] != episode or
                row["samples_since_reset"] != since):
            raise ValueError("incorrect reset schedule: %s index %d" % (protocol, index))
        for key in NUMERIC_FIELDS:
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                raise ValueError("nonfinite/missing diagnostic: %s" % key)
        if row["score_before_update"] != row["current_pre_adapt_score"]:
            raise ValueError("pre-adaptation score aliases disagree")
        if (row["numeric_failure"] is not False or row["resource_failure"] is not False or
                row["bn_running_stats_unchanged"] is not True):
            raise ValueError("failed sample: %s" % row["sample_id"])
        for key in NUMERIC_FIELDS[4:]:
            if row[key] < 0:
                raise ValueError("negative diagnostic: %s" % key)
        if row["adaptation_applied"] is not (row["parameter_delta_norm"] > 0):
            raise ValueError("adaptation flag inconsistent with measured parameter delta")
        if reset and (row["parameter_distance_before"] != 0 or
                      abs(row["source_frozen_score"] - row["score_before_update"]) > 1e-5):
            raise ValueError("reset did not restore source state/path")
        if index and not reset:
            if row["parameter_distance_before"] != rows[index - 1]["parameter_distance_from_source"]:
                raise ValueError("state continuity broken between samples")
    return rows


def validate_run(run_dir, expected_ids):
    if not expected_ids:
        raise ValueError("empty expected sequence")
    records, configs = {}, []
    for protocol in PROTOCOLS:
        directory = Path(run_dir) / protocol
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        if (config["parameters"] != load_config() or config["protocol"] != protocol or
                config["sample_order_policy"] != "manifest_order" or
                config["sample_count"] != len(expected_ids) or
                config["manifest_path"] != str(MANIFEST)):
            raise ValueError("run provenance/config mismatch")
        configs.append(config)
        if (not config["updated_parameter_names"] or
                isinstance(config["tau0"], bool) or not isinstance(config["tau0"], (int, float)) or
                not math.isfinite(config["tau0"])):
            raise ValueError("missing selected parameters or invalid source threshold")
        records[protocol] = read_scores(directory / "scores.jsonl", protocol, expected_ids)
        if not any(row["adaptation_applied"] for row in records[protocol]):
            raise ValueError("no measured adaptation: %s" % protocol)
    for key in ("bundle_path", "checkpoint_ref", "detector_state_ref", "baseline_id", "source_run_id",
                "updated_parameter_names", "tau0", "manifest_path", "asset_root", "parameters"):
        if any(c[key] != configs[0][key] for c in configs[1:]):
            raise ValueError("cross-protocol mismatch: %s" % key)
    for index in range(len(expected_ids)):
        values = [records[p][index]["source_frozen_score"] for p in PROTOCOLS]
        if max(values) - min(values) > 1e-5:
            raise ValueError("source waveform reference differs across GPUs")
    return records, configs[0]
