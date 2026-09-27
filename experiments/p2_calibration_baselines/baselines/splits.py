#!/usr/bin/env python
"""Strict split registry for P2.1 waveform ports.

Paths are parsed from the existing project configs (never guessed).
Worker-facing entries are label-free; label_ref is for aggregators only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from eptta.config.schema import read_document


def _cfg(path):
    return read_document(ROOT / path)


def _load_target_splits():
    itw = _cfg("configs/local_v2/ssl_aasist_target/ssl_itw_ep.yaml")
    la = _cfg("configs/local_v2/ssl_aasist_target/ssl_asv2021_la_ep.yaml")
    df = _cfg("configs/local_v2/ssl_aasist_target/ssl_asv2021_df_ep.yaml")
    control = _cfg("configs/local_v2/ssl_aasist_control_main/ep_tta.yaml")

    itw_root = itw["extraction"]["data_roots"]["in_the_wild"]
    la_root = la["extraction"]["data_roots"]["asvspoof2021_la"]
    df_root = df["extraction"]["data_roots"]["asvspoof2021_df"]
    control_root = control["extraction"]["data_roots"]["asvspoof2019_la"]

    select_manifest = (ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json")
    select_ids = {r["sample_id"] for r in json.loads(select_manifest.read_text())["records"]}

    return {
        "target10": {
            "split_id": "target10",
            "manifest_ref": str(ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"),
            "manifest_format": "json_records",
            "data_roots": {"in_the_wild": itw_root},
            "feature_cache_ref": str(ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"),
            "role": "select",
            "label_ref": str(ROOT / "experiments/target10_selection/manifests/inwild_target10.json"),
            "label_format": "json_records",
            "dataset_id": "in_the_wild",
        },
        "itw_target90": {
            "split_id": "itw_target90",
            "manifest_ref": str(ROOT / "data/manifests_v2/in_the_wild/inference/target_test.jsonl"),
            "manifest_format": "jsonl",
            "data_roots": {"in_the_wild": itw_root},
            "feature_cache_ref": str(ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"),
            "role": "target_test",
            "label_ref": str(ROOT / "data/manifests_v2/in_the_wild/labels/target_test.jsonl"),
            "label_format": "jsonl",
            "dataset_id": "in_the_wild",
            "exclude_select": True,
        },
        "asv2021_la": {
            "split_id": "asv2021_la",
            "manifest_ref": str(ROOT / "data/manifests_v2/asv2021_la_eval/inference/target_test.jsonl"),
            "manifest_format": "jsonl",
            "data_roots": {"asvspoof2021_la": la_root},
            "feature_cache_ref": str(ROOT / "outputs_v2/ssl_aasist/cache-target-asv2021_la_eval"),
            "role": "target_test",
            "label_ref": str(ROOT / "data/manifests_v2/asv2021_la_eval/labels/target_test.jsonl"),
            "label_format": "jsonl",
            "dataset_id": "asvspoof2021_la",
        },
        "asv2021_df": {
            "split_id": "asv2021_df",
            "manifest_ref": str(ROOT / "data/manifests_v2/asv2021_df_eval/inference/target_test.jsonl"),
            "manifest_format": "jsonl",
            "data_roots": {"asvspoof2021_df": df_root},
            "feature_cache_ref": str(ROOT / "outputs_v2/ssl_aasist/cache-target-asv2021_df_eval"),
            "role": "target_test",
            "label_ref": str(ROOT / "data/manifests_v2/asv2021_df_eval/labels/target_test.jsonl"),
            "label_format": "jsonl",
            "dataset_id": "asvspoof2021_df",
        },
        "control_test": {
            "split_id": "control_test",
            "manifest_ref": str(ROOT / "data/manifests_v2/asv2019_la/inference/control_test.jsonl"),
            "manifest_format": "jsonl",
            "data_roots": {"asvspoof2019_la": control_root},
            "feature_cache_ref": str(ROOT / "outputs_v2/ssl_aasist/cache-control_test"),
            "role": "control_test",
            "label_ref": str(ROOT / "data/manifests_v2/asv2019_la/labels/control_test.jsonl"),
            "label_format": "jsonl",
            "dataset_id": "asvspoof2019_la",
        },
    }


SPLITS = _load_target_splits()

TARGET10_SELECT_MANIFEST = ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"


def get_split(split_id):
    if split_id not in SPLITS:
        raise SystemExit("unknown split: %s (known: %s)" % (split_id, sorted(SPLITS)))
    return SPLITS[split_id]


def _target10_select_ids():
    doc = json.loads(TARGET10_SELECT_MANIFEST.read_text(encoding="utf-8"))
    return {r["sample_id"] for r in doc["records"]}


def load_split_sample_ids(split_id):
    """Label-free sample IDs for a split, with ``exclude_select`` actually applied.

    Returns ``(sample_ids, excluded_select_count)``.  For ``itw_target90`` the
    target10 select IDs are removed from the full in-the-wild target_test set and
    the disjointness is asserted.
    """
    split = get_split(split_id)
    fmt = split.get("manifest_format", "jsonl")
    ids = []
    if fmt == "json_records":
        doc = json.loads(Path(split["manifest_ref"]).read_text(encoding="utf-8"))
        for record in doc["records"]:
            if record.get("split_role") == split["role"]:
                ids.append(record["sample_id"])
    else:
        with open(split["manifest_ref"], encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split_role") == split["role"]:
                    ids.append(row["sample_id"])

    excluded = 0
    if split.get("exclude_select"):
        select_ids = _target10_select_ids()
        remaining = [sid for sid in ids if sid not in select_ids]
        excluded = len(ids) - len(remaining)
        assert set(remaining).isdisjoint(select_ids)
        ids = remaining
    return ids, excluded


def load_split_ids_with_meta(split_id):
    ids, excluded = load_split_sample_ids(split_id)
    return {
        "split_id": split_id,
        "sample_ids": ids,
        "excluded_select_count": excluded,
        "remaining_count": len(ids),
    }
