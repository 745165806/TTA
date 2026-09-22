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


def get_split(split_id):
    if split_id not in SPLITS:
        raise SystemExit("unknown split: %s (known: %s)" % (split_id, sorted(SPLITS)))
    return SPLITS[split_id]
