#!/usr/bin/env python
"""Preflight validation + label-free select-manifest derivation.

Derives ``manifests/inwild_target10_select.json`` from the already-fixed
``inwild_target10.json`` (no re-split, no membership change), then hard-checks the
whole protocol before any parameter search may start:

  * select manifest: role == "select", count == len(records) == 3178,
    every record matches TargetInputManifest exactly (6 fields), no label field,
    no source_labels, every split_role == "select".
  * sample_id / sample_index match inwild_target10.json exactly.
  * target90: role == "target_test", count == len(records) == 28601.
  * target10 ∩ target90 == empty; union == 31779.
  * checkpoint / resources / cache paths exist.
  * GPU device count, parameter-combination count, git HEAD/status.

Any failure prints a clear report and exits non-zero (no GPU search).  A
structured report is written to results/preflight.json.
"""
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import torch

from eptta.data.permissions import TargetInputManifest, require_role
from eptta.errors import EPTTAError

from _common import (MANIFEST_DIR, RESULTS_DIR, ROOT, FROZEN_BUNDLE, RESOURCES, CACHE,
                     TARGET10, TARGET10_SELECT, TARGET90, FORBIDDEN_LABEL_KEYS, candidates)

EXPECTED_TARGET10 = 3178
EXPECTED_TARGET90 = 28601
EXPECTED_TOTAL = 31779
SEED = 2026
TARGET_INPUT_FIELDS = {field.name for field in fields(TargetInputManifest)}


def collect_label_keys(node, found):
    if isinstance(node, dict):
        for key, value in node.items():
            low = key.lower()
            if low in FORBIDDEN_LABEL_KEYS or "label" in low:
                found.add(key)
            collect_label_keys(value, found)
    elif isinstance(node, list):
        for item in node:
            collect_label_keys(item, found)


def git(cmd, *args):
    return subprocess.run(["git", cmd, *args], cwd=str(ROOT), text=True,
                          capture_output=True, check=False).stdout.strip()


def main():
    failures = []
    report = {}

    # ---- raw target10 (audit manifest) ----
    raw10 = json.loads(TARGET10.read_text(encoding="utf-8"))
    raw10_records = raw10["records"]
    report["target10_raw_count"] = len(raw10_records)
    report["target10_raw_role"] = raw10.get("role")

    # ---- derive label-free select manifest (no re-split, membership identical) ----
    select_records = []
    for r in raw10_records:
        select_records.append({
            "schema_version": "0.1.0",
            "sample_id": r["sample_id"],
            "sample_index": int(r["sample_index"]),
            "root_key": r["root_key"],
            "audio_relpath": r["path"],
            "split_role": "select",
        })
    select_doc = {
        "schema_version": raw10.get("schema_version", "0.1.0"),
        "dataset_id": "in_the_wild",
        "role": "select",
        "split_seed": SEED,
        "split_ratio": 0.1,
        "source_inference": raw10.get("source_inference"),
        "count": len(select_records),
        "records": select_records,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    TARGET10_SELECT.write_text(json.dumps(select_doc, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    # ---- validate select manifest ----
    report["target10_select_count"] = select_doc["count"]
    report["target10_select_role"] = select_doc.get("role")
    report["source_labels_present"] = "source_labels" in select_doc

    if select_doc.get("role") != "select":
        failures.append("select manifest role != 'select'")
    if select_doc["count"] != EXPECTED_TARGET10 or len(select_records) != EXPECTED_TARGET10:
        failures.append(f"select manifest count != {EXPECTED_TARGET10}")
    if "source_labels" in select_doc:
        failures.append("select manifest contains source_labels")
    if set(select_doc) & {"label", "canonical_label", "original_label", "target", "class", "y"}:
        failures.append("select manifest top level contains a label field")

    # existing permission design: method_selection only permits role "select"
    try:
        require_role(select_doc["role"], "method_selection")
    except EPTTAError as exc:
        failures.append("require_role(method_selection): %s" % exc)

    for i, r in enumerate(select_records):
        if set(r) != TARGET_INPUT_FIELDS:
            failures.append(f"select record {i} fields differ from TargetInputManifest: "
                            f"{sorted(set(r) ^ TARGET_INPUT_FIELDS)}")
        try:
            TargetInputManifest.from_dict(r)
        except EPTTAError as exc:
            failures.append(f"select record {i} invalid: {exc}")
        if r.get("split_role") != "select":
            failures.append(f"select record {i} split_role != 'select'")

    label_keys = set()
    collect_label_keys(select_doc, label_keys)
    report["target10_label_fields_found"] = sorted(label_keys)
    if label_keys:
        failures.append("label fields present in select manifest: %s" % sorted(label_keys))

    # ---- membership identical to raw target10 ----
    raw10_ids = [r["sample_id"] for r in raw10_records]
    raw10_index = [(r["sample_id"], int(r["sample_index"])) for r in raw10_records]
    sel_ids = [r["sample_id"] for r in select_records]
    sel_index = [(r["sample_id"], int(r["sample_index"])) for r in select_records]
    if raw10_ids != sel_ids:
        failures.append("select manifest sample_ids differ from raw target10")
    if raw10_index != sel_index:
        failures.append("select manifest sample_index differs from raw target10")

    # ---- target90 ----
    t90 = json.loads(TARGET90.read_text(encoding="utf-8"))
    t90_records = t90["records"]
    report["target90_count"] = len(t90_records)
    report["target90_role"] = t90.get("role")
    if t90.get("role") != "target_test":
        failures.append("target90 role != 'target_test'")
    if len(t90_records) != EXPECTED_TARGET90:
        failures.append(f"target90 count != {EXPECTED_TARGET90}")

    # ---- disjoint + union ----
    set10 = set(sel_ids)
    set90 = {r["sample_id"] for r in t90_records}
    report["intersection_count"] = len(set10 & set90)
    report["union_count"] = len(set10 | set90)
    if set10 & set90:
        failures.append("target10 and target90 intersect")
    if len(set10 | set90) != EXPECTED_TOTAL:
        failures.append(f"union count != {EXPECTED_TOTAL}")

    # ---- checkpoint / resources / cache paths ----
    try:
        from eptta.models.frozen import verify_frozen_export
        bundle, _m, _p, _s = verify_frozen_export(FROZEN_BUNDLE)
        report["checkpoint"] = bundle.get("checkpoint_ref")
    except EPTTAError as exc:
        report["checkpoint"] = None
        failures.append("frozen bundle verification failed: %s" % exc)
    for name, path in (("frozen_bundle", FROZEN_BUNDLE), ("resources", RESOURCES),
                       ("feature_cache", CACHE)):
        if not path.exists():
            failures.append("%s missing: %s" % (name, path))

    # ---- GPU / parameter combinations ----
    report["gpu_count"] = int(torch.cuda.device_count())
    combos = list(candidates())
    report["parameter_combination_count"] = len(combos)

    # ---- git ----
    report["git_head"] = git("rev-parse", "HEAD")
    report["git_status_short"] = git("status", "--short")

    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", RESULTS_DIR / "preflight.json")

    if failures:
        print("\nPREFLIGHT FAILED:", file=sys.stderr)
        for f in failures:
            print("  -", f, file=sys.stderr)
        sys.exit(1)
    print("\nPREFLIGHT PASSED")


if __name__ == "__main__":
    main()
