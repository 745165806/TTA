#!/usr/bin/env python
"""Step 1: split the in-the-wild manifest into a fixed 10%/90% target split.

Reads the existing In-the-Wild inference/labels manifests (read-only), sorts the
sample ids, and with a fixed seed=2026 draws a random 10% selection (target10)
and the remaining 90% (target90).  The label field is preserved in both output
manifests, but the downstream *selection* stage is forbidden from reading it.

Exits non-zero on any inconsistency so the driver script's ``set -e`` stops.
"""
import json
import sys
from pathlib import Path

import numpy as np

from _common import MANIFEST_DIR, ROOT

INFERENCE = ROOT / "data/manifests_v2/in_the_wild/inference/target_test.jsonl"
LABELS = ROOT / "data/manifests_v2/in_the_wild/labels/target_test.jsonl"
SEED = 2026
RATIO = 0.10
EXPECTED_TOTAL = 31779


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    inference = read_jsonl(INFERENCE)
    labels = {row["sample_id"]: int(row["canonical_label"]) for row in read_jsonl(LABELS)}

    if len(inference) != len(labels):
        print("ERROR: inference/labels length mismatch", file=sys.stderr)
        sys.exit(1)
    if len(inference) != EXPECTED_TOTAL:
        print(f"ERROR: expected {EXPECTED_TOTAL} samples, got {len(inference)}", file=sys.stderr)
        sys.exit(1)
    for row in inference:
        if row["sample_id"] not in labels:
            print(f"ERROR: label missing for {row['sample_id']}", file=sys.stderr)
            sys.exit(1)

    records = []
    for row in inference:
        sample_id = row["sample_id"]
        records.append({
            "sample_id": sample_id,
            "path": row["audio_relpath"],
            "label": labels[sample_id],
            "root_key": row["root_key"],
            "split_role": row["split_role"],
            "sample_index": int(row["sample_index"]),
        })

    records.sort(key=lambda r: r["sample_id"])
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))

    n10 = int(round(RATIO * len(records)))
    target10 = [records[i] for i in order[:n10]]
    target90 = [records[i] for i in order[n10:]]
    target10.sort(key=lambda r: r["sample_id"])
    target90.sort(key=lambda r: r["sample_id"])

    if len(target10) + len(target90) != len(records):
        print("ERROR: split counts do not sum to total", file=sys.stderr)
        sys.exit(1)
    if len({r["sample_id"] for r in target10} & {r["sample_id"] for r in target90}) != 0:
        print("ERROR: target10/target90 overlap", file=sys.stderr)
        sys.exit(1)

    def wrap(subset, ratio):
        return {
            "schema_version": "0.1.0",
            "dataset_id": "in_the_wild",
            "role": "target_test",
            "split_seed": SEED,
            "split_ratio": ratio,
            "source_inference": str(INFERENCE.relative_to(ROOT)),
            "source_labels": str(LABELS.relative_to(ROOT)),
            "count": len(subset),
            "records": subset,
        }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "inwild_target10.json").write_text(
        json.dumps(wrap(target10, RATIO), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (MANIFEST_DIR / "inwild_target90.json").write_text(
        json.dumps(wrap(target90, 1.0 - RATIO), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema_version": "0.1.0",
        "seed": SEED,
        "ratio": RATIO,
        "total": len(records),
        "target10_count": len(target10),
        "target90_count": len(target90),
        "target10_class_counts": {
            "bonafide": sum(1 for r in target10 if r["label"] == 0),
            "spoof": sum(1 for r in target10 if r["label"] == 1),
        },
        "target90_class_counts": {
            "bonafide": sum(1 for r in target90 if r["label"] == 0),
            "spoof": sum(1 for r in target90 if r["label"] == 1),
        },
        "rng": "numpy.random.default_rng(2026).permutation",
    }
    (MANIFEST_DIR / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"target10_count={len(target10)}")
    print(f"target90_count={len(target90)}")
    print(f"total={len(records)}")
    print("manifests written:", MANIFEST_DIR / "inwild_target10.json",
          ",", MANIFEST_DIR / "inwild_target90.json")


if __name__ == "__main__":
    main()
