#!/usr/bin/env python3
"""Compare shared EP-TTA unified manifests with an ALLM-DF manifest directory."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path


LABELS = {"0": 0, "1": 1, "real": 0, "genuine": 0, "bonafide": 0,
          "bona-fide": 0, "fake": 1, "spoof": 1}
SHARED = (
    "asvspoof2019_la_train.csv",
    "asvspoof2019_la_dev.csv",
    "asvspoof2019_la_eval.csv",
    "asvspoof2021_la_eval.csv",
    "in_the_wild_eval.csv",
)


def normalized(row):
    path = row.get("path") or row.get("audio_path")
    label = LABELS.get(str(row.get("label", "")).strip().lower())
    return row.get("utt_id"), path, label, row.get("split")


def compare(left, right):
    count = 0
    with left.open(newline="", encoding="utf-8") as first, right.open(newline="", encoding="utf-8") as second:
        for line_no, pair in enumerate(zip_longest(csv.DictReader(first), csv.DictReader(second)), 2):
            ours, reference = pair
            if ours is None or reference is None:
                raise ValueError(f"row-count mismatch at {left.name}:{line_no}")
            if normalized(ours) != normalized(reference):
                raise ValueError(
                    f"content mismatch at {left.name}:{line_no}: "
                    f"ours={normalized(ours)!r}, reference={normalized(reference)!r}"
                )
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = {}
    for name in SHARED:
        counts[name] = compare(args.labels / "manifests" / name, args.reference_dir / name)
    print(json.dumps({"status": "PASS", "matched_fields": ["utt_id", "path", "label", "split"],
                      "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
