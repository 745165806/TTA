#!/usr/bin/env python
"""Step 6: build the two-row comparison table from the two result documents.

Reads results/target90_result.json and results/baseline_source_select.json and
writes results/comparison.csv with columns:
    method, selection_data, K, lr, steps, EER, minDCF
"""
import csv
import json
import sys

from _common import RESULTS_DIR

FIELDS = ["method", "selection_data", "K", "lr", "steps", "EER", "minDCF"]


def row_from(doc):
    metrics = doc["metrics"]
    return {
        "method": doc["method"],
        "selection_data": doc["selection_data"],
        "K": doc["K"],
        "lr": doc["lr"],
        "steps": doc["steps"],
        "EER": metrics["EER"],
        "minDCF": "" if metrics.get("minDCF") is None else metrics["minDCF"],
    }


def main():
    target90 = RESULTS_DIR / "target90_result.json"
    baseline = RESULTS_DIR / "baseline_source_select.json"
    if not target90.is_file() or not baseline.is_file():
        print("ERROR: target90_result.json and baseline_source_select.json are required",
              file=sys.stderr)
        sys.exit(1)

    sel_doc = json.loads(target90.read_text(encoding="utf-8"))
    base_doc = json.loads(baseline.read_text(encoding="utf-8"))

    rows = [row_from(base_doc), row_from(sel_doc)]
    out = RESULTS_DIR / "comparison.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print("wrote", out)


if __name__ == "__main__":
    main()
