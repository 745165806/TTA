"""Post-hoc WaveFake duration stratum check on existing held-out predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from eptta.evaluation.metrics import binary_metrics


def run(cache_run: Path, capacity_run: Path):
    material = json.loads((cache_run / "diagnostics/materialization.json").read_text())
    groups = defaultdict(dict)
    for sample_id, row in material.items():
        audio_id, code = sample_id.rsplit(":", 1)
        groups[audio_id][code] = row["source_frames"] / 22050
    if len(groups) != 2048 or any(len(group) != 2 or "R" not in group for group in groups.values()):
        raise ValueError("duration analysis lost fixed matched pairs")
    # 64600/16000 is the unchanged production waveform crop/repeat boundary.
    long_groups = {name for name, pair in groups.items() if min(pair.values()) > 64600 / 16000}
    rows = [json.loads(line) for line in (capacity_run / "predictions/held_out.jsonl").open()]
    if len(rows) != 4096 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("held-out predictions missing selected IDs")
    result = []
    for stratum, selected in (("both_longer_than_production_crop", long_groups),
                              ("at_least_one_shorter_or_equal", set(groups) - long_groups)):
        subset = [row for row in rows if row["sample_id"].rsplit(":", 1)[0] in selected]
        if len(subset) != 2 * len(selected):
            raise ValueError("duration stratum lost a pair")
        for arm in ("C0_Frozen", "C1_supervised_R", "C2_linear", "C3_nonlinear"):
            if arm not in subset[0]["scores"]:
                continue
            metrics = binary_metrics([row["scores"][arm] for row in subset],
                                     [row["label"] for row in subset], 0.)
            result.append({"stratum": stratum, "content_pairs": len(selected),
                           "sample_count": len(subset), "arm": arm,
                           "auc": metrics["auroc"], "eer": metrics["eer"]})
    output = capacity_run / "analysis/duration_strata.csv"
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-run", type=Path, required=True)
    parser.add_argument("--capacity-run", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache_run, args.capacity_run)
