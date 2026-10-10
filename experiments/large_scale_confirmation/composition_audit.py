"""Descriptive post-score Codecfake sample-rate check; never controls adaptation."""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from eptta.evaluation.metrics import binary_metrics
from experiments.large_scale_confirmation.analyze import (
    complete_scores, read_json, read_order_scores, selected_labels, write_csv_once,
)
from experiments.large_scale_confirmation.run_scores import ORDERS


def audit(run):
    _config, assignments = complete_scores(run)
    labels, _attacks, _coverage = selected_labels(assignments)
    rows = assignments["codecfake"]
    by_rate = defaultdict(list)
    for row in rows:
        path = Path(row["root_ref"]) / row["audio_relpath"]
        by_rate[sf.info(path).samplerate].append(row["sample_id"])
    tau0 = read_json(run / "diagnostics/codecfake_provenance.json")["tau0"]
    output = []
    for rate, ids in sorted(by_rate.items()):
        ids = sorted(ids)
        y = [labels["codecfake"][sid] for sid in ids]
        if not (0 < sum(y) < len(y)):
            output.append({"sample_rate": rate, "count": len(ids), "bonafide": len(y)-sum(y),
                           "spoof": sum(y), "arm": "TWO_CLASS_UNAVAILABLE",
                           "mean_delta_AUC": None, "std_delta_AUC": None,
                           "positive_orders": None})
            continue
        effects = {"Local-Base B32": [], "Local-O1 B32": []}
        for order in ORDERS:
            data = read_order_scores(run, "codecfake", order, ids)
            frozen = [data[sid]["Frozen"]["score_after"] for sid in ids]
            frozen_auc = binary_metrics(frozen, y, tau0)["auroc"]
            for arm in effects:
                adapted = [data[sid][arm]["score_after"] for sid in ids]
                effects[arm].append(binary_metrics(adapted, y, tau0)["auroc"] - frozen_auc)
        for arm, values in effects.items():
            output.append({"sample_rate": rate, "count": len(ids), "bonafide": len(y)-sum(y),
                           "spoof": sum(y), "arm": arm,
                           "mean_delta_AUC": statistics.mean(values),
                           "std_delta_AUC": statistics.stdev(values),
                           "positive_orders": sum(value > 0 for value in values)})
    write_csv_once(run / "analysis/codecfake_sample_rate_composition.csv", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.run))
