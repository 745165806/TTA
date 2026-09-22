#!/usr/bin/env python
"""P1 confirmatory aggregation: Frozen vs taskaware_full per dataset (post-hoc labels)."""
import argparse
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from eptta.data.io import iter_jsonl
from eptta.evaluation.metrics import binary_metrics
from scripts.common import POST_HOC_MARKER, load_context
from scripts.confirmatory_worker import DATASETS


def dataset_metrics(dataset, records, labels, threshold):
    after = {r["sample_id"]: r["score_after"] for r in records}
    before = {r["sample_id"]: r["score_before"] for r in records}
    order = sorted(after)
    frozen = binary_metrics([before[k] for k in order], [labels[k] for k in order], threshold)
    adapted = binary_metrics([after[k] for k in order], [labels[k] for k in order], threshold,
                             frozen_scores=[before[k] for k in order])
    signed = [(2 * labels[r["sample_id"]] - 1) * r["delta_score"] for r in records]
    return {
        "dataset": dataset,
        "count": len(records),
        "frozen": {"EER": frozen["eer"], "AUC": frozen["auroc"],
                   "accuracy_at_tau0": (frozen["tp"] + frozen["tn"]) / frozen["count"]},
        "taskaware_full": {
            "EER": adapted["eer"], "AUC": adapted["auroc"],
            "accuracy_at_tau0": (adapted["tp"] + adapted["tn"]) / adapted["count"],
            "adaptation_coverage": statistics.fmean(r["adaptation_applied"] for r in records),
            "helpful_flips": adapted["helpful_flips"], "harmful_flips": adapted["harmful_flips"],
            "mean_signed_task_delta": statistics.fmean(signed),
            "mean_R_norm": statistics.fmean(r["final_R_norm"] for r in records),
            "mean_abs_delta_score": statistics.fmean(abs(r["delta_score"]) for r in records),
            "source_safety_reject_rate": statistics.fmean(r["safety_rejected"] for r in records),
            "numeric_fallback_count": sum(1 for r in records if r["numeric_fallback"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    confirm_dir = run_dir / "confirmatory"
    _b, _res, _meta, _cache, _features, threshold = load_context()

    results = []
    for dataset in sorted(DATASETS):
        path = confirm_dir / ("confirm_%s.jsonl" % dataset)
        if not path.is_file():
            continue
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        labels = {row["sample_id"]: int(row["canonical_label"])
                  for row in iter_jsonl(ROOT / DATASETS[dataset]["labels_manifest"])}
        results.append(dataset_metrics(dataset, records, labels, threshold))

    summary = {"POST_HOC_DEVELOPMENT_ONLY": True, "threshold": threshold, "datasets": results}
    (confirm_dir / "confirmatory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
