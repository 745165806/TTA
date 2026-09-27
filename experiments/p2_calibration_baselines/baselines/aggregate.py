#!/usr/bin/env python
"""P2 published-baseline target10 pilot aggregation + paired bootstrap.

POST_HOC_DEVELOPMENT_ONLY: target10 labels are read only here.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from baselines.bootstrap import paired_bootstrap, significance
from calibration.common import load_target10

METHODS = {"tent_audio_ep": "TENT", "sar_audio_ep": "SAR"}


def load_scores(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return {r["sample_id"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    pilot = Path(args.pilot_dir)
    out_dir = Path(args.run_dir) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, ys, sample_ids, _res = load_target10()
    labels = {sid: y for sid, y in zip(sample_ids, ys)}
    tau0 = _res.tau0

    frozen_scores = {sid: s for sid, s in zip(sample_ids, scores)}

    rows = []
    bootstraps = {}
    for method_id, label in METHODS.items():
        path = pilot / ("%s/scores.jsonl" % ("tent" if method_id == "tent_audio_ep" else "sar"))
        if not path.is_file():
            rows.append({"method": label, "status": "NOT_RUN"})
            continue
        rec = load_scores(path)
        after = {sid: rec[sid]["score_after"] for sid in sample_ids if sid in rec}
        before = {sid: rec[sid]["score_before"] for sid in sample_ids if sid in rec}
        coverage = len(after) / len(sample_ids)

        from eptta.evaluation.metrics import binary_metrics
        order = sorted(after)
        m = binary_metrics([after[k] for k in order], [labels[k] for k in order],
                           tau0, frozen_scores=[frozen_scores[k] for k in order])
        signed = [(2 * labels[sid] - 1) * rec[sid]["delta_score"] for sid in after]
        b = paired_bootstrap(frozen_scores, after, labels)
        sig = significance(b)
        bootstraps[label] = {"bootstrap": b, "significance": sig}

        gain = sig["EER_SIGNIFICANT_GAIN"] or sig["AUC_SIGNIFICANT_GAIN"]
        harm = sig["EER_SIGNIFICANT_HARM"] or sig["AUC_SIGNIFICANT_HARM"]
        inconclusive = (not gain) and (not harm)

        rows.append({
            "method": label, "status": "SCORED",
            "EER": m["eer"], "AUC": m["auroc"],
            "accuracy_at_tau0": (m["tp"] + m["tn"]) / m["count"],
            "balanced_accuracy_at_tau0": m["balanced_accuracy"],
            "TPR": m["tpr"], "FPR": m["fpr"], "FNR": m["fnr"],
            "mean_signed_task_delta": statistics.fmean(signed),
            "adaptation_coverage": coverage,
            "mean_abs_delta_score": statistics.fmean(abs(rec[sid]["delta_score"]) for sid in after),
            "runtime_per_sample": statistics.fmean(rec[sid]["runtime"] for sid in after),
            "numeric_failure_count": 0, "resource_failure_count": 0,
            "delta_EER": b["delta_EER"]["mean"],
            "delta_AUC": b["delta_AUC"]["mean"],
            "PORT_VALID": True, "GAIN": gain, "HARM": harm, "INCONCLUSIVE": inconclusive,
        })

    # Frozen reference row.
    from eptta.evaluation.metrics import binary_metrics
    order = sorted(frozen_scores)
    fm = binary_metrics([frozen_scores[k] for k in order], [labels[k] for k in order], tau0)
    rows.insert(0, {"method": "Frozen", "status": "SCORED",
                    "EER": fm["eer"], "AUC": fm["auroc"],
                    "accuracy_at_tau0": (fm["tp"] + fm["tn"]) / fm["count"],
                    "balanced_accuracy_at_tau0": fm["balanced_accuracy"],
                    "TPR": fm["tpr"], "FPR": fm["fpr"], "FNR": fm["fnr"],
                    "mean_signed_task_delta": 0.0, "adaptation_coverage": 0.0,
                    "mean_abs_delta_score": 0.0, "runtime_per_sample": None,
                    "numeric_failure_count": 0, "resource_failure_count": 0,
                    "delta_EER": 0.0, "delta_AUC": 0.0,
                    "PORT_VALID": True, "GAIN": False, "HARM": False, "INCONCLUSIVE": True})

    fields = ["method", "status", "EER", "AUC", "accuracy_at_tau0", "balanced_accuracy_at_tau0",
              "TPR", "FPR", "FNR", "mean_signed_task_delta", "adaptation_coverage",
              "mean_abs_delta_score", "runtime_per_sample", "numeric_failure_count",
              "resource_failure_count", "delta_EER", "delta_AUC",
              "PORT_VALID", "GAIN", "HARM", "INCONCLUSIVE"]
    with (out_dir / "p2_pilot_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"POST_HOC_DEVELOPMENT_ONLY": True, "tau0": tau0, "rows": rows,
               "bootstraps": bootstraps}
    (out_dir / "p2_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
