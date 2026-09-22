#!/usr/bin/env python
"""P2-A: target oracle threshold + bias-only calibration diagnostic (CPU).

POST_HOC_DEVELOPMENT_ONLY.  Bias-only calibration must leave EER/AUC strictly
unchanged (a global shift is monotone in score); only fixed-threshold operating
metrics may change.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from calibration.common import POST_HOC_MARKER, load_cal0, load_target10


def ranking_metrics(scores, labels):
    from eptta.evaluation.metrics import binary_metrics
    m = binary_metrics(list(scores), list(labels), 0.0)  # threshold unused for eer/auc
    return m["eer"], m["auroc"]


def operating_point(scores, labels, threshold):
    tp = sum(1 for s, y in zip(scores, labels) if s > threshold and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s > threshold and y == 0)
    tn = sum(1 for s, y in zip(scores, labels) if s <= threshold and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s <= threshold and y == 1)
    total = len(scores)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": 0.5 * (tpr + (tn / (tn + fp) if (tn + fp) else 0.0)),
        "tpr": tpr, "fpr": fpr, "fnr": fnr, "precision": precision,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def sweep(scores, labels):
    points = []
    for threshold in sorted(set(scores)):
        op = operating_point(scores, labels, threshold)
        points.append({"threshold": threshold, **{k: op[k] for k in
                       ("tpr", "fpr", "fnr", "accuracy", "balanced_accuracy")}})
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.run_dir) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    _sc, _yc, _ic, resources = load_cal0()
    st, yt, _it, _res = load_target10()
    tau0 = resources.tau0

    target_eer, target_auc = ranking_metrics(st, yt)
    frozen_op = operating_point(st, yt, tau0)

    points = sweep(st, yt)
    with (out_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["threshold", "tpr", "fpr", "fnr",
                                                    "accuracy", "balanced_accuracy"])
        writer.writeheader()
        writer.writerows(points)

    # Oracle thresholds (target10 labels, post-hoc only).
    tau_target_eer = min(points, key=lambda p: abs(p["fpr"] - p["fnr"]))["threshold"]
    tau_target_balacc = max(points, key=lambda p: p["balanced_accuracy"])["threshold"]
    bias_eer = tau0 - tau_target_eer
    bias_balacc = tau0 - tau_target_balacc

    # Bias-only oracle evaluation: shift scores, keep threshold tau0.
    shifted_eer = [s + bias_eer for s in st]
    shifted_balacc = [s + bias_balacc for s in st]
    op_eer = operating_point(shifted_eer, yt, tau0)
    op_balacc = operating_point(shifted_balacc, yt, tau0)
    eer_after, auc_after = ranking_metrics(shifted_balacc, yt)

    # Strict monotone-shift invariant check (uses float tolerance).
    tol = 1e-10
    result = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "tau0": tau0,
        "tau_target_eer": tau_target_eer,
        "tau_target_balacc": tau_target_balacc,
        "delta_tau_eer": tau_target_eer - tau0,
        "delta_tau_balacc": tau_target_balacc - tau0,
        "bias_eer": bias_eer,
        "bias_balacc": bias_balacc,
        "frozen": {"EER": target_eer, "AUC": target_auc, **frozen_op},
        "frozen_plus_bias_eer": {"accuracy": op_eer["accuracy"],
                                 "balanced_accuracy": op_eer["balanced_accuracy"],
                                 "tpr": op_eer["tpr"], "fpr": op_eer["fpr"],
                                 "fnr": op_eer["fnr"]},
        "frozen_plus_bias_balacc": {"accuracy": op_balacc["accuracy"],
                                    "balanced_accuracy": op_balacc["balanced_accuracy"],
                                    "tpr": op_balacc["tpr"], "fpr": op_balacc["fpr"],
                                    "fnr": op_balacc["fnr"]},
        "invariant_check": {
            "EER_unchanged": abs(eer_after - target_eer) <= tol,
            "AUC_unchanged": abs(auc_after - target_auc) <= tol,
        },
    }
    if not (result["invariant_check"]["EER_unchanged"] and result["invariant_check"]["AUC_unchanged"]):
        print("ERROR: bias-only calibration changed EER/AUC", file=sys.stderr)
        sys.exit(2)

    (out_dir / "threshold_oracle.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
