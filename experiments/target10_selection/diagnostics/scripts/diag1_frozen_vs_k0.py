#!/usr/bin/env python
"""Diagnostic 1: verify that target10-selected K=0 == pure frozen inference.

Compares, on the *identical* target90 sample IDs / features / model / score
definition:

  A. pure frozen inference      -> run_method("frozen", ...)
  B. EPConfig(steps=0)          -> run_method("ep_tta", ..., steps=0)

Saves per-sample scores and the absolute difference, and reports EER for both.
Verdict FROZEN_K0_EQUIVALENCE = PASS iff max abs score difference < 1e-7.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from _diag_common import (DIAG_RESULTS, EPConfig, FROZEN_BUNDLE, RESOURCES, CACHE,
                          RHO, GAMMA, LAMBDA_KEEP, TargetViews, run_method, load_context,
                          read_target90_records, compute_metrics, write_json_atomic, ensure_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    _bundle, resources, _meta, cache, features, threshold = load_context()
    sample_ids, labels = read_target90_records(args.limit)
    cache_id = cache.cache_id

    cfg_frozen = EPConfig(steps=0, lr=1e-5, rho=RHO, gamma=GAMMA, lambda_keep=LAMBDA_KEEP)

    rows = []
    frozen_scores = {}
    k0_scores = {}
    for sid in sample_ids:
        z = torch.from_numpy(features[sid])
        target = TargetViews(sid, z, cache_id)
        frozen_score = float(run_method("frozen", target, resources, cfg_frozen, {})["score"])
        k0_score = float(run_method("ep_tta", target, resources, cfg_frozen, {})["score"])
        frozen_scores[sid] = frozen_score
        k0_scores[sid] = k0_score
        rows.append({"sample_id": sid, "frozen_score": frozen_score,
                     "k0_score": k0_score, "abs_difference": abs(frozen_score - k0_score)})

    diffs = [r["abs_difference"] for r in rows]
    n = len(rows)
    mean_diff = sum(diffs) / n
    max_diff = max(diffs)
    sorted_diffs = sorted(diffs)
    median_diff = sorted_diffs[n // 2] if n % 2 else (sorted_diffs[n // 2 - 1] + sorted_diffs[n // 2]) / 2.0
    # prediction disagreement at the frozen source cal0 threshold tau0
    disagreements = sum(
        int((frozen_scores[s] > threshold) != (k0_scores[s] > threshold)) for s in sample_ids)

    frozen_metrics = compute_metrics(frozen_scores, labels, threshold)
    k0_metrics = compute_metrics(k0_scores, labels, threshold)

    verdict = "PASS" if max_diff < 1e-7 else "FAIL"

    summary = {
        "schema_version": "0.1.0",
        "diagnostic": "frozen_vs_k0",
        "test_data": "target90",
        "sample_count": n,
        "mean_abs_score_diff": mean_diff,
        "max_abs_score_diff": max_diff,
        "median_abs_score_diff": median_diff,
        "prediction_disagreement_count": disagreements,
        "threshold": threshold,
        "frozen_eer": frozen_metrics["EER"],
        "k0_eer": k0_metrics["EER"],
        "frozen_auc": frozen_metrics["AUC"],
        "k0_auc": k0_metrics["AUC"],
        "frozen_k0_equivalence": verdict,
        "equivalence_criterion": "max_abs_score_diff < 1e-7",
    }

    csv_path = DIAG_RESULTS / "frozen_vs_k0.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "frozen_score", "k0_score", "abs_difference"])
        writer.writeheader()
        writer.writerows(rows)

    write_json_atomic(DIAG_RESULTS / "frozen_vs_k0.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", csv_path)
    print("wrote", DIAG_RESULTS / "frozen_vs_k0.json")
    print("FROZEN_K0_EQUIVALENCE =", verdict)


if __name__ == "__main__":
    main()
