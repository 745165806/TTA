#!/usr/bin/env python
"""Paired bootstrap over sample IDs for delta-EER / delta-AUC (seed fixed)."""
import random

from eptta.training.selection import equal_error_rate


def _auc(scores, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg = ((i + 1) + j) / 2.0
        rank_sum += avg * sum(l for _, l in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _metrics(scores, labels):
    return {"EER": equal_error_rate(scores, labels), "AUC": _auc(scores, labels)}


def paired_bootstrap(frozen_scores, method_scores, labels, seed=2026, n_bootstrap=2000):
    """Paired resampling by sample ID; returns delta-EER / delta-AUC CIs."""
    ids = sorted(frozen_scores)
    frozen = [frozen_scores[k] for k in ids]
    method = [method_scores[k] for k in ids]
    ys = [labels[k] for k in ids]

    frozen_m = _metrics(frozen, ys)
    method_m = _metrics(method, ys)

    rng = random.Random(seed)
    n = len(ids)
    delta_eer, delta_auc = [], []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        f = [frozen[i] for i in idx]
        m = [method[i] for i in idx]
        y = [ys[i] for i in idx]
        fm = _metrics(f, y)
        mm = _metrics(m, y)
        if fm["AUC"] is None or mm["AUC"] is None:
            continue
        delta_eer.append(mm["EER"] - fm["EER"])
        delta_auc.append(mm["AUC"] - fm["AUC"])

    def ci(values):
        values = sorted(values)
        return {
            "mean": sum(values) / len(values),
            "ci_low": values[int(0.025 * (len(values) - 1))],
            "ci_high": values[int(0.975 * (len(values) - 1))],
        }

    return {
        "frozen": frozen_m, "method": method_m,
        "delta_EER": ci(delta_eer), "delta_AUC": ci(delta_auc),
        "n_bootstrap": len(delta_eer),
    }


def significance(bootstrap_result):
    """Interpret paired bootstrap into significance flags."""
    eer_low, eer_high = bootstrap_result["delta_EER"]["ci_low"], bootstrap_result["delta_EER"]["ci_high"]
    auc_low, auc_high = bootstrap_result["delta_AUC"]["ci_low"], bootstrap_result["delta_AUC"]["ci_high"]
    return {
        "EER_SIGNIFICANT_GAIN": eer_high < 0,
        "AUC_SIGNIFICANT_GAIN": auc_low > 0,
        "EER_SIGNIFICANT_HARM": eer_low > 0,
        "AUC_SIGNIFICANT_HARM": auc_high < 0,
    }
