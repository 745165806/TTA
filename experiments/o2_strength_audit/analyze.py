"""Post-score, selected-label development analysis of fixed O2 head updates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np

from eptta.evaluation.metrics import binary_metrics
from experiments.head_capacity_geometry.resources import select_rows
from experiments.head_capacity_geometry.supervised_labels import load_labels
from experiments.o2_strength_audit.run_scores import ARMS, BATCH, HERE


DOMAINS = {"itw": 3178, "wavefake": 4096}


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv_new(path, rows):
    if not rows:
        raise ValueError("empty analysis CSV")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_scores_before_labels(out):
    """Both complete score files must pass coverage/numeric checks before labels open."""
    completion = json.loads((out / "score_completion.json").read_text(encoding="utf-8"))
    if (completion["complete"] is not True or completion["target_labels_read"] is not False or
            completion["target90_labels_or_metrics_read"] is not False or
            completion["buffer_size"] != BATCH or tuple(completion["arms"]) != ARMS):
        raise ValueError("label-free score completion contract failed")
    result = {}
    for domain, count in DOMAINS.items():
        ids, _, _ = select_rows(domain)
        rows = [json.loads(line) for line in
                (out / "scores" / f"{domain}.jsonl").open(encoding="utf-8")]
        if (len(rows) != count or len(ids) != count or
                completion["domains"][domain]["count"] != count or
                completion["domains"][domain]["numeric_failures"] != 0):
            raise ValueError(f"{domain}: score count/completion failure")
        if [row["sample_id"] for row in rows] != ids or len(set(ids)) != count:
            raise ValueError(f"{domain}: score ID/order failure")
        if any(row["domain"] != domain or row["buffer_index"] != i // BATCH or
               set(row["scores"]) != set(ARMS) or
               not all(math.isfinite(float(row["scores"][arm])) for arm in ARMS)
               for i, row in enumerate(rows)):
            raise ValueError(f"{domain}: score arm/buffer/finite failure")
        result[domain] = (ids, {arm: np.asarray(
            [row["scores"][arm] for row in rows], dtype=np.float64) for arm in ARMS})
    return result, completion


def fast_eer_auc(scores, labels):
    """Vectorized exact project EER/AUC semantics for bootstrap draws."""
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives or not np.isfinite(values).all():
        raise ValueError("two finite classes required")
    order = np.argsort(-values, kind="mergesort")
    sorted_values, sorted_y = values[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_values) != 0) + 1, len(values)]
    true_accepts = np.cumsum(sorted_y, dtype=np.int64)[ends - 1]
    false_accepts = ends - true_accepts
    far = np.r_[0., false_accepts / negatives]
    frr = 1. - np.r_[0., true_accepts / positives]
    auc = float(np.trapz(1. - frr, far))
    difference = far - frr
    crossing = int(np.searchsorted(difference, 0., side="left"))
    if difference[crossing] == 0:
        eer = float(far[crossing])
    else:
        left, right = crossing - 1, crossing
        fraction = abs(difference[left]) / (abs(difference[left]) + abs(difference[right]))
        far_cross = far[left] + fraction * (far[right] - far[left])
        frr_cross = frr[left] + fraction * (frr[right] - frr[left])
        eer = float((far_cross + frr_cross) / 2.)
    return eer, auc


def paired_bootstrap(domain, labels, scores):
    rng = np.random.default_rng(2026)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    draws = {arm: {"delta_auc": [], "delta_eer": []} for arm in ARMS[1:]}
    for _ in range(1000):
        selected = np.r_[rng.choice(class0, len(class0), replace=True),
                         rng.choice(class1, len(class1), replace=True)]
        metrics = {arm: fast_eer_auc(values[selected], labels[selected])
                   for arm, values in scores.items()}
        for arm in ARMS[1:]:
            draws[arm]["delta_auc"].append(metrics[arm][1] - metrics["Frozen"][1])
            draws[arm]["delta_eer"].append(metrics[arm][0] - metrics["Frozen"][0])
    return [{"domain": domain, "arm": arm, "replicates": 1000, "seed": 2026,
             "delta_auc_ci_low": float(np.quantile(draws[arm]["delta_auc"], .025)),
             "delta_auc_ci_high": float(np.quantile(draws[arm]["delta_auc"], .975)),
             "delta_eer_ci_low": float(np.quantile(draws[arm]["delta_eer"], .025)),
             "delta_eer_ci_high": float(np.quantile(draws[arm]["delta_eer"], .975))}
            for arm in ARMS[1:]]


def read_diagnostics(out):
    with (out / "diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = sum((n + BATCH - 1) // BATCH for n in DOMAINS.values()) * len(ARMS)
    if len(rows) != expected:
        raise ValueError("buffer diagnostics coverage mismatch")
    result = {}
    for domain, count in DOMAINS.items():
        result[domain] = {}
        for arm in ARMS:
            group = [row for row in rows if row["domain"] == domain and row["arm"] == arm]
            if (len(group) != (count + BATCH - 1) // BATCH or
                    sum(int(row["buffer_size"]) for row in group) != count or
                    [int(row["buffer_index"]) for row in group] != list(range(len(group))) or
                    any(row["numeric_status"] != "ok" for row in group)):
                raise ValueError(f"{domain}/{arm}: buffer diagnostics failure")
            def weighted(field):
                values = [float(row[field]) for row in group]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{domain}/{arm}: nonfinite {field}")
                return sum(int(row["buffer_size"]) * value
                           for row, value in zip(group, values)) / count
            result[domain][arm] = {
                "mean_abs_score_delta": weighted("mean_abs_score_delta"),
                "mean_head_angle_degrees": weighted("head_angle_degrees"),
                "mean_effective_step_norm": weighted("effective_step_norm"),
                "mean_o2_gradient_norm": weighted("gradient_norm"),
                "mean_source_head_norm": weighted("source_head_norm")}
    return result


def decision(metrics):
    def hits(domain, arm):
        row = metrics[domain][arm]
        floor = .005 if domain == "itw" else .01
        return row["delta_auc"] >= floor or row["delta_eer"] <= -floor
    normalized = ARMS[2:]
    both = [arm for arm in normalized if hits("itw", arm) and hits("wavefake", arm)]
    if both and not hits("itw", "O2-raw") and not hits("wavefake", "O2-raw"):
        return "O2_DIRECTION_USEFUL_BUT_RAW_MAGNITUDE_TOO_WEAK", both
    if both:
        return "INCONCLUSIVE", both
    if all(not hits(domain, arm) for domain in DOMAINS for arm in normalized):
        return "O2_ALIGNMENT_INSUFFICIENT_FOR_TASK_CORRECTION", []
    if any(hits(domain, arm) != hits(other, arm) for arm in normalized
           for domain, other in (("itw", "wavefake"),)):
        return "DOMAIN_DEPENDENT_O2_EFFECT", []
    return "INCONCLUSIVE", []


def run(run_id):
    out = HERE / "results" / run_id
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=False)
    try:
        scored, completion = read_scores_before_labels(out)
        diagnostics = read_diagnostics(out)
        # The only target-label access, after both full score files are validated.
        metrics = {}
        table = []
        bootstrap = []
        for domain, (ids, scores) in scored.items():
            labels = load_labels(domain, ids)
            threshold = float(completion["wavefake_cache"]["tau0"])
            frozen = binary_metrics(scores["Frozen"].tolist(), labels.tolist(), threshold)
            metrics[domain] = {}
            for arm in ARMS:
                measure = binary_metrics(scores[arm].tolist(), labels.tolist(), threshold)
                record = {"domain": domain, "arm": arm, "count": len(ids),
                    "bonafide_count": int((labels == 0).sum()),
                    "spoof_count": int((labels == 1).sum()),
                    "auc": measure["auroc"], "eer": measure["eer"],
                    "delta_auc": measure["auroc"] - frozen["auroc"],
                    "delta_eer": measure["eer"] - frozen["eer"],
                    **diagnostics[domain][arm]}
                if not all(math.isfinite(value) for value in record.values()
                           if isinstance(value, float)):
                    raise FloatingPointError("nonfinite O2 development metric")
                table.append(record)
                metrics[domain][arm] = record
            bootstrap.extend(paired_bootstrap(domain, labels, scores))
        classification, qualifying = decision(metrics)
        summary = {"run_id": run_id, "decision": classification,
                   "qualifying_normalized_arms": qualifying,
                   "metrics": metrics, "bootstrap": bootstrap,
                   "score_completion": str(out / "score_completion.json"),
                   "labels_read_only_after_both_domains_scored": True,
                   "target90_labels_or_metrics_accessed": False,
                   "final_heldout_metrics_accessed": False,
                   "new_tta_method_implemented": False}
        write_csv_new(out / "metrics.csv", table)
        write_csv_new(out / "bootstrap.csv", bootstrap)
        write_json_new(out / "summary.json", summary)
        with (out / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("# O2 direction/strength development diagnostic\n\n")
            stream.write("Fixed B128, source-reset, one-step O2 head direction; frozen bias. ")
            stream.write("Selected labels were opened only after both score files passed exact coverage.\n\n")
            stream.write("| Domain | Arm | AUC | EER | ΔAUC | ΔEER | Mean absolute score move | Mean angle (deg) | Mean step norm |\n")
            stream.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in table:
                stream.write(f"| {row['domain']} | {row['arm']} | {row['auc']:.6f} | {row['eer']:.6f} | {row['delta_auc']:+.6f} | {row['delta_eer']:+.6f} | {row['mean_abs_score_delta']:.6f} | {row['mean_head_angle_degrees']:.6f} | {row['mean_effective_step_norm']:.6f} |\n")
            stream.write(f"\nDecision under the fixed contract: **{classification}**. ")
            stream.write("The paired stratified 1,000-draw development intervals are in `bootstrap.csv`. ")
            stream.write("This is a development mechanism test, not final TTA performance.\n")
        return summary
    except BaseException:
        write_json_new(analysis / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    result = run(parser.parse_args().run_id)
    print(json.dumps({"decision": result["decision"],
                      "metrics": result["metrics"]}, indent=2))
