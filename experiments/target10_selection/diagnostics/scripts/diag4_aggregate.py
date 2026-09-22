#!/usr/bin/env python
"""Diagnostic 4+5 (aggregator): true supervised metrics, rank comparison, K*lr analysis.

Reads every ``posthoc_group_*.json`` written by the Diag-4 workers, joins the 13
candidates, computes TRUE EER/AUC/accuracy against the raw target10 audit labels
(POST-HOC ONLY), and produces:

  posthoc_candidate_metrics.csv
  posthoc_rank_comparison.csv
  effective_strength_analysis.csv
  posthoc_correlation.json
"""
import csv
import json
import math

from _diag_common import (DIAG_RESULTS, TARGET10, candidates, load_context,
                          compute_metrics, spearman, pearson, write_json_atomic, ensure_dirs)


def main():
    ensure_dirs()
    _bundle, _resources, _meta, _cache, _features, threshold = load_context()

    group_files = sorted(DIAG_RESULTS.glob("posthoc_group_*.json"))
    if not group_files:
        raise SystemExit("no posthoc_group_*.json found; run diag4_posthoc.py first")

    rows = []
    for path in group_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(doc["candidates"])

    expected = list(candidates())
    got_keys = sorted((r["K"], r["lr"], r["steps"]) for r in rows)
    exp_keys = sorted((c["K"], c["lr"], c["steps"]) for c in expected)
    if got_keys != exp_keys:
        raise SystemExit(f"posthoc candidates mismatch: got {got_keys}, expected {exp_keys}")

    # POST-HOC labels (raw audit manifest; NOT used in the original selection).
    raw = json.loads(TARGET10.read_text(encoding="utf-8"))
    labels = {r["sample_id"]: int(r["label"]) for r in raw["records"]}

    metrics = []
    for row in rows:
        scores = row["per_sample_scores"]
        m = compute_metrics(scores, labels, threshold)
        metrics.append({
            "K": row["K"], "lr": row["lr"], "steps": row["steps"],
            "selection_score": row["selection_score"], "entropy": row["entropy"],
            "consistency": row["consistency"], "stability": row["stability"],
            "true_EER": m["EER"], "true_AUC": m["AUC"], "true_accuracy": m["accuracy"],
            "neg_EER": -m["EER"],
            "effective_strength": 0.0 if row["K"] == 0 else row["K"] * row["lr"],
        })
    # deterministic order: by (K, lr)
    metrics.sort(key=lambda r: (r["K"], r["lr"] if r["lr"] is not None else -1.0))

    csv_path = DIAG_RESULTS / "posthoc_candidate_metrics.csv"
    fieldnames = ["K", "lr", "steps", "effective_strength", "selection_score", "entropy",
                  "consistency", "stability", "true_EER", "true_AUC", "true_accuracy"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics)

    # ---- correlations: unsupervised metric vs true (negative) EER ----
    corr = {}
    for metric_name in ("selection_score", "entropy", "consistency", "stability"):
        xs = [m[metric_name] for m in metrics]
        corr[f"{metric_name}_vs_negEER_spearman"] = spearman(xs, [m["neg_EER"] for m in metrics])
        corr[f"{metric_name}_vs_negEER_pearson"] = pearson(xs, [m["neg_EER"] for m in metrics])
        corr[f"{metric_name}_vs_EER_spearman"] = spearman(xs, [m["true_EER"] for m in metrics])

    # ---- rank comparison ----
    ranked_by_score = sorted(metrics, key=lambda m: (-m["selection_score"], m["K"],
                                                     (m["lr"] if m["lr"] is not None else 0.0)))
    ranked_by_eer = sorted(metrics, key=lambda m: (m["true_EER"], m["K"],
                                                   (m["lr"] if m["lr"] is not None else 0.0)))
    score_rank = {id(m): i + 1 for i, m in enumerate(ranked_by_score)}
    eer_rank = {id(m): i + 1 for i, m in enumerate(ranked_by_eer)}
    rank_rows = []
    for m in metrics:
        rank_rows.append({
            "K": m["K"], "lr": m["lr"], "steps": m["steps"],
            "selection_score": m["selection_score"], "true_EER": m["true_EER"],
            "selection_score_rank": score_rank[id(m)], "true_EER_rank": eer_rank[id(m)],
            "rank_delta": score_rank[id(m)] - eer_rank[id(m)],
        })
    rank_path = DIAG_RESULTS / "posthoc_rank_comparison.csv"
    with rank_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["K", "lr", "steps", "selection_score", "true_EER",
                                          "selection_score_rank", "true_EER_rank", "rank_delta"])
        w.writeheader()
        w.writerows(rank_rows)

    # ---- effective strength (K*lr) analysis ----
    es_rows = sorted(metrics, key=lambda m: m["effective_strength"])
    es_path = DIAG_RESULTS / "effective_strength_analysis.csv"
    with es_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["K", "lr", "effective_strength", "selection_score",
                                          "entropy", "stability", "consistency", "true_EER"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(es_rows)

    es = [m["effective_strength"] for m in metrics]
    es_corr = {
        "effective_strength_vs_selection_score_spearman": spearman(es, [m["selection_score"] for m in metrics]),
        "effective_strength_vs_true_EER_spearman": spearman(es, [m["true_EER"] for m in metrics]),
        "effective_strength_vs_entropy_spearman": spearman(es, [m["entropy"] for m in metrics]),
        "effective_strength_vs_stability_spearman": spearman(es, [m["stability"] for m in metrics]),
    }

    summary = {
        "schema_version": "0.1.0",
        "diagnostic": "posthoc_metrics_and_correlation",
        "labels_note": "POST-HOC ONLY; target10 labels never entered the original selection",
        "candidate_count": len(metrics),
        "candidates": metrics,
        "correlations": corr,
        "effective_strength_correlations": es_corr,
        "rank_comparison": rank_rows,
    }
    write_json_atomic(DIAG_RESULTS / "posthoc_correlation.json", summary)

    print(json.dumps({"correlations": corr, "effective_strength_correlations": es_corr},
                     ensure_ascii=False, indent=2))
    print("wrote", csv_path)
    print("wrote", rank_path)
    print("wrote", es_path)
    print("wrote", DIAG_RESULTS / "posthoc_correlation.json")


if __name__ == "__main__":
    main()
