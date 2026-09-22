#!/usr/bin/env python
"""Assemble diagnostics/results/summary.json from the raw diagnostic outputs.

Reads the machine-readable result files written by diag1..diag4 and emits the
single summary object the report (and the terminal hand-off) depend on.  No core
code is touched; this only reads diagnostics/results/*.
"""
import csv
import json

from _diag_common import DIAG_RESULTS, spearman, write_json_atomic, ensure_dirs


def load(name):
    return json.loads((DIAG_RESULTS / name).read_text(encoding="utf-8"))


def main():
    ensure_dirs()
    d1 = load("frozen_vs_k0.json")
    d2 = load("adaptation_diagnostics.json")
    d3 = load("consistency_diagnostics.json")

    # post-hoc candidate metrics (CSV)
    metrics = []
    with (DIAG_RESULTS / "posthoc_candidate_metrics.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics.append({
                "K": int(row["K"]),
                "lr": None if row["lr"] == "" else float(row["lr"]),
                "steps": int(row["steps"]),
                "effective_strength": float(row["effective_strength"]),
                "selection_score": float(row["selection_score"]),
                "entropy": float(row["entropy"]),
                "consistency": float(row["consistency"]),
                "stability": float(row["stability"]),
                "true_EER": float(row["true_EER"]),
                "true_AUC": float(row["true_AUC"]),
                "true_accuracy": float(row["true_accuracy"]),
            })

    eer = [m["true_EER"] for m in metrics]
    neg_eer = [-v for v in eer]

    def cfg(key):
        return d2["configs"][key]

    # --- pick a representative "strongest searched" config for update summary (D) ---
    D = cfg("D_K10_lr1e-4")
    E = cfg("E_K1_lr0.003")

    adaptation_effective = "WEAK"
    # "NO" would require grad/update ~ 0; here grad ~0.098 and R updates linearly
    # with K*lr, but score deltas are ~1e-4..1e-3 and flip rate is exactly 0.

    summary = {
        "schema_version": "0.1.0",
        "diagnostic": "target10_selection_system_diagnosis",
        "frozen_k0_equivalent": d1["frozen_k0_equivalence"] == "PASS",
        "frozen_eer": d1["frozen_eer"],
        "k0_eer": d1["k0_eer"],
        "frozen_k0_max_abs_score_diff": d1["max_abs_score_diff"],

        "adaptation_effective": adaptation_effective,
        "grad_norm_summary": {
            "strongest_searched_D_K10_lr1e-4_mean_grad_norm": D["grad_norm_summary"]["mean_grad_norm"]["mean"],
            "strongest_searched_D_median_grad_norm": D["grad_norm_summary"]["mean_grad_norm"]["median"],
            "strongest_searched_D_max_grad_norm": D["grad_norm_summary"]["max_grad_norm"]["max"],
            "note": "gradients are non-zero (view_variance objective); margin keep gradient ~0",
        },
        "update_norm_summary": {
            "D_total_update_norm_mean": D["update_norm_summary"]["total_update_norm"]["mean"],
            "E_total_update_norm_mean": E["update_norm_summary"]["total_update_norm"]["mean"],
            "D_relative_rho_update_norm_mean": D["update_norm_summary"]["relative_rho_update_norm"]["mean"],
            "note": "update norm scales ~ linearly with K*lr; projection at rho=0.2 never activates",
        },
        "score_delta_summary": {
            "D_mean_abs_score_delta": D["score_delta_summary"]["mean_abs_score_delta"]["mean"],
            "D_max_abs_score_delta": D["score_delta_summary"]["max_abs_score_delta"]["max"],
            "E_mean_abs_score_delta": E["score_delta_summary"]["mean_abs_score_delta"]["mean"],
            "E_max_abs_score_delta": E["score_delta_summary"]["max_abs_score_delta"]["max"],
        },
        "flip_rate_summary": {
            "D_flip_rate": D["flip_rate_summary"]["flip_rate"],
            "E_flip_rate": E["flip_rate_summary"]["flip_rate"],
            "note": "hard prediction flip rate is 0 across all 5 configs on 500 samples",
        },
        "entropy_summary": {
            "mean_entropy_before": D["entropy_summary"]["mean_entropy_before"],
            "D_mean_entropy_after": D["entropy_summary"]["mean_entropy_after"],
            "D_mean_entropy_delta": D["entropy_summary"]["mean_entropy_delta"],
        },
        "objective_summary": {
            "mean_objective_before": D["objective_summary"]["mean_objective_before"],
            "D_mean_objective_after": D["objective_summary"]["mean_objective_after"],
            "mean_keep_before": D["objective_summary"]["mean_keep_before"],
            "D_mean_keep_after": D["objective_summary"]["mean_keep_after"],
        },
        "consistency": {
            "is_hard_label_agreement": True,
            "frozen_consistency_full_3178": d3["frozen_consistency_full_3178"],
            "reference_constant_from_param_search": d3["reference_constant_from_param_search"],
            "flip_rate_view_vs_frozen": d3["configs"]["D_K10_lr1e-4"]["flip_rate_view_vs_frozen"],
            "soft_mean_abs_prob_diff": d3["configs"]["D_K10_lr1e-4"]["soft_mean_abs_prob_diff"],
            "soft_mean_pairwise_js": d3["configs"]["D_K10_lr1e-4"]["soft_mean_pairwise_js"],
        },

        "selection_eer_spearman": spearman([m["selection_score"] for m in metrics], neg_eer),
        "entropy_eer_spearman": spearman([m["entropy"] for m in metrics], neg_eer),
        "consistency_eer_spearman": spearman([m["consistency"] for m in metrics], neg_eer),
        "stability_eer_spearman": spearman([m["stability"] for m in metrics], neg_eer),
        "selection_metric_vs_eer_note": (
            "true EER is identical (0.09857072449482504) across all 13 candidates, "
            "so every Spearman vs EER is undefined (zero variance in EER)"
        ),
        "true_EER_all_candidates": eer[0],

        "effective_strength_score_spearman": spearman(
            [m["effective_strength"] for m in metrics], [m["selection_score"] for m in metrics]),
        "effective_strength_eer_spearman": spearman(
            [m["effective_strength"] for m in metrics], eer),
        "effective_strength_entropy_spearman": spearman(
            [m["effective_strength"] for m in metrics], [m["entropy"] for m in metrics]),
        "effective_strength_stability_spearman": spearman(
            [m["effective_strength"] for m in metrics], [m["stability"] for m in metrics]),

        "diagnosis": "MIXED",
        "diagnosis_primary": (
            "EP updates R (non-zero gradient ~0.098, update norm ~ K*lr), but the effect on "
            "scores (~1e-3) and predictions (flip rate 0) is negligible at the searched lr grid, "
            "so true EER is invariant across all 13 candidates; consistency is a hard-label "
            "agreement metric and therefore has no discriminative power; there is NO evidence of "
            "negative adaptation (EER identical), and the selector's K=0 choice is equivalent to "
            "frozen (PASS)."
        ),
    }

    write_json_atomic(DIAG_RESULTS / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", DIAG_RESULTS / "summary.json")


if __name__ == "__main__":
    main()
