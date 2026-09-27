"""Post-score selected-development task audit; no adaptation is performed here."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eptta.evaluation.metrics import binary_metrics
from experiments.head_capacity_geometry.resources import ROOT, select_rows
from experiments.head_capacity_geometry.supervised_labels import load_labels
from experiments.head_tta.mixture import ALPHAS


HERE = Path(__file__).resolve().parent
UPPER = {"itw": {"auc": .9731589944070336, "eer": .08529155787641429},
         "wavefake": {"auc": .9517107009887695, "eer": .1142578125}}


def read_checked_scores(out: Path):
    completion = json.loads((out / "score_completion.json").read_text())
    if not completion.get("complete") or completion.get("smoke") or \
            completion.get("target_labels_loaded_in_worker") is not False:
        raise ValueError("full label-free scores must complete before audit")
    by_domain = {}
    for domain in ("itw", "wavefake"):
        ids, _, _ = select_rows(domain)
        rows = [json.loads(line) for line in (out / "scores" / f"{domain}.jsonl").open()]
        if ([row.get("sample_id") for row in rows] != ids or
                len(set(ids)) != len(ids) or
                completion["domains"][domain]["count"] != len(ids) or
                completion["domains"][domain]["unique_ids"] != len(ids)):
            raise ValueError(f"{domain}: exact selected score coverage failure")
        for row in rows:
            if set(row) != {"sample_id", "domain", "buffer_index", "score_frozen",
                            "q_spoof", "reliability", "numeric_status",
                            *(f"score_alpha_{alpha}" for alpha in ALPHAS)}:
                raise ValueError("worker score schema contains missing or extra field")
            if row["domain"] != domain or row["numeric_status"] != "ok" or \
                    any(not np.isfinite(row[key]) for key in ["score_frozen",
                        "q_spoof", "reliability", *(f"score_alpha_{a}" for a in ALPHAS)]):
                raise ValueError("nonfinite or misidentified worker score")
        by_domain[domain] = (ids, rows)
    return by_domain


def analyze(run_id):
    out = HERE / "results" / run_id
    by_domain = read_checked_scores(out)  # all score coverage before any labels open
    provenance = json.loads((out / "provenance.json").read_text())
    table = []
    summary = {"run_id": run_id, "role": "DEVELOPMENT_ONLY_POST_HOC_LABEL_AUDIT",
               "supervised_linear_is_upper_bound_not_tta": True, "domains": {},
               "target90_labels_accessed": False, "target90_metrics_accessed": False,
               "final_heldout_metrics_accessed": False, "method_lock": False}
    for domain, (ids, rows) in by_domain.items():
        labels = load_labels(domain, ids)
        truth = labels.astype(int).tolist()
        threshold = provenance[domain]["tau0"]
        method_scores = {"Frozen": [row["score_frozen"] for row in rows]}
        method_scores.update({f"H-UA1_alpha_{alpha}":
            [row[f"score_alpha_{alpha}"] for row in rows] for alpha in ALPHAS})
        baseline = None
        domain_summary = {"count": len(ids), "class_counts": {
            "bonafide": int((labels == 0).sum()), "spoof": int((labels == 1).sum())},
            "supervised_linear_upper_bound": UPPER[domain], "methods": {}}
        for name, scores in method_scores.items():
            result = binary_metrics(scores, truth, threshold,
                                    None if name == "Frozen" else method_scores["Frozen"])
            if name == "Frozen":
                baseline = result
            delta_auc = result["auroc"]-baseline["auroc"]
            eer_improvement = baseline["eer"]-result["eer"]
            ratio = (delta_auc/(UPPER[domain]["auc"]-baseline["auroc"])
                     if name != "Frozen" and delta_auc > 0 else None)
            compact = {"auc": result["auroc"], "eer": result["eer"],
                       "balanced_accuracy": result["balanced_accuracy"],
                       "fpr": result["fpr"], "fnr": result["fnr"],
                       "delta_auc": delta_auc, "eer_improvement": eer_improvement,
                       "recovery_ratio_auc": ratio}
            domain_summary["methods"][name] = compact
            table.append({"domain": domain, "method": name, "count": len(ids), **compact})
        summary["domains"][domain] = domain_summary
    passing = []
    for alpha in ALPHAS:
        name = f"H-UA1_alpha_{alpha}"
        itw = summary["domains"]["itw"]["methods"][name]
        wave = summary["domains"]["wavefake"]["methods"][name]
        required_itw = itw["delta_auc"] >= .005 or itw["eer_improvement"] >= .005
        required_wave = wave["delta_auc"] >= .01 or wave["eer_improvement"] >= .01
        common = (itw["delta_auc"] > 0 and wave["delta_auc"] > 0) or \
                 (itw["eer_improvement"] > 0 and wave["eer_improvement"] > 0)
        no_collapse = all(item[metric] >= -.005 for item in (itw, wave)
                          for metric in ("delta_auc", "eer_improvement"))
        if required_itw and required_wave and common and no_collapse:
            passing.append(alpha)
    summary["promotion_conditions"] = {
        "itw_auc_gain_or_eer_improvement": [.005, .005],
        "wavefake_auc_gain_or_eer_improvement": [.01, .01],
        "common_positive_direction": True, "each_other_metric_minimum": -.005}
    summary["passing_alphas"] = passing
    summary["decision"] = ("PROMOTE_HEAD_ADAPTATION" if passing else
                           "HEAD_ADAPTATION_NOT_YET_ACTIONABLE")
    summary["promoted_candidate"] = f"H-UA1_alpha_{min(passing)}" if passing else None
    analysis = out / "analysis"
    with (analysis / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (analysis / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (analysis / "report.md").open("x") as stream:
        stream.write("# H-UA1 selected-development audit\n\n")
        stream.write("Supervised Linear is a development upper bound, not a TTA result.\n\n")
        stream.write("| Domain | Arm | AUC | EER | ΔAUC | EER improvement | AUC recovery |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in table:
            stream.write(f"| {row['domain']} | {row['method']} | {row['auc']:.6f} | "
                         f"{row['eer']:.6f} | {row['delta_auc']:+.6f} | "
                         f"{row['eer_improvement']:+.6f} | "
                         f"{row['recovery_ratio_auc'] if row['recovery_ratio_auc'] is not None else 'NA'} |\n")
        stream.write(f"\nDecision: **{summary['decision']}**; candidate: "
                     f"{summary['promoted_candidate'] or 'NONE'}.\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    print(json.dumps(analyze(parser.parse_args().run_id), indent=2))
