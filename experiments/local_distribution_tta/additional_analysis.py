"""Post-hoc paired local-versus-episodic and buffer reset summaries."""
import argparse
from pathlib import Path

from eptta.evaluation.metrics import binary_metrics
from experiments.local_distribution_tta.analyze import (
    average, complete_scores, read_json, selected_codecfake_labels,
    selected_itw_labels, write_csv_once,
)
from experiments.task_objective_discovery.objective_analysis import paired_bootstrap


COMPARISONS = (("Local-Base B16", "Per-sample Base"),
               ("Local-Base B32", "Per-sample Base"),
               ("Local-O1 B16", "Per-sample O1"),
               ("Local-O1 B32", "Per-sample O1"))


def analyze(run):
    _config, scores, dynamics = complete_scores(run)
    labels = {"in_the_wild": selected_itw_labels(scores["in_the_wild"]),
              "codecfake": selected_codecfake_labels(scores["codecfake"])}
    comparisons, buffers = [], []
    for domain, by_id in scores.items():
        ids = sorted(by_id)
        y = [labels[domain][sid] for sid in ids]
        tau0 = read_json(run / "diagnostics" / (domain + "_provenance.json"))["tau0"]
        if 0 < sum(y) < len(y):
            for local, per_sample in COMPARISONS:
                prior = [by_id[sid][per_sample]["score_after"] for sid in ids]
                later = [by_id[sid][local]["score_after"] for sid in ids]
                a = binary_metrics(prior, y, tau0)
                b = binary_metrics(later, y, tau0)
                comparisons.append({"domain": domain, "local_arm": local,
                                    "per_sample_arm": per_sample,
                                    "delta_AUC_local_minus_per_sample": b["auroc"] - a["auroc"],
                                    "delta_EER_local_minus_per_sample": b["eer"] - a["eer"],
                                    **paired_bootstrap(y, prior, later, tau0, 1000, 2026)})
        for arm in (item[0] for item in COMPARISONS):
            arm_buffers = [r for r in dynamics if r["domain"] == domain and r["arm"] == arm]
            buffers.append({"domain": domain, "arm": arm,
                            "buffer_count": len(arm_buffers),
                            "mean_R_norm": average([r["R_norm"] for r in arm_buffers]),
                            "max_R_norm": max(r["R_norm"] for r in arm_buffers),
                            "mean_objective_reduction": average([
                                r["objective_before"] - r["objective_after"]
                                for r in arm_buffers]),
                            "mean_source_evidence_damage": average([
                                r["source_evidence_damage"] for r in arm_buffers]),
                            "numeric_failures": sum(r["numeric_status"] != "ok" for r in arm_buffers),
                            "first_step_reset_violations": sum(
                                r["trace"][0]["parameter_norm_before"] != 0 for r in arm_buffers)})
    write_csv_once(run / "analysis/local_vs_per_sample.csv", comparisons)
    write_csv_once(run / "analysis/buffer_summary.csv", buffers)
    return comparisons, buffers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    comparisons, buffers = analyze(args.run)
    print("paired_comparisons", len(comparisons), "buffer_summaries", len(buffers))
