"""Post-score, selected-development-label audit for the fixed local-context study."""
import argparse
import csv
import json
import math
from pathlib import Path

from eptta.evaluation.metrics import binary_metrics
from experiments.local_distribution_tta.run_study import ARMS
from experiments.task_objective_discovery.objective_analysis import paired_bootstrap


ROOT = Path(__file__).resolve().parents[2]
CODEC_PROTOCOL = Path("/media/dell/data/fakedata/Codecfake_Xie/extracted/label/label/dev.txt")
ITW_SELECTED_AUDIT = ROOT.parent.parent / "experiments/multidomain_mechanism/audit_labels/in_the_wild.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv_once(path, rows):
    if not rows:
        raise ValueError("empty analysis table")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_scores(run):
    """Validate everything that can be checked without opening selected labels."""
    config = read_json(run / "run_config.json")
    marker = read_json(run / "diagnostics/score_completion.json")
    if (config.get("role") != "two_domain_development_scores_no_labels" or
            config.get("target_labels_read") is not False or
            marker.get("status") != "SCORES_COMPLETE_LABELS_NOT_READ" or
            marker.get("audit_labels_read") is not False or
            config.get("datasets") != {"in_the_wild": 512, "codecfake": 512} or
            tuple(config.get("arms", ())) != ARMS):
        raise ValueError("complete fixed two-domain score marker required")
    result, dynamics = {}, []
    for domain in ("in_the_wild", "codecfake"):
        manifest = read_json(run / "manifests" / (domain + "_mechanism_select.json"))
        ids = [row["sample_id"] for row in manifest["records"]]
        if (manifest.get("role") != "mechanism_select" or len(ids) != 512 or
                len(set(ids)) != 512 or ids != sorted(ids)):
            raise ValueError("fixed select coverage/order mismatch")
        score_rows = [json.loads(line) for line in (run / "scores" / (domain + ".jsonl")).open()]
        if len(score_rows) != 512 * len(ARMS):
            raise ValueError("incomplete seven-arm scores")
        by_id = {sid: {} for sid in ids}
        for row in score_rows:
            sid, arm = row["sample_id"], row["arm"]
            if (sid not in by_id or arm not in ARMS or arm in by_id[sid] or
                    row["domain"] != domain or row["numeric_status"] != "ok" or
                    any(type(value) is float and not math.isfinite(value) for value in row.values())):
                raise ValueError("score coverage/numeric failure")
            by_id[sid][arm] = row
        for sid in ids:
            arms = by_id[sid]
            if set(arms) != set(ARMS):
                raise ValueError("missing arm")
            frozen = arms["Frozen"]["score_after"]
            if any(arms[arm]["score_frozen"] != frozen for arm in ARMS):
                raise ValueError("different Frozen reference")
        local_rows = [json.loads(line) for line in
                      (run / "diagnostics" / (domain + "_buffer_dynamics.jsonl")).open()]
        expected_count = 2 * (512 // 16 + 512 // 32)
        if len(local_rows) != expected_count:
            raise ValueError("buffer count mismatch")
        for arm in ARMS[3:]:
            size = int(arm.rsplit("B", 1)[1])
            members = [row for row in local_rows if row["arm"] == arm]
            if len(members) != 512 // size:
                raise ValueError("buffer arm count mismatch")
            for index, row in enumerate(members):
                expected = ids[index * size:(index + 1) * size]
                if (row["buffer_index"] != index or row["sample_ids"] != expected or
                        row["buffer_size"] != size or row["numeric_status"] != "ok" or
                        len(row["trace"]) != 5 or
                        row["trace"][0]["parameter_norm_before"] != 0):
                    raise ValueError("buffer membership or reset violation")
        dynamics.extend(local_rows)
        result[domain] = by_id
    return config, result, dynamics


def selected_itw_labels(ids):
    doc = read_json(ITW_SELECTED_AUDIT)
    if (doc.get("role") != "selected_only_post_score_audit" or
            doc.get("dataset_id") != "in_the_wild" or doc.get("count") != 512):
        raise ValueError("wrong selected-only ITW audit")
    labels = {row["sample_id"]: row["label"] for row in doc["records"]}
    if len(labels) != 512 or set(labels) != set(ids) or any(type(v) is not int or v not in (0, 1)
                                                       for v in labels.values()):
        raise ValueError("selected ITW audit coverage mismatch")
    return labels


def selected_codecfake_labels(ids):
    """Read official dev protocol only after complete_scores returned successfully."""
    labels = {}
    wanted = set(ids)
    with CODEC_PROTOCOL.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 3:
                raise ValueError("unexpected Codecfake dev protocol schema")
            sid = "dev/" + fields[0]
            if sid in wanted:
                if sid in labels or fields[1] not in ("real", "fake"):
                    raise ValueError("Codecfake selected label duplicate/unknown")
                labels[sid] = 0 if fields[1] == "real" else 1
    if set(labels) != wanted:
        raise ValueError("Codecfake selected label coverage mismatch")
    return labels


def average(values):
    return sum(values) / len(values) if values else None


def analyze(run):
    config, scores, dynamics = complete_scores(run)  # Label files are closed until this succeeds.
    labels_by_domain = {"in_the_wild": selected_itw_labels(scores["in_the_wild"]),
                        "codecfake": selected_codecfake_labels(scores["codecfake"])}
    metric_rows, bootstrap_rows = [], []
    for domain, by_id in scores.items():
        ids = sorted(by_id)
        labels = [labels_by_domain[domain][sid] for sid in ids]
        tau0 = read_json(run / "diagnostics" / (domain + "_provenance.json"))["tau0"]
        frozen = [by_id[sid]["Frozen"]["score_after"] for sid in ids]
        frozen_metric = binary_metrics(frozen, labels, tau0) if 0 < sum(labels) < len(labels) else None
        for arm in ARMS:
            arm_rows = [by_id[sid][arm] for sid in ids]
            adapted = [row["score_after"] for row in arm_rows]
            metric = binary_metrics(adapted, labels, tau0, frozen_scores=frozen) if frozen_metric else None
            entry = {"domain": domain, "arm": arm, "count": 512,
                     "bonafide_count": len(labels) - sum(labels), "spoof_count": sum(labels),
                     "EER": metric["eer"] if metric else None,
                     "AUC": metric["auroc"] if metric else None,
                     "balanced_accuracy": metric["balanced_accuracy"] if metric else None,
                     "FPR": metric["fpr"] if metric else None,
                     "FNR": metric["fnr"] if metric else None,
                     "delta_EER_vs_Frozen": metric["eer"] - frozen_metric["eer"] if metric else None,
                     "delta_AUC_vs_Frozen": metric["auroc"] - frozen_metric["auroc"] if metric else None,
                     "helpful_flips": metric["helpful_flips"] if metric else None,
                     "harmful_flips": metric["harmful_flips"] if metric else None,
                     "mean_abs_score_delta": average([abs(a - f) for a, f in zip(adapted, frozen)]),
                     "mean_update_norm": average([r["update_norm"] for r in arm_rows]),
                     "mean_source_evidence_damage": average([r["source_evidence_damage"] for r in arm_rows]),
                     "mean_runtime_seconds": average([r["runtime_seconds"] for r in arm_rows]),
                     "numeric_failures": 0}
            metric_rows.append(entry)
            if metric and arm != "Frozen":
                ci = paired_bootstrap(labels, frozen, adapted, tau0, 1000, 2026)
                bootstrap_rows.append({"domain": domain, "arm": arm, "replicates": 1000,
                                       "seed": 2026, **ci})
    compact_dynamics = [{key: value for key, value in row.items() if key not in ("sample_ids", "trace")}
                        for row in dynamics]
    analysis = run / "analysis"
    write_csv_once(analysis / "metrics.csv", metric_rows)
    write_csv_once(analysis / "bootstrap.csv", bootstrap_rows)
    write_csv_once(analysis / "buffer_dynamics.csv", compact_dynamics)
    summary = {"status": "DEVELOPMENT_ANALYSIS_COMPLETE", "run_id": config["run_id"],
               "role": "development_only", "target90_metrics_accessed": False,
               "final_holdout_metrics_accessed": False,
               "selected_audit_opened_after_exact_score_validation": True,
               "metrics": metric_rows, "bootstrap": bootstrap_rows,
               "buffer_count": len(dynamics), "promoted_candidate": "NONE_PENDING_INTERPRETATION"}
    write_json_once(analysis / "summary.json", summary)
    with (analysis / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("# Local distribution TTA development study\n\n")
        stream.write("Selected development labels were opened only after exact two-domain seven-arm score and buffer checks.\n\n")
        stream.write("| Domain | Arm | EER | AUC | ΔAUC vs Frozen | Mean R norm | Source damage |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in metric_rows:
            stream.write(f"| {row['domain']} | {row['arm']} | {row['EER']} | {row['AUC']} | "
                         f"{row['delta_AUC_vs_Frozen']} | {row['mean_update_norm']} | "
                         f"{row['mean_source_evidence_damage']} |\n")
        stream.write("\nIntervals in `bootstrap.csv` are paired, stratified, 1,000-draw development diagnostics only.\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    print(result["status"], args.run)
