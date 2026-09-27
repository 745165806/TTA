"""Post-score selected-label audit with paired development-only uncertainty."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from eptta.evaluation.metrics import binary_metrics
from experiments.multidomain_mechanism.guard_analysis import selected_labels


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ("Frozen", "ep_no_keep", "O1", "O2", "O3")


def pa_dev_labels(audit, ids):
    """Open official PA *dev* labels only after complete_scores has succeeded."""
    if audit.get("role") != "mechanism_audit" or audit.get("dataset_id") != "asv2019_pa_dev" or \
            {row["sample_id"] for row in audit["records"]} != ids:
        raise ValueError("PA dev audit/score coverage mismatch")
    values = {}
    with Path(audit["label_source_ref"]).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("unexpected PA dev protocol row")
            sample_id = fields[1]
            if sample_id in ids:
                if sample_id in values or fields[4] not in ("bonafide", "spoof"):
                    raise ValueError("duplicate or unknown PA dev label")
                values[sample_id] = 0 if fields[4] == "bonafide" else 1
    if set(values) != ids:
        raise ValueError("PA dev selected audit label coverage mismatch")
    return values


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def complete_scores(run):
    config = read_json(run / "run_config.json")
    marker = read_json(run / "diagnostics/score_completion.json")
    if config["role"] != "mechanism_dev_scores_no_labels" or \
            marker["status"] != "SCORES_COMPLETE_LABELS_NOT_READ":
        raise ValueError("complete formal scores are required before audit")
    result = {}
    for domain, count in config["datasets"].items():
        manifest = read_json(run / "manifests" / (domain + "_mechanism_select.json"))
        ids = {row["sample_id"] for row in manifest["records"]}
        if len(ids) != count:
            raise ValueError("select count mismatch")
        rows = [json.loads(line) for line in (run / "scores" / (domain + ".jsonl")).open()]
        if len(rows) != count * len(ARMS):
            raise ValueError("incomplete scores")
        by_id = defaultdict(dict)
        for row in rows:
            sid, arm = row["sample_id"], row["arm"]
            if sid not in ids or arm not in ARMS or arm in by_id[sid] or row["domain"] != domain:
                raise ValueError("score coverage mismatch")
            if row["numeric_status"] != "ok" or row["score_before"] != row["score_frozen"]:
                raise ValueError("numeric or Frozen reference mismatch")
            if any(type(v) is float and not math.isfinite(v) for v in row.values()):
                raise ValueError("nonfinite score diagnostic")
            by_id[sid][arm] = row
        if set(by_id) != ids or any(set(arms) != set(ARMS) for arms in by_id.values()):
            raise ValueError("exact arm coverage mismatch")
        for arms in by_id.values():
            frozen = arms["Frozen"]["score_after"]
            if any(value["score_frozen"] != frozen for value in arms.values()):
                raise ValueError("Frozen scores differ between arms")
        result[domain] = by_id
    return config, result


def paired_bootstrap(y, frozen, adapted, tau0, reps, seed):
    y = np.asarray(y)
    frozen, adapted = np.asarray(frozen), np.asarray(adapted)
    rng = np.random.default_rng(seed)
    class0 = np.flatnonzero(y == 0)
    class1 = np.flatnonzero(y == 1)
    delta_eer, delta_auc = [], []
    for _ in range(reps):
        picked = np.concatenate((rng.choice(class0, len(class0), replace=True),
                                 rng.choice(class1, len(class1), replace=True)))
        f = binary_metrics(frozen[picked].tolist(), y[picked].tolist(), tau0)
        a = binary_metrics(adapted[picked].tolist(), y[picked].tolist(), tau0)
        delta_eer.append(a["eer"] - f["eer"])
        delta_auc.append(a["auroc"] - f["auroc"])
    return {"delta_EER_ci_low": float(np.quantile(delta_eer, .025)),
            "delta_EER_ci_high": float(np.quantile(delta_eer, .975)),
            "delta_AUC_ci_low": float(np.quantile(delta_auc, .025)),
            "delta_AUC_ci_high": float(np.quantile(delta_auc, .975))}


def mean(values):
    return sum(values) / len(values) if values else None


def analyze(run):
    config, scores = complete_scores(run)  # No label reader can run before this line.
    metrics, intervals = [], []
    for domain, by_id in scores.items():
        audit_root = (HERE / "manifests") if domain == "asv2019_pa_dev" else \
            (ROOT / "experiments/multidomain_mechanism/manifests")
        audit = read_json(audit_root / (domain + "_mechanism_audit.json"))
        labels = (pa_dev_labels(audit, set(by_id)) if domain == "asv2019_pa_dev" else
                  selected_labels(domain, audit, set(by_id)))
        ids = sorted(by_id)
        y = [labels[sid] for sid in ids]
        tau0 = read_json(run / "diagnostics" / (domain + "_provenance.json"))["tau0"]
        frozen = [by_id[sid]["Frozen"]["score_after"] for sid in ids]
        for arm in ARMS:
            rows = [by_id[sid][arm] for sid in ids]
            values = [row["score_after"] for row in rows]
            entry = {"domain": domain, "arm": arm, "count": len(ids),
                     "bonafide_count": len(y) - sum(y), "spoof_count": sum(y)}
            if 0 < sum(y) < len(y):
                raw = binary_metrics(values, y, tau0, frozen_scores=frozen)
                entry.update(EER=raw["eer"], AUC=raw["auroc"],
                             balanced_accuracy=raw["balanced_accuracy"],
                             FPR=raw["fpr"], FNR=raw["fnr"])
            else:
                entry.update({key: None for key in ("EER", "AUC", "balanced_accuracy", "FPR", "FNR")})
            entry["mean_abs_score_delta"] = mean([abs(a - b) for a, b in zip(values, frozen)])
            entry["mean_update_norm"] = mean([row["update_norm"] for row in rows])
            entry["mean_source_evidence_damage"] = mean([row["source_evidence_damage"] for row in rows])
            entry["mean_objective_reduction"] = mean([
                row["objective_before"] - row["objective_after"] for row in rows
                if row["objective_before"] is not None])
            entry["numeric_failures"] = sum(row["numeric_status"] != "ok" for row in rows)
            entry["mean_runtime_seconds"] = mean([row["runtime_seconds"] for row in rows])
            entry["helpful_flips"] = sum(int(f > tau0) != label and int(a > tau0) == label
                                         for f, a, label in zip(frozen, values, y))
            entry["harmful_flips"] = sum(int(f > tau0) == label and int(a > tau0) != label
                                         for f, a, label in zip(frozen, values, y))
            metrics.append(entry)
        base = next(row for row in metrics if row["domain"] == domain and row["arm"] == "Frozen")
        for row in [value for value in metrics if value["domain"] == domain]:
            row["delta_EER"] = row["EER"] - base["EER"] if row["EER"] is not None else None
            row["delta_AUC"] = row["AUC"] - base["AUC"] if row["AUC"] is not None else None
            if row["arm"] != "Frozen" and row["EER"] is not None:
                values = [by_id[sid][row["arm"]]["score_after"] for sid in ids]
                intervals.append({"domain": domain, "arm": row["arm"],
                                  "replicates": config["parameters"]["bootstrap_replicates"],
                                  **paired_bootstrap(y, frozen, values, tau0,
                                     config["parameters"]["bootstrap_replicates"],
                                     config["parameters"]["bootstrap_seed"])})
    with (run / "analysis/metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (run / "analysis/paired_bootstrap.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["domain", "arm", "replicates", "delta_EER_ci_low",
                                                      "delta_EER_ci_high", "delta_AUC_ci_low", "delta_AUC_ci_high"])
        writer.writeheader()
        writer.writerows(intervals)
    ranking_domains = sorted({row["domain"] for row in metrics if row["EER"] is not None})
    summary = {"status": "PARTIAL_ONE_DOMAIN_DEVELOPMENT" if len(ranking_domains) < 2 else "MULTIDOMAIN_DEVELOPMENT",
               "ranking_domains": ranking_domains, "promotion_eligible": False,
               "promotion_reason": ("at least two independent two-class development domains are required"
                                    if len(ranking_domains) < 2 else
                                    "promotion requires explicit cross-domain gain, uncertainty and damage interpretation"),
               "per_domain": metrics, "paired_bootstrap": intervals,
               "target90_labels_or_metrics_read": False,
               "final_holdout_labels_read_this_stage": False}
    write_new(run / "analysis/summary.json", summary)
    lines = ["# Task objective discovery development result", "",
             "All selected scores passed exact coverage before the selected audit labels were opened.",
             "Bootstrap intervals are development-only paired stratified resamples, not final claims.", "",
             "| Domain | Arm | EER | AUC | ΔEER | ΔAUC | Damage | Helpful | Harmful |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        fmt = lambda value: "NA" if value is None else "%.6f" % value
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %d | %d |" % (
            row["domain"], row["arm"], fmt(row["EER"]), fmt(row["AUC"]),
            fmt(row["delta_EER"]), fmt(row["delta_AUC"]),
            fmt(row["mean_source_evidence_damage"]), row["helpful_flips"], row["harmful_flips"]))
    lines.extend(("", "No O1/O2/O3 objective is promoted automatically; apply the predeclared cross-domain criteria.",
                  "Codecfake fixed 512 includes non-16-kHz waveforms incompatible with production extraction; "
                  "WaveFake lacks a compatible Parquet reader; ASV2021 mechanism groups are one-class. "
                  "ASVspoof2019 PA dev, if present, is an auxiliary domain and is not relabeled as ASVspoof2021."))
    with (run / "analysis/report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if run.parent != (HERE / "results").resolve():
        raise ValueError("run must be a direct task-objective result child")
    print(json.dumps(analyze(run), allow_nan=False)[:1200])
