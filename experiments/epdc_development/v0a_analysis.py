"""Post-score v0-A audit with exact coverage before label access."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from eptta.evaluation.metrics import binary_metrics
from experiments.multidomain_mechanism.guard_analysis import selected_labels


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("Frozen", "Base Adapt", "Base + Preserve")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def coverage(run):
    config = read_json(run / "run_config.json")
    marker = read_json(run / "diagnostics/score_completion.json")
    if config["scientific_role"] != "mechanism_dev" or marker["status"] != "SCORES_COMPLETE_LABELS_NOT_READ":
        raise ValueError("formal completed scores required")
    result = {}
    for domain, n in config["datasets"].items():
        select = read_json(run / "manifests" / f"{domain}_mechanism_select.json")
        ids = {r["sample_id"] for r in select["records"]}
        if len(ids) != n:
            raise ValueError("select count mismatch")
        rows = [json.loads(line) for line in (run / "scores" / f"{domain}.jsonl").open()]
        if len(rows) != n * 3:
            raise ValueError("score count mismatch")
        by_id = defaultdict(dict)
        for row in rows:
            sid, arm = row["sample_id"], row["arm"]
            if sid not in ids or arm not in ARMS or arm in by_id[sid] or row["domain"] != domain:
                raise ValueError("score coverage mismatch")
            if row["numeric_status"] != "ok" or row["score_before"] != row["score_frozen"]:
                raise ValueError("numeric or Frozen mismatch")
            if any(type(v) is float and not math.isfinite(v) for v in row.values()):
                raise ValueError("non-finite row")
            by_id[sid][arm] = row
        if set(by_id) != ids or any(set(v) != set(ARMS) for v in by_id.values()):
            raise ValueError("exact score coverage mismatch")
        for values in by_id.values():
            frozen = values["Frozen"]["score_after"]
            if any(row["score_frozen"] != frozen for row in values.values()):
                raise ValueError("Frozen reference mismatch")
        result[domain] = by_id
    return config, result


def analyze(run):
    config, scores = coverage(run)  # No audit reader before this line.
    labels = {}
    for domain, by_id in scores.items():
        audit = read_json(ROOT / "experiments/multidomain_mechanism/manifests" /
                          f"{domain}_mechanism_audit.json")
        labels[domain] = selected_labels(domain, audit, set(by_id))
    metrics = []
    for domain, by_id in scores.items():
        ids = sorted(by_id)
        y = [labels[domain][sid] for sid in ids]
        tau0 = read_json(run / "diagnostics" / f"{domain}_provenance.json")["tau0"]
        frozen = [by_id[sid]["Frozen"]["score_after"] for sid in ids]
        for arm in ARMS:
            rows = [by_id[sid][arm] for sid in ids]
            values = [row["score_after"] for row in rows]
            result = {"domain": domain, "arm": arm, "count": len(ids),
                      "bonafide_count": len(y) - sum(y), "spoof_count": sum(y)}
            if 0 < sum(y) < len(y):
                m = binary_metrics(values, y, tau0, frozen_scores=frozen)
                result.update(EER=m["eer"], AUC=m["auroc"], balanced_accuracy=m["balanced_accuracy"],
                              FPR=m["fpr"], FNR=m["fnr"])
            else:
                result.update({k: None for k in ("EER", "AUC", "balanced_accuracy", "FPR", "FNR")})
            result["mean_abs_score_delta"] = sum(abs(a-b) for a,b in zip(values, frozen)) / len(ids)
            for name in ("update_norm", "distance_from_source", "evidence_damage", "decision_order_damage"):
                present = [row[name] for row in rows if name in row]
                result["mean_" + name] = sum(present) / len(present) if present else None
            result["helpful_updates"] = sum(int(f > tau0) != yy and int(a > tau0) == yy
                                             for f,a,yy in zip(frozen, values, y))
            result["harmful_updates"] = sum(int(f > tau0) == yy and int(a > tau0) != yy
                                             for f,a,yy in zip(frozen, values, y))
            metrics.append(result)
    for row in metrics:
        frozen = next(r for r in metrics if r["domain"] == row["domain"] and r["arm"] == "Frozen")
        row["delta_EER"] = row["EER"] - frozen["EER"] if row["EER"] is not None else None
        row["delta_AUC"] = row["AUC"] - frozen["AUC"] if row["AUC"] is not None else None
    folder = run / "analysis"
    with (folder / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    macro = []
    for arm in ARMS:
        chosen = [r for r in metrics if r["arm"] == arm]
        row = {"arm": arm, "domains": len(chosen),
               "ranking_metric_domains": sum(r["EER"] is not None for r in chosen)}
        for key in ("EER", "AUC", "balanced_accuracy", "mean_abs_score_delta",
                    "mean_update_norm", "mean_distance_from_source", "mean_evidence_damage",
                    "mean_decision_order_damage", "helpful_updates", "harmful_updates"):
            present = [r[key] for r in chosen if r[key] is not None]
            row[key] = sum(present) / len(present) if present else None
        macro.append(row)
    with (folder / "macro.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    summary = {"status": "PARTIAL_THREE_DOMAIN_V0A_DEVELOPMENT", "method_locked": False,
               "target90_accessed": False, "final_holdout_labels_accessed": False,
               "not_run_domains": ["codecfake", "wavefake"],
               "ranking_metric_domains": sum(r["EER"] is not None for r in metrics if r["arm"] == "Frozen"),
               "per_domain": metrics, "macro": macro}
    write_once(folder / "summary.json", summary)
    lines = ["# EPDC v0-A development result", "",
             "Three cached domains scored; only In-the-Wild has both classes. Codecfake/WaveFake NOT_RUN.", "",
             "| Domain | Arm | EER | AUC | Mean evidence damage | Mean order damage | Helpful | Harmful |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics:
        fmt = lambda x: "NA" if x is None else f"{x:.6f}"
        lines.append(f"| {r['domain']} | {r['arm']} | {fmt(r['EER'])} | {fmt(r['AUC'])} | {fmt(r['mean_evidence_damage'])} | {fmt(r['mean_decision_order_damage'])} | {r['helpful_updates']} | {r['harmful_updates']} |")
    lines += ["", "EER/AUC macro values have one-domain coverage; no cross-domain ranking claim."]
    with (folder / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if run.parent != (ROOT / "experiments/epdc_development/results").resolve():
        raise ValueError("run must be a direct child of EPDC development results")
    print(json.dumps(analyze(run), allow_nan=False)[:1000])


if __name__ == "__main__":
    main()
