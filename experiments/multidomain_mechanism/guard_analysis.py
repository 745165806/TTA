"""Post-score audit for guard capacity. Never imported by the adaptation worker."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from eptta.evaluation.metrics import binary_metrics

from experiments.multidomain_mechanism.guard_worker import ARMS, ROOT, SETTINGS

AUDIT_LABELS = ROOT / "experiments/multidomain_mechanism/audit_labels"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def complete_scores(run):
    """Finish exact ID, arm, status, finite and Frozen checks before audit opens."""
    config = read_json(run / "run_config.json")
    summary = read_json(run / "analysis/summary.json")
    if summary.get("status") != "SCORES_COMPLETE_LABELS_NOT_READ" or summary.get("audit_labels_read") is not False:
        raise ValueError("score completion marker absent")
    expected_arms = {("Frozen", "Frozen")}
    expected_arms.update((setting, arm) for setting in SETTINGS for arm in ARMS)
    all_rows = {}
    for domain, count in config["datasets"].items():
        select = read_json(run / "manifests" / f"{domain}_mechanism_select.json")
        ids = [row["sample_id"] for row in select["records"][:count]]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("select coverage mismatch")
        path = run / "scores" / f"{domain}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(rows) != count * len(expected_arms):
            raise ValueError("score row count mismatch")
        by_id = defaultdict(dict)
        for row in rows:
            sid = row["sample_id"]
            key = (row["setting"], row["arm"])
            if sid not in ids or key not in expected_arms or key in by_id[sid]:
                raise ValueError("score ID/arm coverage mismatch")
            if row["domain"] != domain or row.get("numeric_status") != "ok":
                raise ValueError("score provenance or numeric status mismatch")
            if any(type(v) is float and not math.isfinite(v) for v in row.values()):
                raise ValueError("non-finite score or diagnostic")
            if row["score_before"] != row["score_frozen"]:
                raise ValueError("Frozen reference mismatch")
            by_id[sid][key] = row
        if set(by_id) != set(ids) or any(set(values) != expected_arms for values in by_id.values()):
            raise ValueError("score exact coverage mismatch")
        for values in by_id.values():
            frozen = values[("Frozen", "Frozen")]["score_after"]
            if any(row["score_frozen"] != frozen for row in values.values()):
                raise ValueError("inconsistent Frozen reference")
        all_rows[domain] = by_id
    return config, all_rows


def selected_labels(domain, audit, ids):
    """Read only selected audit records after complete_scores has returned."""
    if audit.get("role") != "mechanism_audit" or audit.get("dataset_id") != domain:
        raise ValueError("wrong audit manifest")
    if {row["sample_id"] for row in audit["records"]} != ids:
        raise ValueError("audit/select ID mismatch")
    selected = read_json(AUDIT_LABELS / f"{domain}.json")
    if (selected.get("role") != "selected_only_post_score_audit" or
            selected.get("dataset_id") != domain or selected.get("count") != len(ids)):
        raise ValueError("selected-only audit provenance mismatch")
    rows = selected["records"]
    labels = {row["sample_id"]: row["label"] for row in rows}
    if len(labels) != len(rows):
        raise ValueError("duplicate selected-only audit ID")
    if set(labels) != ids or any(type(v) is not int or v not in (0, 1) for v in labels.values()):
        raise ValueError("canonical label coverage mismatch")
    return labels


def mean(values):
    return sum(values) / len(values) if values else None


def metric_row(domain, setting, arm, records, labels, tau0):
    ids = sorted(records)
    scores = [records[sid]["score_after"] for sid in ids]
    frozen = [records[sid]["score_frozen"] for sid in ids]
    y = [labels[sid] for sid in ids]
    row = {"domain": domain, "setting": setting, "arm": arm, "count": len(ids),
           "bonafide_count": len(y) - sum(y), "spoof_count": sum(y)}
    if 0 < sum(y) < len(y):
        values = binary_metrics(scores, y, tau0, frozen_scores=frozen)
        row.update(EER=values["eer"], AUC=values["auroc"],
                   balanced_accuracy=values["balanced_accuracy"], FPR=values["fpr"], FNR=values["fnr"])
    else:
        row.update({name: None for name in ("EER", "AUC", "balanced_accuracy", "FPR", "FNR")})
    row["mean_abs_score_delta"] = mean([abs(a-b) for a, b in zip(scores, frozen)])
    row["mean_update_norm"] = mean([records[s]["update_norm"] for s in ids if "update_norm" in records[s]])
    row["mean_parameter_drift"] = mean([records[s]["distance_from_source"] for s in ids if "distance_from_source" in records[s]])
    row["evidence_damage"] = mean([records[s]["evidence_damage"] for s in ids if "evidence_damage" in records[s]])
    row["helpful_updates"] = sum(int(f > tau0) != labels[s] and int(records[s]["score_after"] > tau0) == labels[s]
                                 for s, f in zip(ids, frozen))
    row["harmful_updates"] = sum(int(f > tau0) == labels[s] and int(records[s]["score_after"] > tau0) != labels[s]
                                 for s, f in zip(ids, frozen))
    row["runtime_seconds_per_sample"] = mean([records[s]["runtime_seconds"] for s in ids if "runtime_seconds" in records[s]])
    return row


def write_csv_once(path, rows):
    if not rows:
        raise ValueError("empty CSV")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plots(folder, sample_rows, metric_rows):
    # Pillow is present in the fixed tta environment; Matplotlib cannot import
    # here because its pyparsing dependency is absent. Render standalone PNGs.
    from PIL import Image, ImageDraw
    colors = {"in_the_wild": "#1f77b4", "asv2021_la": "#ff7f0e",
              "asv2021_df": "#2ca02c", "codecfake": "#d62728", "wavefake": "#9467bd"}

    def scatter(rows, x, y, name):
        points = [(row["domain"], row[x], row[y]) for row in rows
                  if row.get(x) is not None and row.get(y) is not None
                  and math.isfinite(row[x]) and math.isfinite(row[y])]
        if not points:
            raise ValueError("plot has no finite points: " + name)
        width, height = 800, 540
        left, right, top, bottom = 95, 25, 55, 75
        xmin, xmax = min(p[1] for p in points), max(p[1] for p in points)
        ymin, ymax = min(p[2] for p in points), max(p[2] for p in points)
        if xmin == xmax:
            xmin -= 1
            xmax += 1
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.line((left, top, left, height-bottom), fill="black", width=2)
        draw.line((left, height-bottom, width-right, height-bottom), fill="black", width=2)
        for domain, xv, yv in points:
            px = left + (xv - xmin) / (xmax - xmin) * (width-left-right)
            py = height-bottom - (yv - ymin) / (ymax - ymin) * (height-top-bottom)
            draw.ellipse((px-2, py-2, px+2, py+2), fill=colors[domain])
        draw.text((left, 15), name.replace("_", " "), fill="black")
        draw.text((left, height-40), x, fill="black")
        draw.text((5, top), y, fill="black")
        draw.text((left, height-bottom+10), f"{xmin:.3g}", fill="black")
        draw.text((width-right-75, height-bottom+10), f"{xmax:.3g}", fill="black")
        draw.text((left-80, height-bottom-8), f"{ymin:.3g}", fill="black")
        draw.text((left-80, top), f"{ymax:.3g}", fill="black")
        for i, domain in enumerate(sorted({p[0] for p in points})):
            draw.rectangle((width-180, top+18*i, width-170, top+18*i+10), fill=colors[domain])
            draw.text((width-165, top+18*i), domain, fill="black")
        image.save(folder / f"{name}.png")

    pairs = [
        ("target_benefit", "evidence_damage", "benefit_vs_evidence_damage"),
        ("target_benefit", "distance_from_source", "benefit_vs_parameter_drift"),
        ("evidence_damage", "harmful_update", "evidence_damage_vs_harmful_update"),
        ("entropy_reduction", "task_benefit", "entropy_reduction_vs_detection_improvement"),
    ]
    for x, y, name in pairs:
        scatter(sample_rows, x, y, name)
    nonfrozen = [row for row in metric_rows if row["arm"] != "Frozen"]
    scatter(nonfrozen, "mean_parameter_drift", "delta_EER_vs_Frozen", "drift_vs_eer_degradation")
    scatter(nonfrozen, "mean_parameter_drift", "delta_AUC_vs_Frozen", "drift_vs_auc_change")


def analyze(run):
    config, scores = complete_scores(run)  # audit is not opened before this line
    if config["scientific_role"] != "mechanism_dev":
        raise ValueError("engineering smoke must not produce scientific metrics")
    labels = {}
    for domain, by_id in scores.items():
        audit = read_json(ROOT / "experiments/multidomain_mechanism/manifests" /
                          f"{domain}_mechanism_audit.json")
        labels[domain] = selected_labels(domain, audit, set(by_id))
    metrics = []
    samples = []
    for domain, by_id in scores.items():
        tau0 = read_json(run / "diagnostics" / f"{domain}_provenance.json")["tau0"]
        for setting, arm in (("Frozen", "Frozen"), *[(s, a) for s in SETTINGS for a in ARMS]):
            records = {sid: values[(setting, arm)] for sid, values in by_id.items()}
            metric = metric_row(domain, setting, arm, records, labels[domain], tau0)
            metrics.append(metric)
            for sid, row in records.items():
                if arm == "Frozen":
                    continue
                y = labels[domain][sid]
                before = int(row["score_frozen"] > tau0) == y
                after = int(row["score_after"] > tau0) == y
                samples.append({"domain": domain, "setting": setting, "arm": arm, "sample_id": sid,
                                "label": y, "correct_before": int(before), "correct_after": int(after),
                                "helpful_update": int(not before and after),
                                "harmful_update": int(before and not after),
                                "task_benefit": int(after) - int(before),
                                "target_benefit": row["view_before"] - row["view_after"],
                                "entropy_reduction": row["entropy_before"] - row["entropy_after"],
                                "evidence_damage": row["evidence_damage"],
                                "distance_from_source": row["distance_from_source"]})
    by_domain = defaultdict(dict)
    for row in metrics:
        by_domain[row["domain"]][(row["setting"], row["arm"])] = row
    for row in metrics:
        frozen = by_domain[row["domain"]][("Frozen", "Frozen")]
        row["delta_EER_vs_Frozen"] = row["EER"] - frozen["EER"] if row["EER"] is not None else None
        row["delta_AUC_vs_Frozen"] = row["AUC"] - frozen["AUC"] if row["AUC"] is not None else None
    macro = []
    for setting, arm in (("Frozen", "Frozen"), *[(s, a) for s in SETTINGS for a in ARMS]):
        subset = [r for r in metrics if r["setting"] == setting and r["arm"] == arm]
        macro.append({"setting": setting, "arm": arm, "domains": len(subset),
                      "ranking_metric_domains": sum(r["EER"] is not None for r in subset),
                      **{name: mean([r[name] for r in subset if r[name] is not None])
                         for name in ("EER", "AUC", "balanced_accuracy", "FPR", "FNR",
                                      "mean_abs_score_delta", "mean_update_norm", "mean_parameter_drift",
                                      "evidence_damage", "helpful_updates", "harmful_updates")}})
    folder = run / "analysis"
    write_csv_once(folder / "metrics.csv", metrics)
    write_csv_once(folder / "sample_mechanisms.csv", samples)
    write_csv_once(folder / "macro.csv", macro)
    plots(folder, samples, metrics)
    summary = {"status": "PARTIAL_THREE_DOMAIN_MECHANISM_DEV", "target90_accessed": False,
               "final_holdout_labels_accessed": False, "audited_domains": list(scores),
               "not_run_domains": ["codecfake", "wavefake"], "per_domain": metrics, "macro": macro,
               "single_class_domains": sorted({r["domain"] for r in metrics if r["EER"] is None}),
               "interpretation": "descriptive guard-capacity contrast; no 5-domain or EPDC claim"}
    # The score worker leaves this marker; move it, then create the final analysis
    # summary without overwriting any artifact or run directory.
    (folder / "summary.json").rename(run / "diagnostics" / "score_completion.json")
    write_once(folder / "summary.json", summary)
    lines = ["# Guard capacity: partial mechanism-dev analysis", "",
             "Three cached domains only. Codecfake and WaveFake: NOT_RUN. No target90 or final-holdout labels used.", "",
             "| Domain | Setting | Arm | EER | AUC | Helpful | Harmful | Evidence damage |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in metrics:
        fmt = lambda v: "NA" if v is None else f"{v:.6f}"
        lines.append(f"| {row['domain']} | {row['setting']} | {row['arm']} | {fmt(row['EER'])} | {fmt(row['AUC'])} | {row['helpful_updates']} | {row['harmful_updates']} | {fmt(row['evidence_damage'])} |")
    lines.extend(["", "Single-class domains have undefined EER/AUC and are excluded from those macro means. The domain count for ranking metrics must be reported separately.",
                  "Figures show descriptive associations. Labels appear only in post-score analysis; no gate uses them."])
    with (folder / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if run.parent != (ROOT / "experiments/multidomain_mechanism/results").resolve():
        raise ValueError("run must be a direct child of guard results")
    print(json.dumps(analyze(run), allow_nan=False)[0:1000])


if __name__ == "__main__":
    main()
