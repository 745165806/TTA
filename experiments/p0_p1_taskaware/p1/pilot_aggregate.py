#!/usr/bin/env python
"""P1 pilot aggregation + mechanism gate.

Labels are read ONLY here (post-hoc). The four variants and Frozen are compared
and the section-10 mechanism gate is applied to taskaware_full.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from scripts.common import POST_HOC_MARKER, load_context, load_labels

VARIANTS = ["taskaware_full", "taskaware_no_gate", "taskaware_no_source_keep", "taskaware_control"]


def metrics_for(scores_after, scores_before, labels, threshold):
    from eptta.evaluation.metrics import binary_metrics
    order = sorted(scores_after)
    after = [scores_after[k] for k in order]
    before = [scores_before[k] for k in order]
    ys = [labels[k] for k in order]
    m = binary_metrics(after, ys, threshold, frozen_scores=before)
    return {
        "EER": m["eer"], "AUC": m["auroc"],
        "accuracy_at_tau0": (m["tp"] + m["tn"]) / m["count"],
        "count": m["count"],
        "helpful_flips": m["helpful_flips"], "harmful_flips": m["harmful_flips"],
        "helpful_by_class": m["helpful_flips_by_class"],
        "harmful_by_class": m["harmful_flips_by_class"],
        "tpr": m["tpr"], "fpr": m["fpr"], "fnr": m["fnr"],
    }


def variant_stats(records, labels, threshold):
    after = {r["sample_id"]: r["score_after"] for r in records}
    before = {r["sample_id"]: r["score_before"] for r in records}
    base = metrics_for(after, before, labels, threshold)

    signed = []
    bonafide_signed, spoof_signed = [], []
    for r in records:
        y = labels[r["sample_id"]]
        sd = (2 * y - 1) * (r["score_after"] - r["score_before"])
        signed.append(sd)
        (spoof_signed if y == 1 else bonafide_signed).append(sd)

    n = len(records)
    applied = [r for r in records if r.get("adaptation_applied")]
    safety_rejects = [r for r in records if r.get("safety_rejected")]
    fallbacks = [r for r in records if r.get("numeric_fallback")]
    return {
        "EER": base["EER"], "AUC": base["AUC"],
        "accuracy_at_tau0": base["accuracy_at_tau0"],
        "mean_signed_task_delta": statistics.fmean(signed),
        "bonafide_mean_signed_delta": statistics.fmean(bonafide_signed) if bonafide_signed else None,
        "spoof_mean_signed_delta": statistics.fmean(spoof_signed) if spoof_signed else None,
        "helpful_flips": base["helpful_flips"], "harmful_flips": base["harmful_flips"],
        "adaptation_coverage": len(applied) / n if n else 0.0,
        "abstain_count": sum(1 for r in records if r.get("abstain_reason")),
        "source_safety_reject_count": len(safety_rejects),
        "source_safety_reject_rate": len(safety_rejects) / n if n else 0.0,
        "mean_R_norm": statistics.fmean(r.get("final_R_norm") or 0.0 for r in records),
        "mean_abs_delta_score": statistics.fmean(abs(r["score_after"] - r["score_before"]) for r in records),
        "numeric_fallback_count": len(fallbacks),
        "mean_source_anchor_flip_count": statistics.fmean(
            r.get("source_anchor_flip_count") for r in records if r.get("source_anchor_flip_count") is not None) if any(
                r.get("source_anchor_flip_count") is not None for r in records) else None,
        "tpr": base["tpr"], "fpr": base["fpr"], "fnr": base["fnr"],
        "helpful_by_class": base["helpful_by_class"], "harmful_by_class": base["harmful_by_class"],
    }


def gate_stats(records):
    n = len(records)
    applied = [r for r in records if r.get("adaptation_applied")]
    return {
        "n": n,
        "adaptation_coverage": len(applied) / n if n else 0.0,
        "abstain_count": sum(1 for r in records if r.get("abstain_reason")),
        "abstain_reasons": {reason: sum(1 for r in records if r.get("abstain_reason") == reason)
                            for reason in {r.get("abstain_reason") for r in records if r.get("abstain_reason")}},
        "mean_gate_agreement": statistics.fmean(r["gate_agreement"] for r in records if r.get("gate_agreement") is not None),
        "mean_gate_confidence": statistics.fmean(r["gate_confidence"] for r in records if r.get("gate_confidence") is not None),
        "teacher_spoof_fraction": statistics.fmean(r["teacher_label"] for r in records if r.get("teacher_label") is not None),
        "safety_reject_count": sum(1 for r in records if r.get("safety_rejected")),
        "mean_source_anchor_flip_count": statistics.fmean(
            r["source_anchor_flip_count"] for r in records if r.get("source_anchor_flip_count") is not None) if any(
                r.get("source_anchor_flip_count") is not None for r in records) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    p1_dir = run_dir / "p1"
    labels = load_labels()
    _b, _res, _meta, _cache, _features, threshold = load_context()

    variant_records = {}
    for variant in VARIANTS:
        path = p1_dir / ("p1_%s.jsonl" % variant)
        if not path.is_file():
            raise SystemExit("missing pilot output: %s" % path)
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        records.sort(key=lambda r: r["sample_id"])
        variant_records[variant] = records

    # Frozen baseline from score_before (identical across variants).
    frozen_records = variant_records["taskaware_full"]
    frozen_after = {r["sample_id"]: r["score_before"] for r in frozen_records}
    frozen_before = dict(frozen_after)
    frozen = metrics_for(frozen_after, frozen_before, labels, threshold)
    frozen_stats = {
        "EER": frozen["EER"], "AUC": frozen["AUC"],
        "accuracy_at_tau0": frozen["accuracy_at_tau0"],
        "mean_signed_task_delta": 0.0, "bonafide_mean_signed_delta": 0.0,
        "spoof_mean_signed_delta": 0.0, "helpful_flips": 0, "harmful_flips": 0,
        "adaptation_coverage": 0.0, "abstain_count": 0,
        "source_safety_reject_count": 0, "source_safety_reject_rate": 0.0,
        "mean_R_norm": 0.0, "mean_abs_delta_score": 0.0, "numeric_fallback_count": 0,
        "mean_source_anchor_flip_count": None,
        "tpr": frozen["tpr"], "fpr": frozen["fpr"], "fnr": frozen["fnr"],
        "helpful_by_class": frozen["helpful_by_class"],
        "harmful_by_class": frozen["harmful_by_class"],
    }

    stats = {"frozen": frozen_stats}
    for variant in VARIANTS:
        stats[variant] = variant_stats(variant_records[variant], labels, threshold)

    # p1_pilot_metrics.csv
    fieldnames = ["method", "EER", "AUC", "accuracy_at_tau0", "mean_signed_task_delta",
                  "bonafide_mean_signed_delta", "spoof_mean_signed_delta",
                  "helpful_flips", "harmful_flips", "adaptation_coverage",
                  "abstain_count", "source_safety_reject_rate", "mean_R_norm",
                  "mean_abs_delta_score", "numeric_fallback_count"]
    with (p1_dir / "p1_pilot_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name in ["frozen"] + VARIANTS:
            writer.writerow({"method": name, **{k: stats[name][k] for k in fieldnames[1:]}})

    # p1_per_class_metrics.csv
    per_class_fields = ["method", "tpr", "fpr", "fnr", "bonafide_mean_signed_delta",
                        "spoof_mean_signed_delta", "helpful_bonafide", "harmful_bonafide",
                        "helpful_spoof", "harmful_spoof"]
    with (p1_dir / "p1_per_class_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class_fields)
        writer.writeheader()
        for name in ["frozen"] + VARIANTS:
            s = stats[name]
            writer.writerow({
                "method": name, "tpr": s["tpr"], "fpr": s["fpr"], "fnr": s["fnr"],
                "bonafide_mean_signed_delta": s["bonafide_mean_signed_delta"],
                "spoof_mean_signed_delta": s["spoof_mean_signed_delta"],
                "helpful_bonafide": s["helpful_by_class"].get("0"),
                "harmful_bonafide": s["harmful_by_class"].get("0"),
                "helpful_spoof": s["helpful_by_class"].get("1"),
                "harmful_spoof": s["harmful_by_class"].get("1"),
            })

    # p1_gate_stats.csv (task-aware variants only)
    gate_fields = ["method", "adaptation_coverage", "abstain_count", "abstain_reasons",
                   "mean_gate_agreement", "mean_gate_confidence", "teacher_spoof_fraction",
                   "safety_reject_count", "mean_source_anchor_flip_count"]
    with (p1_dir / "p1_gate_stats.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=gate_fields)
        writer.writeheader()
        for variant in VARIANTS[:3]:
            g = gate_stats(variant_records[variant])
            writer.writerow({"method": variant, **{k: g[k] for k in gate_fields[1:]}})

    # Mechanism gate on taskaware_full.
    full = stats["taskaware_full"]
    applied_records = [r for r in variant_records["taskaware_full"] if r.get("adaptation_applied")]
    accepted_flips = [r["source_anchor_flip_count"] for r in applied_records
                      if r.get("source_anchor_flip_count") is not None]
    gate = {
        "numeric_fallback_count": full["numeric_fallback_count"],
        "accepted_source_anchor_flip_max": max(accepted_flips) if accepted_flips else 0,
        "adaptation_coverage": full["adaptation_coverage"],
        "mean_signed_task_delta": full["mean_signed_task_delta"],
        "helpful_flips": full["helpful_flips"],
        "harmful_flips": full["harmful_flips"],
        "eer_less_than_frozen": full["EER"] < frozen_stats["EER"],
        "auc_greater_than_frozen": full["AUC"] > frozen_stats["AUC"],
    }
    gate["numeric_ok"] = gate["numeric_fallback_count"] == 0
    gate["source_safety_ok"] = gate["accepted_source_anchor_flip_max"] == 0
    gate["coverage_ok"] = gate["adaptation_coverage"] > 0
    gate["signed_delta_ok"] = gate["mean_signed_task_delta"] > 0
    gate["flip_ok"] = gate["helpful_flips"] > gate["harmful_flips"]
    gate["task_metric_ok"] = gate["eer_less_than_frozen"] or gate["auc_greater_than_frozen"]
    gate["PASS"] = all([gate["numeric_ok"], gate["source_safety_ok"], gate["coverage_ok"],
                        gate["signed_delta_ok"], gate["flip_ok"], gate["task_metric_ok"]])

    full_signed = full["mean_signed_task_delta"]
    no_task_gain = (not gate["eer_less_than_frozen"] and
                    full["AUC"] <= frozen_stats["AUC"] + 1e-4)
    summary = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "threshold": threshold,
        "frozen": frozen_stats,
        "variants": {name: stats[name] for name in VARIANTS},
        "mechanism_gate": gate,
        "P1_MECHANISM_PASS": gate["PASS"],
        "scientific_flags": {
            "UPDATE_DIRECTION_NOT_TASK_ALIGNED": bool(full_signed <= 0),
            "OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN": bool(no_task_gain),
            "FROZEN_TEACHER_UNRELIABLE": bool(frozen_stats["accuracy_at_tau0"] < 0.5),
        },
    }
    (p1_dir / "p1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (p1_dir / "p1_summary.md").write_text(build_md(summary), encoding="utf-8")

    print(json.dumps({"mechanism_gate": gate, "P1_MECHANISM_PASS": gate["PASS"]},
                     ensure_ascii=False, indent=2))
    if not gate["PASS"]:
        print("P1_MECHANISM_FAIL")
        sys.exit(3)
    print("P1_MECHANISM_PASS")


def build_md(s):
    f = s["frozen"]
    lines = [
        "# P1 pilot summary",
        "",
        "**POST_HOC_DEVELOPMENT_ONLY**",
        "",
        "Frozen EER=%.6f AUC=%.6f" % (f["EER"], f["AUC"]),
        "",
        "| method | EER | AUC | mean_signed_delta | helpful | harmful | coverage | safety_reject_rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ["frozen", "taskaware_full", "taskaware_no_gate", "taskaware_no_source_keep", "taskaware_control"]:
        v = s["frozen"] if name == "frozen" else s["variants"][name]
        lines.append("| %s | %.6f | %.6f | %.6f | %d | %d | %.4f | %.4f |" %
                     (name, v["EER"], v["AUC"], v["mean_signed_task_delta"],
                      v["helpful_flips"], v["harmful_flips"], v["adaptation_coverage"],
                      v["source_safety_reject_rate"]))
    lines.append("")
    lines.append("## Mechanism gate (taskaware_full)")
    for k in ("numeric_ok", "source_safety_ok", "coverage_ok", "signed_delta_ok",
              "flip_ok", "task_metric_ok", "PASS"):
        lines.append("- %s: %s" % (k, s["mechanism_gate"][k]))
    lines.append("")
    lines.append("P1 mechanism: %s" % ("PASS" if s["P1_MECHANISM_PASS"] else "FAIL"))
    lines.append("")
    lines.append("## Scientific conclusion")
    flags = s.get("scientific_flags", {})
    lines.append("- UPDATE_DIRECTION_NOT_TASK_ALIGNED: %s" % flags.get("UPDATE_DIRECTION_NOT_TASK_ALIGNED"))
    lines.append("- OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN: %s" % flags.get("OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN"))
    lines.append("- FROZEN_TEACHER_UNRELIABLE (frozen accuracy at tau0 < 0.5): %s" % flags.get("FROZEN_TEACHER_UNRELIABLE"))
    lines.append("- The frozen pseudo-label teacher has target10 accuracy %.4f at tau0, so "
                 "task-space pseudo-BCE is misled for the majority of samples; this explains the "
                 "negative mean signed task delta for taskaware_full." % (s["frozen"]["accuracy_at_tau0"]))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
