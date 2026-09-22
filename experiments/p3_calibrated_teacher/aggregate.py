#!/usr/bin/env python
"""Post-hoc labelled aggregation for the P3 mechanism study."""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from eptta.evaluation.metrics import binary_metrics
from p3_common import load_context, load_ids, resolve_input


VARIANT_FILES = {
    "Frozen": "frozen_calonly.jsonl",
    "CalOnly": "frozen_calonly.jsonl",
    "SourceTauTeacher": "source_tau_teacher.jsonl",
    "CalibratedTeacher-NoSelect": "calibrated_no_select.jsonl",
    "CalibratedSelective": "calibrated_selective.jsonl",
    "OracleTeacher": "oracle_teacher.jsonl",
}


def read_records(path, expected_ids):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != set(expected_ids):
        raise RuntimeError("record coverage mismatch: %s" % path)
    return [by_id[sid] for sid in expected_ids]


def load_labels(config, expected_ids):
    doc = json.loads(resolve_input(config["inputs"]["target10_label_manifest"]).read_text(
        encoding="utf-8"))
    all_labels = {row["sample_id"]: int(row["label"]) for row in doc["records"]}
    if not set(expected_ids).issubset(all_labels):
        raise RuntimeError("post-hoc labels do not cover evalU")
    return {sid: all_labels[sid] for sid in expected_ids}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap_mean(values, seed, n_boot):
    rng = random.Random(seed)
    n = len(values)
    estimates = [statistics.fmean(values[rng.randrange(n)] for _ in range(n))
                 for _ in range(n_boot)]
    return {"estimate": statistics.fmean(values), "ci95": [percentile(estimates, 0.025),
                                                              percentile(estimates, 0.975)],
            "seed": seed, "n": n_boot}


def threshold_oracle(scores, labels):
    points = []
    for threshold in sorted(set(scores)):
        prediction = [int(score > threshold) for score in scores]
        fp = sum(p == 1 and y == 0 for p, y in zip(prediction, labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(prediction, labels))
        n0 = sum(y == 0 for y in labels)
        n1 = len(labels) - n0
        points.append((abs(fp / n0 - fn / n1), threshold))
    return min(points)[1]


def variant_metrics(records, labels, threshold):
    scores = [row["score_after"] for row in records]
    before = [row["score_before"] for row in records]
    ys = [labels[row["sample_id"]] for row in records]
    metric = binary_metrics(scores, ys, threshold, frozen_scores=before)
    signed = [(2 * y - 1) * (after - frozen)
              for y, after, frozen in zip(ys, scores, before)]
    bona = [value for value, y in zip(signed, ys) if y == 0]
    spoof = [value for value, y in zip(signed, ys) if y == 1]
    n = len(records)
    accepted = [row for row in records if row.get("adaptation_applied")]
    rejects = [row for row in records if row.get("safety_rejected")]
    abstained = [row for row in records if row.get("abstain_reason")]
    accepted_flips = [row.get("final_source_anchor_flip_count") for row in accepted]
    return {
        "EER": metric["eer"], "AUC": metric["auroc"],
        "balanced_accuracy": metric["balanced_accuracy"],
        "FPR": metric["fpr"], "FNR": metric["fnr"],
        "mean_signed_task_delta": statistics.fmean(signed),
        "bonafide_signed_delta": statistics.fmean(bona),
        "spoof_signed_delta": statistics.fmean(spoof),
        "helpful_flips": metric["helpful_flips"],
        "harmful_flips": metric["harmful_flips"],
        "adaptation_coverage": len(accepted) / n,
        "abstention_rate": len(abstained) / n,
        "safety_reject_rate": len(rejects) / n,
        "mean_abs_delta_score": statistics.fmean(abs(a - b) for a, b in zip(scores, before)),
        "numeric_fallback_count": sum(bool(row.get("numeric_fallback")) for row in records),
        "accepted_source_flips": max((value or 0 for value in accepted_flips), default=0),
        "abstain_reasons": {reason: sum(row.get("abstain_reason") == reason for row in records)
                            for reason in sorted({row.get("abstain_reason") for row in records
                                                  if row.get("abstain_reason")})},
        "signed_delta_bootstrap": None,
    }, signed


def teacher_quality(records, labels):
    rows = [row for row in records if row.get("teacher_label") is not None]
    correct = [int(row["teacher_label"] == labels[row["sample_id"]]) for row in rows]
    by_class = {}
    for y, name in ((0, "bonafide"), (1, "spoof")):
        selected = [int(row["teacher_label"] == y) for row in rows
                    if labels[row["sample_id"]] == y]
        by_class[name] = statistics.fmean(selected)
    return {"accuracy": statistics.fmean(correct),
            "balanced_accuracy": 0.5 * (by_class["bonafide"] + by_class["spoof"]),
            "bonafide_accuracy": by_class["bonafide"],
            "spoof_accuracy": by_class["spoof"], "correct": correct}


def write_csv(path, metrics):
    fields = ["variant", "EER", "AUC", "balanced_accuracy", "FPR", "FNR",
              "mean_signed_task_delta", "bonafide_signed_delta", "spoof_signed_delta",
              "helpful_flips", "harmful_flips", "adaptation_coverage", "abstention_rate",
              "safety_reject_rate", "mean_abs_delta_score", "numeric_fallback_count",
              "accepted_source_flips"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"variant": name, **{key: values[key] for key in fields[1:]}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config, resources, _meta, _cache, _features = load_context()
    ids = load_ids("evalU")
    labels = load_labels(config, ids)
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("status") != "OK":
        raise SystemExit("CALIBRATION_FIT_FAIL: adaptation was disabled; no P3 success claim")
    records = {name: read_records(run_dir / filename, ids)
               for name, filename in VARIANT_FILES.items()}
    # Frozen and CalOnly are two evaluations of exactly the same score records.
    thresholds = {name: resources.tau0 for name in VARIANT_FILES}
    for name in ("CalOnly", "CalibratedTeacher-NoSelect", "CalibratedSelective", "OracleTeacher"):
        thresholds[name] = calibration["tau_hat"]
    metrics = {}
    signed = {}
    seed, n_boot = config["bootstrap"]["seed"], config["bootstrap"]["n"]
    for name in VARIANT_FILES:
        metrics[name], signed[name] = variant_metrics(records[name], labels, thresholds[name])
        metrics[name]["signed_delta_bootstrap"] = bootstrap_mean(signed[name], seed, n_boot)

    invariant = {
        "EER_unchanged": abs(metrics["CalOnly"]["EER"] - metrics["Frozen"]["EER"]) <= 1e-12,
        "AUC_unchanged": abs(metrics["CalOnly"]["AUC"] - metrics["Frozen"]["AUC"]) <= 1e-12,
    }
    invariant["PASS"] = invariant["EER_unchanged"] and invariant["AUC_unchanged"]
    if not invariant["PASS"]:
        raise RuntimeError("implementation ERROR: CalOnly changed ranking metrics")

    source_quality = teacher_quality(records["SourceTauTeacher"], labels)
    calibrated_quality = teacher_quality(records["CalibratedSelective"], labels)
    source_by_id = {row["sample_id"]: row for row in records["SourceTauTeacher"]}
    cal_by_id = {row["sample_id"]: row for row in records["CalibratedSelective"]}
    paired = [int(cal_by_id[sid]["teacher_label"] == labels[sid]) -
              int(source_by_id[sid]["teacher_label"] == labels[sid]) for sid in ids]
    teacher_bootstrap = bootstrap_mean(paired, seed, n_boot)
    disagreement = statistics.fmean(
        int(cal_by_id[sid]["teacher_label"] != source_by_id[sid]["teacher_label"])
        for sid in ids)

    frozen_scores = [records["Frozen"][i]["score_before"] for i in range(len(ids))]
    ys = [labels[sid] for sid in ids]
    tau_oracle = threshold_oracle(frozen_scores, ys)
    selected = metrics["CalibratedSelective"]
    teacher_significant = teacher_bootstrap["ci95"][0] > 0
    signed_positive = selected["mean_signed_task_delta"] > 0
    metric_improved = (selected["EER"] < metrics["Frozen"]["EER"] or
                       selected["AUC"] > metrics["Frozen"]["AUC"])
    numeric_ok = selected["numeric_fallback_count"] == 0
    source_safe = selected["accepted_source_flips"] == 0
    mechanism_pass = all((teacher_significant, signed_positive, metric_improved,
                          numeric_ok, source_safe))
    mechanism_partial = bool(teacher_significant and signed_positive and not metric_improved and
                             numeric_ok and source_safe)
    calibration_teacher_fail = calibrated_quality["accuracy"] <= source_quality["accuracy"]

    for quality in (source_quality, calibrated_quality):
        quality.pop("correct")
    summary = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "protocol": config["protocol"], "main_split_rule": config["split"],
        "calU_count": calibration["calU_count"], "evalU_count": len(ids),
        "GMM": calibration, "tau0": resources.tau0, "tau_hat": calibration["tau_hat"],
        "tau_oracle_POST_HOC": tau_oracle,
        "abs_tau0_oracle": abs(resources.tau0 - tau_oracle),
        "abs_tauhat_oracle": abs(calibration["tau_hat"] - tau_oracle),
        "teacher_quality": {
            "SourceTauTeacher": source_quality,
            "CalibratedTeacher": calibrated_quality,
            "teacher_disagreement_rate": disagreement,
            "teacher_accuracy_delta": teacher_bootstrap,
        },
        "variants": metrics, "CalOnly_invariant": invariant,
        "P3_MECHANISM_PASS": mechanism_pass,
        "P3_MECHANISM_PARTIAL": mechanism_partial,
        "UNSUPERVISED_CALIBRATION_TEACHER_FAIL": calibration_teacher_fail,
        "accepted_source_flips": selected["accepted_source_flips"],
        "multi_domain_confirmatory": "NOT RUN", "main_merged": "NO",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(run_dir / "metrics.csv", metrics)
    lines = [
        "# P3 Calibration-Aware Selective EP-TTA", "", "**POST_HOC_DEVELOPMENT_ONLY**", "",
        "- protocol: `%s`" % config["protocol"],
        "- split: `%s`" % config["split"]["salt"],
        "- calU/evalU: %d/%d" % (calibration["calU_count"], len(ids)),
        "- tau0/tau_hat/tau_oracle: %.6f / %.6f / %.6f" %
        (resources.tau0, calibration["tau_hat"], tau_oracle), "",
        "| variant | EER | AUC | signed delta | coverage | abstention | safety reject |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_FILES:
        m = metrics[name]
        lines.append("| %s | %.6f | %.6f | %+.6f | %.4f | %.4f | %.4f |" %
                     (name, m["EER"], m["AUC"], m["mean_signed_task_delta"],
                      m["adaptation_coverage"], m["abstention_rate"],
                      m["safety_reject_rate"]))
    lines += ["", "- teacher accuracy source/calibrated: %.6f / %.6f" %
              (source_quality["accuracy"], calibrated_quality["accuracy"]),
              "- teacher delta 95%% CI: %s" % teacher_bootstrap["ci95"],
              "- CalOnly EER/AUC invariant: %s" % invariant["PASS"],
              "- P3_MECHANISM_PASS: %s" % mechanism_pass,
              "- P3_MECHANISM_PARTIAL: %s" % mechanism_partial,
              "- UNSUPERVISED_CALIBRATION_TEACHER_FAIL: %s" % calibration_teacher_fail, ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "P3_MECHANISM_PASS", "P3_MECHANISM_PARTIAL",
        "UNSUPERVISED_CALIBRATION_TEACHER_FAIL")}, indent=2))


if __name__ == "__main__":
    main()
