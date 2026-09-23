#!/usr/bin/env python
"""Aggregate P3 sensitivity splits and class-asymmetric diagnostics."""
import argparse
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from eptta.evaluation.metrics import binary_metrics
from followup_stats import (paired_mean_bootstrap, paired_ranking_bootstrap,
                            task_gain_supported)
from p3_common import load_config, load_context, load_ids, resolve_input


FILES = {
    "Frozen": "frozen_calonly.jsonl",
    "CalOnly": "frozen_calonly.jsonl",
    "SourceTauTeacher": "source_tau_teacher.jsonl",
    "CalibratedTeacher-NoSelect": "calibrated_no_select.jsonl",
    "CalibratedSelective": "calibrated_selective.jsonl",
    "OracleTeacher": "oracle_teacher.jsonl",
}


def read_records(path, ids):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise RuntimeError("record coverage mismatch: %s" % path)
    return [by_id[sid] for sid in ids]


def label_map(config):
    doc = json.loads(resolve_input(config["inputs"]["target10_label_manifest"]).read_text(
        encoding="utf-8"))
    return {row["sample_id"]: int(row["label"]) for row in doc["records"]}


def oracle_threshold(scores, labels):
    best = None
    n0, n1 = labels.count(0), labels.count(1)
    for threshold in sorted(set(scores)):
        fp = sum(score > threshold and y == 0 for score, y in zip(scores, labels))
        fn = sum(score <= threshold and y == 1 for score, y in zip(scores, labels))
        candidate = (abs(fp / n0 - fn / n1), threshold)
        if best is None or candidate < best:
            best = candidate
    return best[1]


def teacher_accuracy(records, labels):
    correct = [int(row["teacher_label"] == y) for row, y in zip(records, labels)]
    by_class = {}
    for y, name in ((0, "bonafide"), (1, "spoof")):
        selected = [value for value, label in zip(correct, labels) if label == y]
        by_class[name] = statistics.fmean(selected)
    return {
        "accuracy": statistics.fmean(correct),
        "balanced_accuracy": 0.5 * (by_class["bonafide"] + by_class["spoof"]),
        "bonafide_accuracy": by_class["bonafide"],
        "spoof_accuracy": by_class["spoof"], "correct": correct,
    }


def metrics(records, labels, threshold, seed, n_boot):
    after = [row["score_after"] for row in records]
    before = [row["score_before"] for row in records]
    metric = binary_metrics(after, labels, threshold, frozen_scores=before)
    signed = [(2 * y - 1) * (a - b) for y, a, b in zip(labels, after, before)]
    bona = [value for value, y in zip(signed, labels) if y == 0]
    spoof = [value for value, y in zip(signed, labels) if y == 1]
    n = len(records)
    return {
        "EER": metric["eer"], "AUC": metric["auroc"],
        "balanced_accuracy": metric["balanced_accuracy"],
        "FPR": metric["fpr"], "FNR": metric["fnr"],
        "mean_signed_delta": statistics.fmean(signed),
        "signed_delta_ci95": paired_mean_bootstrap(
            signed, seed=seed, n_boot=n_boot)["ci95"],
        "bonafide_signed_delta": statistics.fmean(bona),
        "spoof_signed_delta": statistics.fmean(spoof),
        "coverage": sum(bool(row.get("adaptation_applied")) for row in records) / n,
        "abstention": sum(bool(row.get("abstain_reason")) for row in records) / n,
        "safety_reject": sum(bool(row.get("safety_rejected")) for row in records) / n,
        "numeric_fallback_count": sum(bool(row.get("numeric_fallback")) for row in records),
        "accepted_source_flips": max((row.get("final_source_anchor_flip_count") or 0
                                      for row in records if row.get("adaptation_applied")), default=0),
    }


def sensitivity_summary(run_dir, manifest_dir, labels_by_id, resources, seed, n_boot):
    ids = load_ids("evalU", manifest_dir)
    labels = [labels_by_id[sid] for sid in ids]
    records = {name: read_records(run_dir / filename, ids) for name, filename in FILES.items()}
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    thresholds = {name: resources.tau0 for name in FILES}
    for name in ("CalOnly", "CalibratedTeacher-NoSelect", "CalibratedSelective", "OracleTeacher"):
        thresholds[name] = calibration["tau_hat"]
    variant_metrics = {name: metrics(rows, labels, thresholds[name], seed, n_boot)
                       for name, rows in records.items()}
    source = teacher_accuracy(records["SourceTauTeacher"], labels)
    calibrated = teacher_accuracy(records["CalibratedSelective"], labels)
    teacher_delta = paired_mean_bootstrap(
        [c - s for c, s in zip(calibrated.pop("correct"), source.pop("correct"))],
        seed=seed, n_boot=n_boot)
    frozen_scores = [row["score_before"] for row in records["Frozen"]]
    selected_scores = [row["score_after"] for row in records["CalibratedSelective"]]
    task = paired_ranking_bootstrap(selected_scores, frozen_scores, labels,
                                    seed=seed, n_boot=n_boot)
    selected = variant_metrics["CalibratedSelective"]
    return {
        "salt": json.loads((Path(manifest_dir) / "target10_evalU.json").read_text(
            encoding="utf-8"))["split_rule"]["salt"],
        "calU_count": calibration["calU_count"], "evalU_count": len(ids),
        "tau_hat": calibration["tau_hat"],
        "tau_oracle_POST_HOC": oracle_threshold(frozen_scores, labels),
        "teacher": {"source": source, "calibrated": calibrated, "delta": teacher_delta},
        "variants": variant_metrics,
        "CalibratedSelective_vs_Frozen": task,
        "P3_TEACHER_MECHANISM_SUPPORTED": teacher_delta["ci95"][0] > 0,
        "P3_ADAPTATION_DIRECTION_SUPPORTED": selected["signed_delta_ci95"][0] > 0,
        "P3_TASK_GAIN_SUPPORTED": task_gain_supported(task),
    }


def asymmetry_summary(asym_dir, main_run, labels_by_id, resources, seed, n_boot):
    ids = load_ids("evalU")
    labels = [labels_by_id[sid] for sid in ids]
    calibration = json.loads((main_run / "calibration.json").read_text(encoding="utf-8"))
    frozen = read_records(main_run / "frozen_calonly.jsonl", ids)
    variants = {
        "Oracle-All": read_records(main_run / "oracle_teacher.jsonl", ids),
        "Oracle-BonaOnly": read_records(asym_dir / "oracle_bona_only.jsonl", ids),
        "Oracle-SpoofOnly": read_records(asym_dir / "oracle_spoof_only.jsonl", ids),
        "Calibrated-All": read_records(main_run / "calibrated_selective.jsonl", ids),
        "Calibrated-BonaOnly": read_records(asym_dir / "calibrated_bona_only.jsonl", ids),
        "Calibrated-SpoofOnly": read_records(asym_dir / "calibrated_spoof_only.jsonl", ids),
    }
    threshold = calibration["tau_hat"]
    variant_metrics = {name: metrics(rows, labels, threshold, seed, n_boot)
                       for name, rows in variants.items()}
    frozen_scores = [row["score_before"] for row in frozen]
    comparisons = {}
    for name, rows in variants.items():
        comparisons[name + "_vs_Frozen"] = paired_ranking_bootstrap(
            [row["score_after"] for row in rows], frozen_scores, labels,
            seed=seed, n_boot=n_boot)
    for family in ("Oracle", "Calibrated"):
        all_scores = [row["score_after"] for row in variants[family + "-All"]]
        for mode in ("BonaOnly", "SpoofOnly"):
            name = family + "-" + mode
            comparisons[name + "_vs_" + family + "-All"] = paired_ranking_bootstrap(
                [row["score_after"] for row in variants[name]], all_scores, labels,
                seed=seed, n_boot=n_boot)
    oracle_bona = comparisons["Oracle-BonaOnly_vs_Oracle-All"]
    oracle_spoof = comparisons["Oracle-SpoofOnly_vs_Oracle-All"]
    calibrated_bona = comparisons["Calibrated-BonaOnly_vs_Calibrated-All"]
    oracle_bona_supported = task_gain_supported(oracle_bona)
    oracle_spoof_supported = task_gain_supported(oracle_spoof)
    oracle_spoof_not_better = (oracle_spoof["delta_EER"] >= 0 and
                               oracle_spoof["delta_AUC"] <= 0)
    oracle_spoof_statistically_worse = (
        oracle_spoof["delta_EER_ci95"][0] > 0 or
        oracle_spoof["delta_AUC_ci95"][1] < 0)
    class_supported = oracle_bona_supported and oracle_spoof_not_better
    deployable_supported = class_supported and task_gain_supported(calibrated_bona)
    return {
        "POST_HOC_ORACLE_ONLY_variants": ["Oracle-All", "Oracle-BonaOnly", "Oracle-SpoofOnly"],
        "NOT_DEPLOYABLE_variants": ["Oracle-All", "Oracle-BonaOnly", "Oracle-SpoofOnly"],
        "variants": variant_metrics, "comparisons": comparisons,
        "oracle_bona_only_statistically_better_than_all": oracle_bona_supported,
        "oracle_spoof_only_statistically_better_than_all": oracle_spoof_supported,
        "oracle_spoof_only_not_better_than_all": oracle_spoof_not_better,
        "oracle_spoof_only_statistically_worse_than_all": oracle_spoof_statistically_worse,
        "CLASS_ASYMMETRIC_ADAPTATION_SUPPORTED": class_supported,
        "DEPLOYABLE_BONA_ONLY_SUPPORTED": deployable_supported,
        "source_safety_unchanged": all(v["accepted_source_flips"] == 0
                                       for v in variant_metrics.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--followup-dir", required=True)
    parser.add_argument("--main-run", required=True)
    args = parser.parse_args()
    followup = Path(args.followup_dir)
    main_run = Path(args.main_run)
    config, resources, _meta, _cache, _features = load_context()
    labels = label_map(config)
    followup_cfg = json.loads((EXP_DIR / "configs/followup.json").read_text(encoding="utf-8"))
    seed, n_boot = followup_cfg["bootstrap"]["seed"], followup_cfg["bootstrap"]["n"]
    sensitivities = {}
    for year in ("2027", "2028"):
        run_dir = followup / ("sensitivity_" + year)
        manifest_dir = EXP_DIR / "manifests" / ("p3-sensitivity-" + year)
        summary = sensitivity_summary(run_dir, manifest_dir, labels, resources, seed, n_boot)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sensitivities[year] = summary
    asym_dir = followup / "asymmetry"
    asymmetry = asymmetry_summary(asym_dir, main_run, labels, resources, seed, n_boot)
    (asym_dir / "summary.json").write_text(
        json.dumps(asymmetry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "bootstrap": {"seed": seed, "n": n_boot},
        "sensitivities": sensitivities,
        "SENSITIVITY_TEACHER_STABLE": all(
            row["P3_TEACHER_MECHANISM_SUPPORTED"] for row in sensitivities.values()),
        "SENSITIVITY_DIRECTION_STABLE": all(
            row["P3_ADAPTATION_DIRECTION_SUPPORTED"] for row in sensitivities.values()),
        "SENSITIVITY_TASK_GAIN_STABLE": all(
            row["P3_TASK_GAIN_SUPPORTED"] for row in sensitivities.values()),
        "asymmetry": asymmetry, "GPU_used": False,
        "historical_results_modified": False, "main_merged": False,
    }
    (followup / "followup_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# P3.1/P3.2 follow-up", ""]
    for year, row in sensitivities.items():
        task = row["CalibratedSelective_vs_Frozen"]
        lines += ["## sensitivity-%s" % year, "",
                  "- tau_hat/tau_oracle: %.6f / %.6f" %
                  (row["tau_hat"], row["tau_oracle_POST_HOC"]),
                  "- teacher delta CI: `%s`" % row["teacher"]["delta"]["ci95"],
                  "- task delta EER CI: `%s`" % task["delta_EER_ci95"],
                  "- task delta AUC CI: `%s`" % task["delta_AUC_ci95"], ""]
    lines += ["- SENSITIVITY_TEACHER_STABLE: `%s`" % result["SENSITIVITY_TEACHER_STABLE"],
              "- SENSITIVITY_DIRECTION_STABLE: `%s`" % result["SENSITIVITY_DIRECTION_STABLE"],
              "- SENSITIVITY_TASK_GAIN_STABLE: `%s`" % result["SENSITIVITY_TASK_GAIN_STABLE"],
              "- CLASS_ASYMMETRIC_ADAPTATION_SUPPORTED: `%s`" %
              asymmetry["CLASS_ASYMMETRIC_ADAPTATION_SUPPORTED"],
              "- DEPLOYABLE_BONA_ONLY_SUPPORTED: `%s`" %
              asymmetry["DEPLOYABLE_BONA_ONLY_SUPPORTED"], ""]
    (followup / "followup_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "SENSITIVITY_TEACHER_STABLE", "SENSITIVITY_DIRECTION_STABLE",
        "SENSITIVITY_TASK_GAIN_STABLE")}, indent=2))
    print(json.dumps({key: asymmetry[key] for key in (
        "CLASS_ASYMMETRIC_ADAPTATION_SUPPORTED", "DEPLOYABLE_BONA_ONLY_SUPPORTED")},
        indent=2))


if __name__ == "__main__":
    main()
