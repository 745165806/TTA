#!/usr/bin/env python
"""Post-hoc target10 aggregation with same-path baselines and paired inference."""
import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "experiments/p2_calibration_baselines"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(P2))

from baselines.bootstrap import paired_bootstrap, significance
from eptta.evaluation.metrics import binary_metrics
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

BEFORE_TOLERANCE = 1e-5
METHODS = {
    "tent_audio_native_v1": ("TENT-Audio", "TENT_AUDIO_GAIN_SUPPORTED", "TENT_AUDIO_NEUTRAL"),
    "sar_audio_native_v1": ("SAR-Audio", "SAR_AUDIO_GAIN_SUPPORTED", "SAR_AUDIO_NEUTRAL"),
    "memo_audio_full_safeaug_v1": ("MEMO-FullSafeAug", "MEMO_FULL_SAFE_AUG_GAIN_SUPPORTED",
                                    "MEMO_FULL_SAFE_AUG_NEUTRAL"),
    "memo_audio_native_v1": ("MEMO-Audio-Limited", "MEMO_LIMITED_GAIN_SUPPORTED",
                             "MEMO_LIMITED_NEUTRAL"),
    "tent_audio_native_scope_b_v1": ("Scope-B", "SCOPE_B_GAIN_SUPPORTED", "SCOPE_B_NEUTRAL"),
}
REF_DIRS = {"NormOnly": "norm", "TENT-Episodic-Ref": "tent",
            "SAR-Episodic-Ref": "sar", "MEMO-Audio-Ref": "memo"}


def read_scores(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample ID in %s" % path)
    return result


def read_labels(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"]: int(row["label"]) for row in doc["records"]}


def compare(reference, method, labels):
    result = paired_bootstrap(reference, method, labels, seed=2026, n_bootstrap=2000)
    return {"bootstrap": result, "significance": significance(result)}


def supported_gain(comparison):
    sig = comparison["significance"]
    return bool(sig["EER_SIGNIFICANT_GAIN"] or sig["AUC_SIGNIFICANT_GAIN"])


def supported_harm(comparison):
    sig = comparison["significance"]
    return bool(sig["EER_SIGNIFICANT_HARM"] or sig["AUC_SIGNIFICANT_HARM"])


def conclusion(comparison, delta_eer, delta_auc):
    if supported_gain(comparison):
        return "GAIN_SUPPORTED"
    if supported_harm(comparison):
        return "SIGNIFICANT_HARM"
    if abs(delta_eer) <= 0.01 and abs(delta_auc) <= 0.01:
        return "NEUTRAL"
    return "NO_SIGNIFICANT_GAIN"


def before_path_diagnostics(records, tolerance=BEFORE_TOLERANCE):
    methods = sorted(records)
    if not methods:
        raise ValueError("no method records")
    ids = set(records[methods[0]])
    if any(set(records[method]) != ids for method in methods[1:]):
        raise ValueError("cross-method before-path coverage mismatch")
    ranges = []
    for sid in sorted(ids):
        values = [float(records[method][sid]["score_before_update"]) for method in methods]
        ranges.append(max(values) - min(values))
    return {"tolerance": tolerance, "sample_count": len(ids),
            "max_abs_before_diff_across_methods": max(ranges),
            "mean_abs_before_diff_across_methods": statistics.fmean(ranges),
            "AUDIO_NATIVE_BEFORE_PATH_PARITY_FAIL": max(ranges) > tolerance}


def cache_waveform_diagnostics(record):
    diffs = sorted(abs(float(row["score_before_update"]) - float(row["score_frozen_reference"]))
                   for row in record.values())
    p95_index = max(0, int(math.ceil(0.95 * len(diffs))) - 1)
    return {"diagnostic_role": "NUMERICAL_PATH_DIAGNOSTIC",
            "max_abs_cache_waveform_diff": max(diffs),
            "mean_abs_cache_waveform_diff": statistics.fmean(diffs),
            "p95_abs_cache_waveform_diff": diffs[p95_index]}


def metrics(scores, labels, order, tau0, frozen=None):
    return binary_metrics([scores[sid] for sid in order], [labels[sid] for sid in order], tau0,
                          frozen_scores=None if frozen is None else [frozen[sid] for sid in order])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path,
                        default=Path(os.environ.get("TTA_ASSET_ROOT", ROOT.parent / "TTA")))
    args = parser.parse_args()
    labels = read_labels(ROOT / "experiments/target10_selection/manifests/inwild_target10.json")
    order = sorted(labels)
    bundle_path = args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, _m, _p, _s = verify_frozen_export(bundle_path)
    resources, _extras, _meta = load_frozen_resources(
        args.asset_root / "outputs_v2/ssl_aasist/resources", bundle)
    tau0 = float(resources.tau0)

    records = {}
    for method_id, (label, _gain, _neutral) in METHODS.items():
        path = args.run_dir / method_id / "scores.jsonl"
        if not path.is_file():
            raise ValueError("missing formal method scores: %s" % path)
        records[label] = read_scores(path)
        if set(records[label]) != set(labels):
            raise ValueError("target score/label coverage mismatch: %s" % label)

    parity = before_path_diagnostics(records)
    canonical = records["TENT-Audio"]
    frozen_cache = {sid: float(canonical[sid]["score_frozen_reference"]) for sid in order}
    frozen_waveform = {sid: float(canonical[sid]["score_before_update"]) for sid in order}
    cache_diag = cache_waveform_diagnostics(canonical)
    fm_cache = metrics(frozen_cache, labels, order, tau0)
    fm_wave = metrics(frozen_waveform, labels, order, tau0)
    rows = [
        {"method": "Frozen-Cache", "EER": fm_cache["eer"], "AUC": fm_cache["auroc"],
         "balanced_accuracy_at_tau0": fm_cache["balanced_accuracy"], "FPR": fm_cache["fpr"],
         "FNR": fm_cache["fnr"], "baseline_role": "historical_continuity"},
        {"method": "Frozen-Waveform", "EER": fm_wave["eer"], "AUC": fm_wave["auroc"],
         "balanced_accuracy_at_tau0": fm_wave["balanced_accuracy"], "FPR": fm_wave["fpr"],
         "FNR": fm_wave["fnr"], "baseline_role": "primary_same_path"},
    ]
    comparisons, method_scores, method_rows = {}, {}, {}
    for method_id, (label, _gain, _neutral) in METHODS.items():
        rec = records[label]
        before = {sid: float(rec[sid]["score_before_update"]) for sid in order}
        after = {sid: float(rec[sid]["score_after"]) for sid in order}
        method_scores[label] = after
        before_metrics = metrics(before, labels, order, tau0)
        after_metrics = metrics(after, labels, order, tau0, frozen=before)
        comp = compare(before, after, labels)
        comparisons["%s_vs_Frozen-Waveform" % label] = comp
        signed_wave = [(2 * labels[sid] - 1) * (after[sid] - before[sid]) for sid in order]
        signed_cache = [(2 * labels[sid] - 1) * (after[sid] - frozen_cache[sid]) for sid in order]
        row = {
            "method": label, "method_id": method_id, "EER": after_metrics["eer"],
            "AUC": after_metrics["auroc"],
            "balanced_accuracy_at_tau0": after_metrics["balanced_accuracy"],
            "FPR": after_metrics["fpr"], "FNR": after_metrics["fnr"],
            "delta_EER_vs_Frozen_Waveform": after_metrics["eer"] - before_metrics["eer"],
            "delta_AUC_vs_Frozen_Waveform": after_metrics["auroc"] - before_metrics["auroc"],
            "delta_EER_vs_Frozen_Cache": after_metrics["eer"] - fm_cache["eer"],
            "delta_AUC_vs_Frozen_Cache": after_metrics["auroc"] - fm_cache["auroc"],
            "mean_signed_delta_vs_waveform_frozen": statistics.fmean(signed_wave),
            "mean_signed_delta_vs_cache_frozen": statistics.fmean(signed_cache),
            "bonafide_signed_delta": statistics.fmean(
                -(after[sid] - before[sid]) for sid in order if labels[sid] == 0),
            "spoof_signed_delta": statistics.fmean(
                after[sid] - before[sid] for sid in order if labels[sid] == 1),
            "mean_abs_delta": statistics.fmean(abs(after[sid] - before[sid]) for sid in order),
            "adaptation_coverage": sum(bool(rec[sid]["adaptation_applied"]) for sid in order) / len(order),
            "runtime_per_sample": statistics.fmean(float(rec[sid]["runtime"]) for sid in order),
            "numeric_failures": sum(bool(rec[sid]["numeric_failure"]) for sid in order),
            "resource_failures": sum(bool(rec[sid]["resource_failure"]) for sid in order),
            "ablation_only": method_id == "tent_audio_native_scope_b_v1",
        }
        row["conclusion_vs_waveform_frozen"] = conclusion(
            comp, row["delta_EER_vs_Frozen_Waveform"], row["delta_AUC_vs_Frozen_Waveform"])
        rows.append(row)
        method_rows[label] = row

    historical, historical_scores = {}, {}
    historical_root = (args.asset_root /
        "experiments/p2_calibration_baselines/results/p2_1_20260922_181728/pilot")
    for label, dirname in REF_DIRS.items():
        rec = read_scores(historical_root / dirname / "scores.jsonl")
        scores = {sid: float(rec[sid]["score_after"]) for sid in order}
        historical_scores[label] = scores
        value = metrics(scores, labels, order, tau0)
        historical[label] = {"EER": value["eer"], "AUC": value["auroc"]}

    recovery_pairs = {
        "TENT_REF_TO_AUDIO_RECOVERY_SUPPORTED": ("TENT-Episodic-Ref", "TENT-Audio"),
        "SAR_REF_TO_AUDIO_RECOVERY_SUPPORTED": ("SAR-Episodic-Ref", "SAR-Audio"),
        "MEMO_SAFE_AUG_RECOVERY_SUPPORTED": ("MEMO-Audio-Ref", "MEMO-FullSafeAug"),
    }
    for _flag, (reference_label, method_label) in recovery_pairs.items():
        comparisons["%s_vs_%s" % (method_label, reference_label)] = compare(
            historical_scores[reference_label], method_scores[method_label], labels)
    comparisons["MEMO-Audio-Limited_vs_MEMO-FullSafeAug"] = compare(
        method_scores["MEMO-FullSafeAug"], method_scores["MEMO-Audio-Limited"], labels)

    allowed = not parity["AUDIO_NATIVE_BEFORE_PATH_PARITY_FAIL"]
    flags = {"AUDIO_NATIVE_BEFORE_PATH_PARITY_FAIL": not allowed}
    for _method_id, (label, gain_flag, neutral_flag) in METHODS.items():
        comp = comparisons["%s_vs_Frozen-Waveform" % label]
        flags[gain_flag] = allowed and supported_gain(comp)
        flags[neutral_flag] = allowed and method_rows[label]["conclusion_vs_waveform_frozen"] == "NEUTRAL"
    for flag, (reference_label, method_label) in recovery_pairs.items():
        flags[flag] = allowed and supported_gain(
            comparisons["%s_vs_%s" % (method_label, reference_label)])

    taskaware = json.loads((ROOT /
        "experiments/p0_p1_taskaware/results/run_20260922_133805/final_summary.json").read_text())
    summary = {"POST_HOC_DEVELOPMENT_ONLY": True,
               "comparison_track": "audio_native_standard_tta",
               "scope": "SSL-AASIST / target10 development / episodic / tested mappings",
               "primary_frozen_baseline": "Frozen-Waveform", "tau0": tau0,
               "before_path_parity": parity, "cache_waveform_parity": cache_diag,
               "scientific_conclusions_allowed": allowed, "rows": rows,
               "historical_references": historical,
               "historical_EP_taskaware_reference": taskaware["rows"],
               "paired_bootstraps": comparisons, "scientific_flags": flags}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (args.run_dir / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
