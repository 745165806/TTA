#!/usr/bin/env python
"""Post-hoc target10 development aggregation; this is the only label reader."""
import argparse
import csv
import json
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

METHOD_LABELS = {
    "tent_audio_native_v1": "TENT-Audio",
    "sar_audio_native_v1": "SAR-Audio",
    "memo_audio_native_v1": "MEMO-Audio-v1",
    "tent_audio_native_scope_b_v1": "AudioNative-Scope-B",
}
REF_DIRS = {"TENT-Episodic-Ref": "tent", "SAR-Episodic-Ref": "sar",
            "MEMO-Audio-Ref": "memo", "NormOnly historical control": "norm"}


def read_scores(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["sample_id"]: row for row in rows}


def read_labels(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"]: int(row["label"]) for row in doc["records"]}


def compare(reference, method, labels):
    result = paired_bootstrap(reference, method, labels, seed=2026, n_bootstrap=2000)
    return {"bootstrap": result, "significance": significance(result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path,
                        default=Path(os.environ.get("TTA_ASSET_ROOT", ROOT.parent / "TTA")))
    args = parser.parse_args()
    labels = read_labels(ROOT / "experiments/target10_selection/manifests/inwild_target10.json")
    bundle_path = args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, _m, _p, _s = verify_frozen_export(bundle_path)
    resources, _extras, _meta = load_frozen_resources(
        args.asset_root / "outputs_v2/ssl_aasist/resources", bundle)
    tau0 = float(resources.tau0)

    methods = {}
    records = {}
    for method_id, label in METHOD_LABELS.items():
        path = args.run_dir / method_id / "scores.jsonl"
        if path.is_file():
            records[label] = read_scores(path)
    if not records:
        raise ValueError("no audio-native score files found")
    first = next(iter(records.values()))
    frozen = {sid: row["score_frozen_reference"] for sid, row in first.items()}
    if set(frozen) != set(labels):
        raise ValueError("target score/label coverage mismatch")
    order = sorted(labels)
    frozen_metrics = binary_metrics([frozen[sid] for sid in order],
                                    [labels[sid] for sid in order], tau0)
    rows = [{"method": "Frozen", "EER": frozen_metrics["eer"],
             "AUC": frozen_metrics["auroc"],
             "balanced_accuracy_at_tau0": frozen_metrics["balanced_accuracy"],
             "FPR": frozen_metrics["fpr"], "FNR": frozen_metrics["fnr"],
             "delta_EER_vs_Frozen": 0.0, "delta_AUC_vs_Frozen": 0.0,
             "adaptation_coverage": 0.0}]
    bootstraps = {}
    for label, rec in records.items():
        if set(rec) != set(labels):
            raise ValueError("incomplete method coverage: %s" % label)
        scores = {sid: rec[sid]["score_after"] for sid in order}
        metric = binary_metrics([scores[sid] for sid in order], [labels[sid] for sid in order],
                                tau0, frozen_scores=[frozen[sid] for sid in order])
        bootstraps["%s_vs_Frozen" % label] = compare(frozen, scores, labels)
        signed = [(2 * labels[sid] - 1) * (scores[sid] - frozen[sid]) for sid in order]
        signed_bona = [-1.0 * (scores[sid] - frozen[sid]) for sid in order if labels[sid] == 0]
        signed_spoof = [scores[sid] - frozen[sid] for sid in order if labels[sid] == 1]
        rows.append({
            "method": label, "EER": metric["eer"], "AUC": metric["auroc"],
            "balanced_accuracy_at_tau0": metric["balanced_accuracy"],
            "FPR": metric["fpr"], "FNR": metric["fnr"],
            "delta_EER_vs_Frozen": metric["eer"] - frozen_metrics["eer"],
            "delta_AUC_vs_Frozen": metric["auroc"] - frozen_metrics["auroc"],
            "mean_signed_task_delta": statistics.fmean(signed),
            "bonafide_signed_delta": statistics.fmean(signed_bona),
            "spoof_signed_delta": statistics.fmean(signed_spoof),
            "mean_abs_delta": statistics.fmean(abs(scores[sid] - frozen[sid]) for sid in order),
            "adaptation_coverage": sum(bool(rec[sid]["adaptation_applied"]) for sid in order) / len(order),
            "runtime_per_sample": statistics.fmean(rec[sid]["runtime"] for sid in order),
            "numeric_failures": sum(bool(rec[sid]["numeric_failure"]) for sid in order),
            "resource_failures": sum(bool(rec[sid]["resource_failure"]) for sid in order),
        })
        methods[label] = scores

    historical = {}
    historical_root = (args.asset_root /
        "experiments/p2_calibration_baselines/results/p2_1_20260922_181728/pilot")
    for label, dirname in REF_DIRS.items():
        path = historical_root / dirname / "scores.jsonl"
        if path.is_file():
            ref_rows = read_scores(path)
            ref_scores = {sid: ref_rows[sid]["score_after"] for sid in order}
            metric = binary_metrics([ref_scores[sid] for sid in order],
                                    [labels[sid] for sid in order], tau0)
            historical[label] = {"EER": metric["eer"], "AUC": metric["auroc"]}
            target_label = {"TENT-Episodic-Ref": "TENT-Audio",
                            "SAR-Episodic-Ref": "SAR-Audio",
                            "MEMO-Audio-Ref": "MEMO-Audio-v1"}.get(label)
            if target_label in methods:
                bootstraps["%s_vs_%s" % (target_label, label)] = compare(
                    ref_scores, methods[target_label], labels)

    if "TENT-Audio" in methods and "SAR-Audio" in methods:
        bootstraps["SAR-Audio_vs_TENT-Audio"] = compare(
            methods["TENT-Audio"], methods["SAR-Audio"], labels)
    tent_sig = bootstraps.get("TENT-Audio_vs_Frozen", {}).get("significance", {})
    sar_tent_sig = bootstraps.get("SAR-Audio_vs_TENT-Audio", {}).get("significance", {})
    tent_row = next((row for row in rows if row["method"] == "TENT-Audio"), None)
    significant_gain = tent_sig.get("EER_SIGNIFICANT_GAIN", False) or tent_sig.get("AUC_SIGNIFICANT_GAIN", False)
    significant_harm = tent_sig.get("EER_SIGNIFICANT_HARM", False) or tent_sig.get("AUC_SIGNIFICANT_HARM", False)
    flags = {
        "AUDIO_NATIVE_ENTROPY_TTA_SUPPORTED": significant_gain,
        "AUDIO_NATIVE_RELIABLE_SAM_SUPPORTED": (
            sar_tent_sig.get("EER_SIGNIFICANT_GAIN", False) or
            sar_tent_sig.get("AUC_SIGNIFICANT_GAIN", False)),
        "AUDIO_NATIVE_TTA_NEUTRAL": bool(tent_row and not significant_gain and not significant_harm and
            abs(tent_row["delta_EER_vs_Frozen"]) <= 0.01 and
            abs(tent_row["delta_AUC_vs_Frozen"]) <= 0.01),
        "AUDIO_NATIVE_ENTROPY_SIGNAL_HARM_SUPPORTED": significant_harm,
    }
    taskaware = json.loads((ROOT /
        "experiments/p0_p1_taskaware/results/run_20260922_133805/final_summary.json").read_text())
    summary = {
        "POST_HOC_DEVELOPMENT_ONLY": True, "comparison_track": "audio_native_standard_tta",
        "scope": "SSL-AASIST / target10 development / episodic / tested mappings",
        "tau0": tau0, "rows": rows, "historical_references": historical,
        "historical_EP_taskaware_reference": taskaware["rows"],
        "bootstraps": bootstraps, "scientific_flags": flags,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (args.run_dir / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
