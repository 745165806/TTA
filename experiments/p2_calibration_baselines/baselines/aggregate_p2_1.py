#!/usr/bin/env python
"""P2.1 target10 pilot aggregation: NormOnly / TENT / SAR / MEMO with 3-score semantics.

POST_HOC_DEVELOPMENT_ONLY.
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

from baselines.bootstrap import paired_bootstrap, significance
from baselines import port_validation
from baselines.provenance import git_commit
from calibration.common import load_target10

METHODS = {"norm": "NormOnly", "tent": "TENT", "sar": "SAR", "memo": "MEMO"}
# Key -> published method_id (None for control/reference).
METHOD_IDS = {"norm": None, "tent": "tent_audio_ep", "sar": "sar_audio_ep",
              "memo": "memo_audio_ep_full"}


def load_scores(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return {r["sample_id"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--validation-evidence", type=str, required=True)
    args = parser.parse_args()
    pilot = Path(args.pilot_dir)
    out_dir = Path(args.run_dir) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, ys, sample_ids, resources = load_target10()
    labels = {sid: y for sid, y in zip(sample_ids, ys)}
    tau0 = resources.tau0

    frozen_scores = {sid: s for sid, s in zip(sample_ids, scores)}

    from eptta.evaluation.metrics import binary_metrics
    order = sorted(frozen_scores)
    fm = binary_metrics([frozen_scores[k] for k in order], [labels[k] for k in order], tau0)

    rows = [{"method": "Frozen", "EER": fm["eer"], "AUC": fm["auroc"],
             "balanced_accuracy_at_tau0": fm["balanced_accuracy"],
             "FPR": fm["fpr"], "FNR": fm["fnr"], "delta_EER_vs_frozen": 0.0,
             "delta_AUC_vs_frozen": 0.0, "PORT_VALID": False, "CONTROL_VALID": False,
             "status": "SCORED"}]

    # Direct parity evidence (hard gate).
    parity_path = Path(args.run_dir) / "parity/waveform_frozen_parity.json"
    direct_parity_pass = False
    if parity_path.is_file():
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        direct_parity_pass = parity.get("all_within_project_tolerance", False)

    bootstraps = {}
    validations = {}
    for key, label in METHODS.items():
        path = pilot / ("%s/scores.jsonl" % key)
        if not path.is_file():
            rows.append({"method": label, "status": "NOT_RUN"})
            continue
        rec = load_scores(path)
        after = {sid: rec[sid]["score_after"] for sid in sample_ids if sid in rec}
        sample_coverage = len(after) / len(sample_ids)
        adaptation_coverage = (
            sum(bool(r["adaptation_applied"]) for r in rec.values()) / len(rec)
            if rec else 0.0)

        m = binary_metrics([after[k] for k in order], [labels[k] for k in order], tau0,
                           frozen_scores=[frozen_scores[k] for k in order])
        signed_total = [(2 * labels[sid] - 1) * (rec[sid]["score_after"] - frozen_scores[sid])
                        for sid in after]
        signed_norm = [(2 * labels[sid] - 1) * (rec[sid]["score_norm_only"] - frozen_scores[sid])
                       for sid in after]
        signed_update = [(2 * labels[sid] - 1) * (rec[sid]["score_after"] - rec[sid]["score_norm_only"])
                         for sid in after]

        b = paired_bootstrap(frozen_scores, after, labels)
        sig = significance(b)
        bootstraps[label] = {"bootstrap": b, "significance": sig}
        gain = sig["EER_SIGNIFICANT_GAIN"] or sig["AUC_SIGNIFICANT_GAIN"]
        harm = sig["EER_SIGNIFICANT_HARM"] or sig["AUC_SIGNIFICANT_HARM"]

        numeric_failure = sum(1 for r in rec.values() if r.get("numeric_failure"))
        resource_failure = sum(1 for r in rec.values() if r.get("resource_failure"))

        method_id = METHOD_IDS[key]
        if method_id is None:  # NormOnly control
            validation = port_validation.compute_control_validation(
                "norm_only_audio", sample_coverage, numeric_failure, resource_failure)
            port_valid = False
        else:
            audit = port_validation.load_audit(method_id)
            validation = port_validation.compute_evidence_backed_validation(
                method_id, audit, args.validation_evidence, direct_parity_pass,
                sample_coverage, numeric_failure, resource_failure, git_commit())
            port_valid = validation["PORT_VALID"]
        validations[label] = validation

        rows.append({
            "method": label, "status": "SCORED",
            "EER": m["eer"], "AUC": m["auroc"],
            "balanced_accuracy_at_tau0": m["balanced_accuracy"],
            "FPR": m["fpr"], "FNR": m["fnr"],
            "delta_EER_vs_frozen": b["delta_EER"]["mean"],
            "delta_AUC_vs_frozen": b["delta_AUC"]["mean"],
            "mean_signed_delta_total": statistics.fmean(signed_total),
            "mean_signed_delta_norm": statistics.fmean(signed_norm),
            "mean_signed_delta_update": statistics.fmean(signed_update),
            "mean_abs_delta_total": statistics.fmean(abs(r["delta_total"]) for r in rec.values()),
            "mean_abs_delta_norm": statistics.fmean(abs(r["delta_norm"]) for r in rec.values()),
            "mean_abs_delta_update": statistics.fmean(abs(r["delta_update"]) for r in rec.values()),
            "sample_coverage": sample_coverage,
            "adaptation_coverage": adaptation_coverage,
            "runtime_per_sample": statistics.fmean(r["runtime"] for r in rec.values()),
            "numeric_failure": numeric_failure,
            "resource_failure": resource_failure,
            "PORT_VALID": port_valid,
            "CONTROL_VALID": validation.get("CONTROL_VALID", False),
            "GAIN": gain, "HARM": harm,
        })

    fields = ["method", "status", "EER", "AUC", "balanced_accuracy_at_tau0", "FPR", "FNR",
              "delta_EER_vs_frozen", "delta_AUC_vs_frozen",
              "mean_signed_delta_total", "mean_signed_delta_norm", "mean_signed_delta_update",
              "mean_abs_delta_total", "mean_abs_delta_norm", "mean_abs_delta_update",
              "sample_coverage", "adaptation_coverage", "runtime_per_sample",
              "numeric_failure", "resource_failure", "PORT_VALID", "CONTROL_VALID", "GAIN", "HARM"]
    with (out_dir / "p2_1_pilot_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"POST_HOC_DEVELOPMENT_ONLY": True, "tau0": tau0, "frozen": {
        "EER": fm["eer"], "AUC": fm["auroc"], "balanced_accuracy_at_tau0": fm["balanced_accuracy"],
        "FPR": fm["fpr"], "FNR": fm["fnr"]}, "rows": rows, "bootstraps": bootstraps,
        "port_validations": validations}
    (out_dir / "p2_1_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "port_validation.json").write_text(
        json.dumps({"POST_HOC_DEVELOPMENT_ONLY": True, "validations": validations,
                    "direct_parity_pass": direct_parity_pass},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
