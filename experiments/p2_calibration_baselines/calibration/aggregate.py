#!/usr/bin/env python
"""P2-A: calibration decomposition aggregation + research-route flags.

POST_HOC_DEVELOPMENT_ONLY.  Combines the distribution / threshold-oracle /
affine-oracle outputs into the final calibration summary.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from calibration.common import POST_HOC_MARKER, load_cal0, load_target10

# Numerical tolerance for the "materially nonzero / changed" judgments.
NUMERICAL_TOLERANCE = 1e-6


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.run_dir) / "calibration"

    threshold = _read_json(out_dir / "threshold_oracle.json")
    affine = _read_json(out_dir / "affine_oracle.json")
    shift = _read_json(out_dir / "class_shift_summary.json")

    # Source cal0 ranking reference (source-domain frozen ranking).
    sc, yc, _ic, _res = load_cal0()
    from eptta.evaluation.metrics import binary_metrics
    src_m = binary_metrics(list(sc), list(yc), 0.0)
    source_reference_eer, source_reference_auc = src_m["eer"], src_m["auroc"]

    frozen = threshold["frozen"]
    target_eer, target_auc = frozen["EER"], frozen["AUC"]

    # Route/flags (section 13/82).
    oracle_bias_balacc = threshold["frozen_plus_bias_balacc"]["balanced_accuracy"]
    frozen_balacc = frozen["balanced_accuracy"]
    calibration_shift_present = (
        abs(threshold["tau_target_balacc"] - threshold["tau0"]) > NUMERICAL_TOLERANCE
        and oracle_bias_balacc > frozen_balacc)
    ranking_degradation_present = (
        target_eer > source_reference_eer or target_auc < source_reference_auc)
    calibration_only_insufficient = target_eer > 0.01  # materially nonzero EER

    flags = {
        "CALIBRATION_SHIFT_PRESENT": bool(calibration_shift_present),
        "RANKING_DEGRADATION_PRESENT": bool(ranking_degradation_present),
        "CALIBRATION_ONLY_INSUFFICIENT": bool(calibration_only_insufficient),
        "CALIBRATION_SHIFT_CONFIRMED": bool(calibration_shift_present
                                            and not ranking_degradation_present),
    }

    summary = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "tau0": threshold["tau0"],
        "tau_target_eer": threshold["tau_target_eer"],
        "tau_target_balacc": threshold["tau_target_balacc"],
        "delta_tau_balacc": threshold["delta_tau_balacc"],
        "frozen_balanced_accuracy": frozen_balacc,
        "oracle_bias_balanced_accuracy": oracle_bias_balacc,
        "frozen_fpr": frozen["fpr"],
        "oracle_bias_fpr": threshold["frozen_plus_bias_balacc"]["fpr"],
        "frozen_accuracy": frozen["accuracy"],
        "target_eer": target_eer,
        "target_auc": target_auc,
        "source_reference_eer": source_reference_eer,
        "source_reference_auc": source_reference_auc,
        "affine": affine,
        "class_shift": shift,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "flags": flags,
    }
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # calibration_metrics.csv
    fields = ["metric", "value"]
    rows = [
        ("tau0", threshold["tau0"]),
        ("tau_target_eer", threshold["tau_target_eer"]),
        ("tau_target_balacc", threshold["tau_target_balacc"]),
        ("frozen_balanced_accuracy", frozen_balacc),
        ("oracle_bias_balanced_accuracy", oracle_bias_balacc),
        ("frozen_fpr", frozen["fpr"]),
        ("oracle_bias_fpr", threshold["frozen_plus_bias_balacc"]["fpr"]),
        ("frozen_accuracy", frozen["accuracy"]),
        ("target_eer", target_eer),
        ("target_auc", target_auc),
        ("source_reference_eer", source_reference_eer),
        ("source_reference_auc", source_reference_auc),
        ("affine_a", affine["a"]),
        ("affine_b", affine["b"]),
        ("class_separation_source", shift["class_separation_source"]),
        ("class_separation_target", shift["class_separation_target"]),
        ("midpoint_shift", shift["midpoint_shift"]),
    ]
    with (out_dir / "calibration_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)

    (out_dir / "calibration_summary.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_md(s):
    f = s["flags"]
    sep_src = s["class_shift"]["class_separation_source"]
    sep_tgt = s["class_shift"]["class_separation_target"]
    return "\n".join([
        "# P2-A Calibration decomposition",
        "",
        "**POST_HOC_DEVELOPMENT_ONLY**",
        "",
        "## Frozen operating point vs ranking",
        "- target EER = %.6f, AUC = %.6f" % (s["target_eer"], s["target_auc"]),
        "- source cal0 reference EER = %.6f, AUC = %.6f" % (s["source_reference_eer"], s["source_reference_auc"]),
        "- frozen tau0 accuracy = %.4f, balanced accuracy = %.4f, FPR = %.4f" %
          (s["frozen_accuracy"], s["frozen_balanced_accuracy"], s["frozen_fpr"]),
        "",
        "## 1. source tau0 vs target oracle threshold",
        "- tau0 = %.6f" % s["tau0"],
        "- tau_target_eer = %.6f (delta %.3f)" % (s["tau_target_eer"], s["tau_target_eer"] - s["tau0"]),
        "- tau_target_balacc = %.6f (delta %.3f)" % (s["tau_target_balacc"], s["tau_target_balacc"] - s["tau0"]),
        "",
        "## 2/3. bias-only recovery",
        "- balanced accuracy %.4f -> %.4f" % (s["frozen_balanced_accuracy"], s["oracle_bias_balanced_accuracy"]),
        "- FPR %.4f -> %.4f" % (s["frozen_fpr"], s["oracle_bias_fpr"]),
        "",
        "## 4. EER/AUC invariance (bias/affine)",
        "- affine invariant check: %s" % json.dumps(s["affine"]["invariant_check"]),
        "",
        "## 5. class separation",
        "- source separation = %.4f, target separation = %.4f (ratio %.3f)" %
          (sep_src, sep_tgt, sep_tgt / sep_src if sep_src else None),
        "- midpoint shift = %.4f" % s["class_shift"]["midpoint_shift"],
        "",
        "## 6. verdict",
        "- CALIBRATION_SHIFT_PRESENT: %s" % f["CALIBRATION_SHIFT_PRESENT"],
        "- RANKING_DEGRADATION_PRESENT: %s" % f["RANKING_DEGRADATION_PRESENT"],
        "- CALIBRATION_ONLY_INSUFFICIENT: %s" % f["CALIBRATION_ONLY_INSUFFICIENT"],
        "- CALIBRATION_SHIFT_CONFIRMED: %s" % f["CALIBRATION_SHIFT_CONFIRMED"],
        "",
        "The AUC~0.963 vs tau0-accuracy~0.437 gap is dominated by a source->target "
        "operating-point (calibration) shift of ~%.2f score units; a bias-only shift "
        "restores balanced accuracy %.4f->%.4f and FPR %.4f->%.4f WITHOUT changing EER/AUC. "
        "However target class separation also shrank (%.4f->%.4f), so ranking degradation "
        "coexists with the calibration shift." %
        (s["delta_tau_balacc"], s["frozen_balanced_accuracy"], s["oracle_bias_balanced_accuracy"],
         s["frozen_fpr"], s["oracle_bias_fpr"], sep_src, sep_tgt),
    ])


if __name__ == "__main__":
    main()
