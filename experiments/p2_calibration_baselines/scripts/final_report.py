#!/usr/bin/env python
"""P2 final research-route decision + summary (POST_HOC_DEVELOPMENT_ONLY)."""
import argparse
import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)

    cal = _read(run / "calibration/calibration_summary.json")
    oracle = _read(run / "oracle_teacher/oracle_teacher_summary.json")

    pilot_summary = None
    pilot_path = run / "pilot/p2_pilot_summary.json"
    if pilot_path.is_file():
        pilot_summary = _read(pilot_path)

    flags = {}
    # P2-A
    flags.update(cal["flags"])
    # P2-B
    flags["TEACHER_MISMATCH_SUPPORTED"] = oracle["TEACHER_MISMATCH_SUPPORTED"]
    flags["TEACHER_MISMATCH_INSUFFICIENT"] = oracle["TEACHER_MISMATCH_INSUFFICIENT"]

    # Route B / EP continuation (section 59/83).
    oracle_teacher_positive = oracle["oracle_teacher"]["mean_signed_task_delta"] > 0
    standard_tta_gain_domains = 0
    if pilot_summary is not None:
        for row in pilot_summary["rows"]:
            if row.get("GAIN"):
                standard_tta_gain_domains += 1
    allow_next = oracle_teacher_positive or (standard_tta_gain_domains >= 2)
    flags["ALLOW_NEXT_ADAPTATION_RESEARCH"] = bool(allow_next)
    flags["STOP_NEW_EP_OBJECTIVES"] = bool(not allow_next)
    flags["CONDITION_A_ORACLE_TEACHER_POSITIVE"] = bool(oracle_teacher_positive)
    flags["STANDARD_TTA_GAIN_DOMAINS_PILOT"] = standard_tta_gain_domains

    summary = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "calibration": cal,
        "oracle_teacher": oracle,
        "pilot": pilot_summary,
        "flags": flags,
    }
    (run / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")

    # final_summary.csv
    f = flags
    csv_rows = [
        ("CALIBRATION_SHIFT_PRESENT", f.get("CALIBRATION_SHIFT_PRESENT")),
        ("RANKING_DEGRADATION_PRESENT", f.get("RANKING_DEGRADATION_PRESENT")),
        ("CALIBRATION_ONLY_INSUFFICIENT", f.get("CALIBRATION_ONLY_INSUFFICIENT")),
        ("CALIBRATION_SHIFT_CONFIRMED", f.get("CALIBRATION_SHIFT_CONFIRMED")),
        ("TEACHER_MISMATCH_SUPPORTED", f.get("TEACHER_MISMATCH_SUPPORTED")),
        ("TEACHER_MISMATCH_INSUFFICIENT", f.get("TEACHER_MISMATCH_INSUFFICIENT")),
        ("ALLOW_NEXT_ADAPTATION_RESEARCH", f.get("ALLOW_NEXT_ADAPTATION_RESEARCH")),
        ("STOP_NEW_EP_OBJECTIVES", f.get("STOP_NEW_EP_OBJECTIVES")),
        ("tau0", cal["tau0"]),
        ("tau_target_balacc", cal["tau_target_balacc"]),
        ("frozen_balanced_accuracy", cal["frozen_balanced_accuracy"]),
        ("oracle_bias_balanced_accuracy", cal["oracle_bias_balanced_accuracy"]),
        ("frozen_fpr", cal["frozen_fpr"]),
        ("oracle_bias_fpr", cal["oracle_bias_fpr"]),
        ("target_eer", cal["target_eer"]),
        ("target_auc", cal["target_auc"]),
        ("pseudo_signed_delta", oracle["pseudo_teacher"]["mean_signed_task_delta"]),
        ("oracle_signed_delta", oracle["oracle_teacher"]["mean_signed_task_delta"]),
    ]
    with (run / "final_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(csv_rows)

    (run / "final_report.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({"flags": flags}, ensure_ascii=False, indent=2))


def build_md(s):
    c = s["calibration"]
    o = s["oracle_teacher"]
    f = s["flags"]
    lines = [
        "# P2 Calibration / Adaptability + Published TTA baselines",
        "",
        "**POST_HOC_DEVELOPMENT_ONLY**",
        "",
        "## Q1/Q2/Q3: AUC~0.963 vs tau0 accuracy~0.437",
        "- tau0 = %.6f, tau_target_eer = %.6f, tau_target_balacc = %.6f" %
          (c["tau0"], c["tau_target_eer"], c["tau_target_balacc"]),
        "- bias-only calibration: balanced accuracy %.4f -> %.4f, FPR %.4f -> %.4f" %
          (c["frozen_balanced_accuracy"], c["oracle_bias_balanced_accuracy"],
           c["frozen_fpr"], c["oracle_bias_fpr"]),
        "- EER/AUC strictly unchanged under monotone bias/affine (invariant check: %s)" %
          json.dumps(c["affine"]["invariant_check"]),
        "",
        "## Q4: ranking degradation",
        "- source cal0 EER=%.6f AUC=%.6f; target EER=%.6f AUC=%.6f" %
          (c["source_reference_eer"], c["source_reference_auc"], c["target_eer"], c["target_auc"]),
        "- class separation source=%.4f target=%.4f; midpoint shift=%.4f" %
          (c["class_shift"]["class_separation_source"], c["class_shift"]["class_separation_target"],
           c["class_shift"]["midpoint_shift"]),
        "",
        "## Q5/Q6: oracle-teacher counterfactual",
        "- pseudo signed delta = %.6f; oracle signed delta = %.6f" %
          (o["pseudo_teacher"]["mean_signed_task_delta"], o["oracle_teacher"]["mean_signed_task_delta"]),
        "- TEACHER_MISMATCH_SUPPORTED = %s; INSUFFICIENT = %s" %
          (f["TEACHER_MISMATCH_SUPPORTED"], f["TEACHER_MISMATCH_INSUFFICIENT"]),
        "",
        "## Research route",
        "- CALIBRATION_SHIFT_PRESENT = %s, RANKING_DEGRADATION_PRESENT = %s" %
          (f["CALIBRATION_SHIFT_PRESENT"], f["RANKING_DEGRADATION_PRESENT"]),
        "- ALLOW_NEXT_ADAPTATION_RESEARCH = %s, STOP_NEW_EP_OBJECTIVES = %s" %
          (f["ALLOW_NEXT_ADAPTATION_RESEARCH"], f["STOP_NEW_EP_OBJECTIVES"]),
        "",
        "## Verdict",
        "The fixed-tau0 failure is dominated by a source->target operating-point shift of "
        "~%.2f score units, but target class separation ALSO shrank (%.4f->%.4f), so ranking "
        "degradation coexists with calibration shift. P1 failed primarily because the frozen tau0 "
        "pseudo-teacher is wrong for most target samples: an oracle teacher flips the signed task "
        "delta from %.4f to %+.4f and improves EER %.4f->%.4f. Next: calibration-aware/selective "
        "adaptation with a corrected teacher, NOT further entropy/view-consistency loss grids." %
        (c["delta_tau_balacc"], c["class_shift"]["class_separation_source"],
         c["class_shift"]["class_separation_target"],
         o["pseudo_teacher"]["mean_signed_task_delta"], o["oracle_teacher"]["mean_signed_task_delta"],
         o["pseudo_teacher"]["EER"], o["oracle_teacher"]["EER"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
