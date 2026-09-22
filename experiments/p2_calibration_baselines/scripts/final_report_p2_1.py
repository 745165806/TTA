#!/usr/bin/env python
"""P2.1 final research-route decision (POST_HOC_DEVELOPMENT_ONLY)."""
import argparse
import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--historical-p2", type=str,
                        default=str(ROOT / "experiments/p2_calibration_baselines/results/run_20260922_155336"))
    args = parser.parse_args()
    run = Path(args.run_dir)

    cal = _read(Path(args.historical_p2) / "calibration/calibration_summary.json")
    oracle = _read(Path(args.historical_p2) / "oracle_teacher/oracle_teacher_summary.json")

    pilot = None
    pilot_path = run / "pilot/p2_1_pilot_summary.json"
    if pilot_path.is_file():
        pilot = _read(pilot_path)

    flags = {}
    # Calibration (section 37 semantics).
    flags["CALIBRATION_SHIFT_PRESENT"] = cal["flags"]["CALIBRATION_SHIFT_PRESENT"]
    flags["CALIBRATION_SHIFT_CONFIRMED"] = cal["flags"]["CALIBRATION_SHIFT_PRESENT"]
    flags["CALIBRATION_ONLY"] = (cal["flags"]["CALIBRATION_SHIFT_PRESENT"]
                                 and not cal["flags"]["RANKING_DEGRADATION_PRESENT"])
    flags["RANKING_DEGRADATION_PRESENT"] = cal["flags"]["RANKING_DEGRADATION_PRESENT"]
    flags["TEACHER_MISMATCH_SUPPORTED"] = oracle["TEACHER_MISMATCH_SUPPORTED"]
    flags["ALLOW_NEXT_ADAPTATION_RESEARCH"] = oracle["TEACHER_MISMATCH_SUPPORTED"]
    flags["STOP_NEW_EP_OBJECTIVES"] = not flags["ALLOW_NEXT_ADAPTATION_RESEARCH"]

    if pilot is not None:
        rows = {r["method"]: r for r in pilot["rows"]}
        for m in ("NormOnly", "TENT", "SAR", "MEMO"):
            flags["%s_TARGET10_GAIN" % m.upper().replace("NORMONLY", "NORMONLY")] = bool(
                rows.get(m, {}).get("GAIN"))
        # NormOnly diagnosis (single domain -> provisional).
        normonly = rows.get("NormOnly", {})
        flags["TEST_TIME_NORMALIZATION_HARM_SUPPORTED"] = bool(normonly.get("HARM"))
        flags["STANDARD_TTA_GAIN_ON_TARGET10_PILOT"] = sum(
            1 for m in ("TENT", "SAR", "MEMO") if rows.get(m, {}).get("GAIN"))

    # Next-priority recommendation (section 75).
    flags["NEXT_PRIORITY_1"] = "UNSUPERVISED_TARGET_CALIBRATION"
    flags["NEXT_PRIORITY_2"] = "RELIABLE_PSEUDO_TEACHER"
    flags["NEXT_PRIORITY_3"] = "SELECTIVE_EVIDENCE_PRESERVING_ADAPTATION"

    summary = {"POST_HOC_DEVELOPMENT_ONLY": True, "calibration": cal, "oracle_teacher": oracle,
               "pilot": pilot, "flags": flags}
    (run / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")

    with (run / "final_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["flag", "value"])
        for k, v in flags.items():
            writer.writerow([k, v])

    (run / "final_report.md").write_text(build_md(summary), encoding="utf-8")
    print(json.dumps({"flags": flags}, ensure_ascii=False, indent=2))


def build_md(s):
    f = s["flags"]
    c = s["calibration"]
    o = s["oracle_teacher"]
    lines = [
        "# P2.1 Published baseline validation + confirmatory",
        "",
        "**POST_HOC_DEVELOPMENT_ONLY**",
        "",
        "## Calibration (historical P2-A, read-only)",
        "- CALIBRATION_SHIFT_CONFIRMED = %s, CALIBRATION_ONLY = %s, RANKING_DEGRADATION_PRESENT = %s" %
          (f["CALIBRATION_SHIFT_CONFIRMED"], f["CALIBRATION_ONLY"], f["RANKING_DEGRADATION_PRESENT"]),
        "- tau0=%.4f tau_target_balacc=%.4f; bias restores FPR %.4f->%.4f (EER/AUC unchanged)" %
          (c["tau0"], c["tau_target_balacc"], c["frozen_fpr"], c["oracle_bias_fpr"]),
        "",
        "## Oracle teacher (historical P2-B, read-only)",
        "- TEACHER_MISMATCH_SUPPORTED = %s" % f["TEACHER_MISMATCH_SUPPORTED"],
        "- pseudo signed delta=%.4f, oracle signed delta=%+.4f" %
          (o["pseudo_teacher"]["mean_signed_task_delta"], o["oracle_teacher"]["mean_signed_task_delta"]),
        "",
        "## P2.1 target10 pilot (NormOnly / TENT / SAR / MEMO)",
    ]
    if s.get("pilot"):
        lines.append("| method | EER | AUC | dEER | dAUC | sdt_total | sdt_norm | sdt_update |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in s["pilot"]["rows"]:
            lines.append("| %s | %.4f | %.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f |" %
                         (r.get("method"), r.get("EER", 0), r.get("AUC", 0),
                          r.get("delta_EER_vs_frozen", 0), r.get("delta_AUC_vs_frozen", 0),
                          r.get("mean_signed_delta_total", 0), r.get("mean_signed_delta_norm", 0),
                          r.get("mean_signed_delta_update", 0)))
    lines += [
        "",
        "## Research route",
        "- ALLOW_NEXT_ADAPTATION_RESEARCH = %s, STOP_NEW_EP_OBJECTIVES = %s" %
          (f["ALLOW_NEXT_ADAPTATION_RESEARCH"], f["STOP_NEW_EP_OBJECTIVES"]),
        "- NEXT: %s -> %s -> %s" % (f["NEXT_PRIORITY_1"], f["NEXT_PRIORITY_2"], f["NEXT_PRIORITY_3"]),
        "",
        "NOTE: published ports are audited episodic audio ports under the project's "
        "no-target-history protocol; they are not claimed as exact reproductions of the "
        "original online evaluation protocol.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
