#!/usr/bin/env python
"""Final master summary (section 15).

Frozen is always the first baseline; new EP is compared against Frozen, not
only against old EP.  Reads P1 pilot summary and (if present) the confirmatory
summary, and emits final_summary.csv / .json / .md.
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

from scripts.common import POST_HOC_MARKER

FIELDS = ["dataset", "method", "EER", "AUC", "delta_EER_vs_frozen", "delta_AUC_vs_frozen",
          "adaptation_coverage", "helpful_flips", "harmful_flips",
          "mean_signed_task_delta", "mean_R_norm", "mean_abs_delta_score",
          "source_safety_reject_rate", "numeric_fallback_count"]


def row_from_stats(dataset, method, stats, frozen_eer, frozen_auc):
    return {
        "dataset": dataset, "method": method,
        "EER": stats["EER"], "AUC": stats["AUC"],
        "delta_EER_vs_frozen": stats["EER"] - frozen_eer,
        "delta_AUC_vs_frozen": stats["AUC"] - frozen_auc,
        "adaptation_coverage": stats.get("adaptation_coverage", 0.0),
        "helpful_flips": stats.get("helpful_flips", 0),
        "harmful_flips": stats.get("harmful_flips", 0),
        "mean_signed_task_delta": stats.get("mean_signed_task_delta", 0.0),
        "mean_R_norm": stats.get("mean_R_norm", 0.0),
        "mean_abs_delta_score": stats.get("mean_abs_delta_score", 0.0),
        "source_safety_reject_rate": stats.get("source_safety_reject_rate", 0.0),
        "numeric_fallback_count": stats.get("numeric_fallback_count", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    rows = []
    meta = {}

    p1_path = run_dir / "p1" / "p1_summary.json"
    p0_path = run_dir / "p0" / "p0_summary.json"
    if p1_path.is_file():
        p1 = json.loads(p1_path.read_text(encoding="utf-8"))
        frozen = p1["frozen"]
        frozen_eer, frozen_auc = frozen["EER"], frozen["AUC"]
        dataset = "in_the_wild_target10_development"
        rows.append(row_from_stats(dataset, "frozen", frozen, frozen_eer, frozen_auc))
        for method in ("taskaware_control", "taskaware_full", "taskaware_no_gate", "taskaware_no_source_keep"):
            stats = p1["variants"][method]
            rows.append(row_from_stats(dataset, method, stats, frozen_eer, frozen_auc))
        meta["p1_mechanism_pass"] = p1.get("P1_MECHANISM_PASS")
        meta["mechanism_gate"] = p1.get("mechanism_gate")

    confirm_path = run_dir / "confirmatory" / "confirmatory_summary.json"
    if confirm_path.is_file():
        confirm = json.loads(confirm_path.read_text(encoding="utf-8"))
        for d in confirm["datasets"]:
            frozen = d["frozen"]
            rows.append(row_from_stats(d["dataset"], "frozen", frozen, frozen["EER"], frozen["AUC"]))
            rows.append(row_from_stats(d["dataset"], "taskaware_full", d["taskaware_full"],
                                       frozen["EER"], frozen["AUC"]))
        meta["confirmatory_run"] = True
    else:
        meta["confirmatory_run"] = False

    out = run_dir
    with (out / "final_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    doc = {"POST_HOC_DEVELOPMENT_ONLY": True, "rows": rows, "meta": meta}
    (out / "final_summary.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    (out / "final_report.md").write_text(build_md(rows, meta), encoding="utf-8")

    print(json.dumps(doc, ensure_ascii=False, indent=2))


def build_md(rows, meta):
    lines = ["# P0/P1 final report", "", "**POST_HOC_DEVELOPMENT_ONLY**", "",
             "Frozen is always the first baseline.", "",
             "| dataset | method | EER | AUC | dEER_vs_frozen | dAUC_vs_frozen | coverage | helpful | harmful | mean_signed_delta | mean_R_norm | safety_reject_rate | fallback |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %.6f | %.6f | %+.6f | %+.6f | %.4f | %d | %d | %+.6f | %.6f | %.4f | %d |" %
                     (r["dataset"], r["method"], r["EER"], r["AUC"], r["delta_EER_vs_frozen"],
                      r["delta_AUC_vs_frozen"], r["adaptation_coverage"], r["helpful_flips"],
                      r["harmful_flips"], r["mean_signed_task_delta"], r["mean_R_norm"],
                      r["source_safety_reject_rate"], r["numeric_fallback_count"]))
    lines.append("")
    lines.append("mechanism gate (taskaware_full): %s" % json.dumps(meta.get("mechanism_gate")))
    lines.append("confirmatory_run: %s" % meta.get("confirmatory_run"))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
