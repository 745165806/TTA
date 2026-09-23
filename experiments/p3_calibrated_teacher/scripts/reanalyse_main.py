#!/usr/bin/env python
"""Statistical hardening of the existing main P3 run without rerunning adaptation."""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from followup_stats import (paired_mean_bootstrap, paired_ranking_bootstrap,
                            task_gain_supported)
from p3_common import load_config, load_ids, resolve_input


def records(path, ids):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise RuntimeError("main reanalysis record coverage mismatch: %s" % path)
    return [by_id[sid] for sid in ids]


def labels_for(config, ids):
    doc = json.loads(resolve_input(config["inputs"]["target10_label_manifest"]).read_text(
        encoding="utf-8"))
    labels = {row["sample_id"]: int(row["label"]) for row in doc["records"]}
    return [labels[sid] for sid in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_json = run_dir / "statistical_reanalysis.json"
    output_md = run_dir / "statistical_reanalysis.md"
    if output_json.exists() or output_md.exists():
        raise SystemExit("refusing to overwrite existing statistical reanalysis")
    config = load_config()
    ids = load_ids("evalU")
    labels = labels_for(config, ids)
    source = records(run_dir / "source_tau_teacher.jsonl", ids)
    calibrated = records(run_dir / "calibrated_selective.jsonl", ids)
    frozen = records(run_dir / "frozen_calonly.jsonl", ids)
    seed, n_boot = config["bootstrap"]["seed"], config["bootstrap"]["n"]

    teacher_differences = [
        int(cal["teacher_label"] == y) - int(src["teacher_label"] == y)
        for cal, src, y in zip(calibrated, source, labels)
    ]
    teacher = paired_mean_bootstrap(teacher_differences, seed=seed, n_boot=n_boot)
    signed = [(2 * y - 1) * (row["score_after"] - row["score_before"])
              for row, y in zip(calibrated, labels)]
    direction = paired_mean_bootstrap(signed, seed=seed, n_boot=n_boot)
    task = paired_ranking_bootstrap(
        [row["score_after"] for row in calibrated],
        [row["score_before"] for row in frozen], labels,
        seed=seed, n_boot=n_boot)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    result = {
        "source_run": str(run_dir), "adaptation_rerun": False,
        "bootstrap": {"seed": seed, "n": n_boot},
        "teacher_accuracy_delta": teacher,
        "signed_task_delta": direction,
        "CalibratedSelective_vs_Frozen": task,
        "P3_TEACHER_MECHANISM_SUPPORTED": teacher["ci95"][0] > 0,
        "P3_ADAPTATION_DIRECTION_SUPPORTED": direction["ci95"][0] > 0,
        "P3_TASK_GAIN_SUPPORTED": task_gain_supported(task),
        "P3_MECHANISM_PASS_LEGACY": bool(summary.get("P3_MECHANISM_PASS")),
        "legacy_definition_note": (
            "Legacy PASS accepted any strictly improved EER or AUC point estimate; "
            "it did not require paired-bootstrap significance for task gain."
        ),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# P3 main-split statistical reanalysis", "",
        "No adaptation was rerun; this analysis reads the immutable main-run records.", "",
        "- teacher accuracy delta: `%+.8f`, 95%% CI `%s`" %
        (teacher["estimate"], teacher["ci95"]),
        "- signed task delta: `%+.8f`, 95%% CI `%s`" %
        (direction["estimate"], direction["ci95"]),
        "- delta EER: `%+.8f`, 95%% CI `%s`" %
        (task["delta_EER"], task["delta_EER_ci95"]),
        "- delta AUC: `%+.8f`, 95%% CI `%s`" %
        (task["delta_AUC"], task["delta_AUC_ci95"]), "",
        "- P3_TEACHER_MECHANISM_SUPPORTED: `%s`" % result["P3_TEACHER_MECHANISM_SUPPORTED"],
        "- P3_ADAPTATION_DIRECTION_SUPPORTED: `%s`" % result["P3_ADAPTATION_DIRECTION_SUPPORTED"],
        "- P3_TASK_GAIN_SUPPORTED: `%s`" % result["P3_TASK_GAIN_SUPPORTED"],
        "- P3_MECHANISM_PASS_LEGACY: `%s`" % result["P3_MECHANISM_PASS_LEGACY"], "",
        "Legacy PASS was over-broad because any favorable EER/AUC point estimate was sufficient; "
        "the hardened task-gain conclusion requires a paired-bootstrap CI to exclude zero.", "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
