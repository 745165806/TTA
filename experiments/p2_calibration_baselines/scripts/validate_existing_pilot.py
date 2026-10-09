#!/usr/bin/env python
"""Validate an immutable historical target10 pilot against the current code."""
import argparse
import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR))

from baselines.locked_config import PUBLISHED_METHODS, load_locked_document
from baselines.port_validation import (compute_evidence_backed_validation,
                                       load_audit, load_validation_evidence)
from baselines.provenance import (git_commit, path_introduction_commit,
                                  require_clean_tree)
from baselines.splits import load_split_sample_ids

METHOD_OUTPUTS = {
    "tent_audio_ep": ("TENT", "tent"),
    "sar_audio_ep": ("SAR", "sar"),
    "memo_audio_ep_full": ("MEMO", "memo"),
}


def _load_json(path, description):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("%s is missing or invalid: %s" % (description, path)) from exc


def _load_score_records(path):
    records = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("historical score records are missing or invalid: %s" % path) from exc
    return records


def validate_existing_pilot(historical_run, output, evidence_path, locked_config,
                            parity_artifact=None, current_commit=None):
    historical_run = Path(historical_run).resolve()
    output = Path(output).resolve()
    parity_source_path = Path(parity_artifact).resolve() if parity_artifact else None
    protected_outputs = (
        output / "source_run_ref.json", output / "pilot/port_validation.json",
        output / "validation_summary.json", output / "validation_summary.md")
    parity_out = output / "parity/waveform_frozen_parity.json"
    if (any(path.exists() for path in protected_outputs)
            or (parity_out.exists() and parity_source_path != parity_out)):
        raise ValueError("refusing to overwrite validation run: %s" % output)
    if current_commit is None:
        require_clean_tree()
        current_commit = git_commit()
    summary_path = historical_run / "pilot/p2_1_pilot_summary.json"
    metrics_path = historical_run / "pilot/p2_1_pilot_metrics.csv"
    historical_parity = historical_run / "parity/waveform_frozen_parity.json"
    _load_json(summary_path, "historical pilot summary")
    try:
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError("historical pilot metrics are missing") from exc
    _load_json(historical_parity, "historical parity provenance")

    locked_content = load_locked_document(locked_config)
    for method_id in PUBLISHED_METHODS:
        if load_validation_evidence(evidence_path, method_id, current_commit) is None:
            raise ValueError("validation evidence is absent or stale for %s" % method_id)

    if parity_artifact is None:
        parity = {
            "schema_version": "0.2.0", "git_commit": current_commit,
            "status": "NOT_RUN_RESOURCE", "real_waveform_parity_executed": False,
            "all_within_project_tolerance": False,
        }
        direct_parity_pass = False
        parity_source = None
    else:
        parity_source = str(parity_source_path)
        parity = _load_json(parity_artifact, "current parity artifact")
        if parity.get("git_commit") != current_commit:
            raise ValueError("current parity artifact git commit mismatch")
        direct_parity_pass = parity.get("all_within_project_tolerance") is True

    expected_ids, _ = load_split_sample_ids("target10")
    expected = set(expected_ids)
    validations = {}
    for method_id, (label, directory) in METHOD_OUTPUTS.items():
        records = _load_score_records(historical_run / "pilot" / directory / "scores.jsonl")
        ids = [record.get("sample_id") for record in records]
        sample_coverage = (len(ids) / len(expected)) if expected else 0.0
        if len(ids) != len(set(ids)) or set(ids) != expected:
            sample_coverage = 0.0
        numeric_failure = sum(bool(record.get("numeric_failure")) for record in records)
        resource_failure = sum(bool(record.get("resource_failure")) for record in records)
        validations[label] = compute_evidence_backed_validation(
            method_id, load_audit(method_id), evidence_path, direct_parity_pass,
            sample_coverage, numeric_failure, resource_failure, current_commit)

    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot").mkdir(exist_ok=True)
    (output / "parity").mkdir(exist_ok=True)
    if parity_source_path != parity_out:
        parity_out.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    historical_commit = path_introduction_commit(historical_run)
    source_ref = {
        "schema_version": "0.2.0",
        "historical_result_run": str(historical_run),
        "historical_result_git_commit": historical_commit,
        "historical_summary_ref": str(summary_path),
        "historical_metrics_ref": str(metrics_path),
        "historical_parity_ref": str(historical_parity),
        "current_parity_source_ref": parity_source,
        "validation_git_commit": current_commit,
        "validation_evidence_ref": str(Path(evidence_path).resolve()),
        "locked_config_ref": str(Path(locked_config).resolve()),
        "parity_ref": str(parity_out),
        "result_values_recomputed": False,
    }
    (output / "source_run_ref.json").write_text(
        json.dumps(source_ref, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    port_record = {
        "schema_version": "0.2.0", "validation_git_commit": current_commit,
        "validation_evidence_ref": str(Path(evidence_path).resolve()),
        "locked_config_content": locked_content,
        "direct_parity_pass": direct_parity_pass, "validations": validations,
    }
    (output / "pilot/port_validation.json").write_text(
        json.dumps(port_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_valid = all(item["PORT_VALID"] for item in validations.values())
    validation_summary = {
        **source_ref, "schema_version": "0.2.0",
        "validation_status": "PASS" if all_valid else "BLOCKED",
        "published_ports_valid": all_valid,
        "real_cuda_parity_status": parity.get("status", "PASS" if direct_parity_pass else "FAIL"),
    }
    (output / "validation_summary.json").write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = (
        "# P2 validation-only summary\n\n"
        "- Historical result run: `%s`\n"
        "- Historical result commit: `%s`\n"
        "- Validation commit: `%s`\n"
        "- Result values recomputed: `false`\n"
        "- Validation status: `%s`\n"
        "- Real CUDA parity: `%s`\n" %
        (historical_run, historical_commit, current_commit,
         validation_summary["validation_status"],
         validation_summary["real_cuda_parity_status"]))
    (output / "validation_summary.md").write_text(markdown, encoding="utf-8")
    return validation_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-evidence", required=True)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--parity-artifact")
    args = parser.parse_args()
    try:
        summary = validate_existing_pilot(
            args.historical_run, args.output, args.validation_evidence,
            args.locked_config, args.parity_artifact)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
