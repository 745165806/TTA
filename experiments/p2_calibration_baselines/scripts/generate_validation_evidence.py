#!/usr/bin/env python
"""Generate commit-bound P2 port evidence after required tests pass."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR))

from baselines.locked_config import PUBLISHED_METHODS
from baselines.port_validation import load_audit
from baselines.provenance import git_commit, require_clean_tree

TEST_FILES = (
    "tests/unit/test_p2_2_port_validation.py",
    "tests/unit/test_p2_1_baseline_semantics.py",
    "tests/unit/test_p2_waveform_ports.py",
)


def validate_audit(method_id):
    audit = load_audit(method_id)
    if audit.get("method_id") != method_id:
        raise ValueError("audit method mismatch: %s" % method_id)
    if audit.get("algorithm_audit_status") != "VERIFIED":
        raise ValueError("algorithm audit is not VERIFIED: %s" % method_id)
    commit = audit.get("repo_commit")
    if audit.get("repo_commit_pinned") is not True or not isinstance(commit, str) \
            or len(commit) != 40:
        raise ValueError("official commit is not pinned: %s" % method_id)
    return audit


def run_required_tests():
    command = [sys.executable, "-m", "pytest", "-q", *TEST_FILES]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               check=False)
    if completed.returncode != 0:
        raise ValueError("required validation tests failed:\n%s\n%s" %
                         (completed.stdout, completed.stderr))
    return {"pytest_selected_status": "PASS", "test_files": list(TEST_FILES),
            "pytest_output": completed.stdout.strip()}


def build_evidence(current_commit, test_evidence):
    methods = {}
    for method_id in PUBLISHED_METHODS:
        audit = validate_audit(method_id)
        methods[method_id] = {
            "method_id": method_id,
            "parameter_scope_verified": True,
            "reset_contract_pass": True,
            "label_isolation_pass": True,
            "audit_verified": True,
            "official_commit_pinned": True,
            "official_repo_commit": audit["repo_commit"],
            "evidence_refs": [
                "tests/unit/test_p2_waveform_ports.py",
                "tests/unit/test_p2_1_baseline_semantics.py",
                "tests/unit/test_p2_2_port_validation.py",
                "audits/%s_audit.json" % method_id.split("_", 1)[0],
            ],
            "test_evidence": test_evidence,
        }
    return {
        "schema_version": "0.2.0",
        "validated_git_commit": current_commit,
        "validated_tree_or_code_version": "clean_git_tree_at_exact_commit",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "methods": methods,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite validation evidence: %s" % output)
    try:
        require_clean_tree()
        evidence = build_evidence(git_commit(), run_required_tests())
    except ValueError as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
