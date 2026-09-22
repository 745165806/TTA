#!/usr/bin/env python
"""Unified published-port / control validation contract.

``PORT_VALID`` is computed from the real contract (never hard-coded); NormOnly is
a mechanism control (``CONTROL_VALID``) and Frozen is a reference, not a port.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def compute_port_validation(method_id, audit_status, official_repo_commit,
                            official_commit_pinned, direct_parity_pass,
                            parameter_scope_verified, reset_contract_pass,
                            label_isolation_pass, sample_coverage,
                            numeric_failure_count, resource_failure_count):
    """Return the validation record with ``PORT_VALID`` computed from all conditions."""
    audit_verified = audit_status == "VERIFIED"
    coverage_complete = float(sample_coverage) == 1.0
    port_valid = all([
        audit_verified,
        bool(official_commit_pinned),
        bool(direct_parity_pass),
        bool(parameter_scope_verified),
        bool(reset_contract_pass),
        bool(label_isolation_pass),
        coverage_complete,
        int(numeric_failure_count) == 0,
        int(resource_failure_count) == 0,
    ])
    record = {
        "method_id": method_id,
        "audit_verified": audit_verified,
        "official_repo_commit": official_repo_commit,
        "official_commit_pinned": bool(official_commit_pinned),
        "direct_parity_pass": bool(direct_parity_pass),
        "parameter_scope_verified": bool(parameter_scope_verified),
        "reset_contract_pass": bool(reset_contract_pass),
        "label_isolation_pass": bool(label_isolation_pass),
        "sample_coverage": float(sample_coverage),
        "numeric_failure_count": int(numeric_failure_count),
        "resource_failure_count": int(resource_failure_count),
        "PORT_VALID": port_valid,
    }
    return record


def compute_control_validation(method_id, sample_coverage, numeric_failure_count,
                               resource_failure_count):
    """NormOnly is a mechanism control, not a published port."""
    record = {
        "method_id": method_id,
        "sample_coverage": float(sample_coverage),
        "numeric_failure_count": int(numeric_failure_count),
        "resource_failure_count": int(resource_failure_count),
        "CONTROL_VALID": (float(sample_coverage) == 1.0
                          and int(numeric_failure_count) == 0
                          and int(resource_failure_count) == 0),
        "PORT_VALID": False,
    }
    return record


def compute_coverages(records, expected_ids):
    """Distinct sample coverage vs adaptation coverage.

    ``sample_coverage`` = fraction of expected IDs with a record; SAR/TENT can
    have sample_coverage == 1 while adaptation_coverage < 1 (reliability
    abstention), and NormOnly has adaptation_coverage == 0.
    """
    expected = list(expected_ids)
    if not expected:
        raise ValueError("expected_ids must be nonempty")
    n_expected = len(expected)
    n_records = len(records)
    sample_coverage = n_records / n_expected
    adaptation_coverage = (
        sum(bool(r.get("adaptation_applied")) for r in records) / n_records
        if n_records else 0.0)
    return sample_coverage, adaptation_coverage


def load_audit(method_id):
    """Load the machine-readable audit for a published method."""
    mapping = {"tent_audio_ep": "tent_audit.json", "sar_audio_ep": "sar_audit.json",
               "memo_audio_ep_full": "memo_audit.json"}
    if method_id not in mapping:
        raise ValueError("no audit registered for %s" % method_id)
    path = ROOT / "experiments/p2_calibration_baselines/audits" / mapping[method_id]
    return json.loads(path.read_text(encoding="utf-8"))
