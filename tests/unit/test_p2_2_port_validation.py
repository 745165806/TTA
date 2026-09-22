"""Unit tests for P2.2 port-validation hardening."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "experiments/p2_calibration_baselines/baselines"

SPEC = importlib.util.spec_from_file_location("p2_port_validation",
                                              BASELINES / "port_validation.py")
PV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PV)

SPLIT_SPEC = importlib.util.spec_from_file_location("p2_splits", BASELINES / "splits.py")
SPLITS = importlib.util.module_from_spec(SPLIT_SPEC)
SPLIT_SPEC.loader.exec_module(SPLITS)


def _valid_inputs(**overrides):
    base = dict(method_id="tent_audio_ep", audit_status="VERIFIED",
                official_repo_commit="abc", official_commit_pinned=True,
                direct_parity_pass=True, parameter_scope_verified=True,
                reset_contract_pass=True, label_isolation_pass=True,
                sample_coverage=1.0, numeric_failure_count=0, resource_failure_count=0)
    base.update(overrides)
    return base


def test_port_valid_true_only_when_all_conditions_hold():
    assert PV.compute_port_validation(**_valid_inputs())["PORT_VALID"] is True


def test_port_valid_false_when_audit_fails():
    assert PV.compute_port_validation(**_valid_inputs(audit_status="NOT_VERIFIED"))["PORT_VALID"] is False
    assert PV.compute_port_validation(**_valid_inputs(official_commit_pinned=False))["PORT_VALID"] is False


def test_port_valid_false_when_parity_fails():
    assert PV.compute_port_validation(**_valid_inputs(direct_parity_pass=False))["PORT_VALID"] is False


def test_port_valid_false_when_coverage_incomplete():
    assert PV.compute_port_validation(**_valid_inputs(sample_coverage=0.99))["PORT_VALID"] is False


def test_port_valid_false_on_numeric_failure():
    assert PV.compute_port_validation(**_valid_inputs(numeric_failure_count=1))["PORT_VALID"] is False
    assert PV.compute_port_validation(**_valid_inputs(resource_failure_count=1))["PORT_VALID"] is False


def test_sample_coverage_and_adaptation_coverage_are_distinct():
    expected = ["a", "b", "c", "d"]
    records = [
        {"sample_id": "a", "adaptation_applied": True},
        {"sample_id": "b", "adaptation_applied": False},
        {"sample_id": "c", "adaptation_applied": False},
        {"sample_id": "d", "adaptation_applied": False},
    ]
    sample_cov, adaptation_cov = PV.compute_coverages(records, expected)
    assert sample_cov == 1.0
    assert adaptation_cov == 0.25


def test_itw_target90_excludes_target10():
    ids, excluded = SPLITS.load_split_sample_ids("itw_target90")
    target10_ids, _ = SPLITS.load_split_sample_ids("target10")
    assert excluded > 0
    assert set(ids).isdisjoint(set(target10_ids))


def test_unknown_split_rejected():
    with pytest.raises(SystemExit):
        SPLITS.get_split("banana")


def test_confirmatory_config_is_locked():
    cfg = json.loads((BASELINES / "locked_port_configs.json").read_text(encoding="utf-8"))
    assert cfg["methods"]["tent_audio_ep"]["optimizer"] == "Adam"
    assert cfg["methods"]["tent_audio_ep"]["lr"] == 0.001
    assert cfg["methods"]["sar_audio_ep"]["lr"] == 1.5625e-05
    assert cfg["methods"]["memo_audio_ep_full"]["lr"] == 0.00025
    assert cfg["methods"]["norm_only_audio"]["control"] is True
    for m in ("tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full"):
        assert cfg["methods"][m]["official_repo_commit"]


def test_worker_cannot_read_label_fields(tmp_path):
    # The production worker uses TargetWaveformDataset which must reject labels.
    spec = importlib.util.spec_from_file_location("p2_target_waveform",
                                                  BASELINES / "target_waveform.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    root = tmp_path / "audio"
    root.mkdir()
    (root / "a.wav").write_bytes(b"x")
    for key in ("label", "canonical_label", "attack", "class", "target", "y"):
        bad = tmp_path / ("bad_%s.jsonl" % key)
        bad.write_text(json.dumps({"schema_version": "0.1.0", "sample_id": "a.wav",
                                   "root_key": "root", "audio_relpath": "a.wav",
                                   "split_role": "target_test", key: 1}) + "\n")
        with pytest.raises(ValueError):
            mod.TargetWaveformDataset(str(bad), {"root": str(root)}, role="target_test")
