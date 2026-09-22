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

LOCKED_SPEC = importlib.util.spec_from_file_location(
    "p2_locked_config", BASELINES / "locked_config.py")
LOCKED = importlib.util.module_from_spec(LOCKED_SPEC)
LOCKED_SPEC.loader.exec_module(LOCKED)

QUEUE_SPEC = importlib.util.spec_from_file_location(
    "p2_confirmatory_queue",
    ROOT / "experiments/p2_calibration_baselines/scripts/confirmatory_queue.py")
QUEUE = importlib.util.module_from_spec(QUEUE_SPEC)
QUEUE_SPEC.loader.exec_module(QUEUE)

PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "p2_confirmatory_preflight",
    ROOT / "experiments/p2_calibration_baselines/scripts/confirmatory_preflight.py")
PREFLIGHT = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT)


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


def test_locked_config_parameters_are_passed_to_method(tmp_path):
    document = json.loads((BASELINES / "locked_port_configs.json").read_text())
    document["methods"]["tent_audio_ep"].update(
        {"lr": 0.123, "steps": 7, "weight_decay": 0.456})
    path = tmp_path / "locked.json"
    path.write_text(json.dumps(document))
    entry = LOCKED.load_locked_method(path, "tent_audio_ep")
    observed = {}

    def spy(*args, **kwargs):
        observed.update(kwargs)
        return "called"

    assert LOCKED.invoke_locked("tent_audio_ep", entry, spy, object()) == "called"
    assert observed == {"lr": 0.123, "steps": 7, "weight_decay": 0.456}


def test_missing_locked_method_fails(tmp_path):
    path = tmp_path / "locked.json"
    path.write_text(json.dumps({"methods": {}}))
    with pytest.raises(ValueError, match="no method"):
        LOCKED.load_locked_method(path, "tent_audio_ep")


def test_missing_validation_evidence_fails_closed(tmp_path):
    result = PV.compute_evidence_backed_validation(
        "tent_audio_ep", {"repo_commit": "abc"}, tmp_path / "missing.json",
        True, 1.0, 0, 0)
    assert result["PORT_VALID"] is False


@pytest.mark.parametrize("field", ["parameter_scope_verified", "reset_contract_pass",
                                    "label_isolation_pass"])
def test_false_contract_evidence_fails_closed(tmp_path, field):
    record = {"method_id": "tent_audio_ep", "parameter_scope_verified": True,
              "reset_contract_pass": True, "label_isolation_pass": True,
              "audit_verified": True, "official_commit_pinned": True,
              "official_repo_commit": "abc"}
    record[field] = False
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"methods": {"tent_audio_ep": record}}))
    result = PV.compute_evidence_backed_validation(
        "tent_audio_ep", {"repo_commit": "abc", "repo_commit_pinned": True,
                          "algorithm_audit_status": "VERIFIED"},
        path, True, 1.0, 0, 0)
    assert result["PORT_VALID"] is False


def test_failed_confirmatory_job_causes_nonzero_final_status(tmp_path, monkeypatch):
    class Failed:
        returncode = 3

    monkeypatch.setattr(QUEUE.subprocess, "run", lambda *args, **kwargs: Failed())
    store = QUEUE.StatusStore(tmp_path / "confirmatory")
    QUEUE.run_job("split", "method", "0", tmp_path, tmp_path / "locked.json", store)
    assert store.records["split/method"]["status"] == "FAILED"
    assert QUEUE.final_exit_code(store.records) != 0


def test_successful_confirmatory_job_can_resume_without_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "confirmatory/split/method"
    output.mkdir(parents=True)
    scores = output / "scores.jsonl"
    scores.write_text("original\n")
    store = QUEUE.StatusStore(tmp_path / "confirmatory")
    store.put("split/method", {"dataset": "split", "method": "method",
                               "status": "SUCCESS", "returncode": 0, "gpu": "0",
                               "resumed_skipped": False, "output": str(output)})

    def must_not_run(*args, **kwargs):
        raise AssertionError("successful job was dispatched again")

    monkeypatch.setattr(QUEUE.subprocess, "run", must_not_run)
    QUEUE.run_job("split", "method", "0", tmp_path, tmp_path / "locked.json", store)
    assert scores.read_text() == "original\n"
    assert store.records["split/method"]["resumed_skipped"] is True


def test_confirmatory_status_store_writes_json_and_csv(tmp_path):
    store = QUEUE.StatusStore(tmp_path)
    store.put("split/method", {"dataset": "split", "method": "method",
                               "status": "SUCCESS", "returncode": 0, "gpu": "0",
                               "resumed_skipped": False, "output": "out"})
    assert json.loads((tmp_path / "job_status.json").read_text())["jobs"][
        "split/method"]["status"] == "SUCCESS"
    assert "SUCCESS" in (tmp_path / "job_status.csv").read_text()


def test_preflight_blocks_invalid_port(tmp_path):
    path = tmp_path / "port_validation.json"
    path.write_text(json.dumps({"validations": {
        "TENT": {"method_id": "tent_audio_ep", "PORT_VALID": False},
        "SAR": {"method_id": "sar_audio_ep", "PORT_VALID": True},
        "MEMO": {"method_id": "memo_audio_ep_full", "PORT_VALID": True}}}))
    with pytest.raises(ValueError, match="not PORT_VALID"):
        PREFLIGHT.validate_port_results(path)


def test_confirmatory_preflight_passes_all_hard_gates(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "pilot").mkdir(parents=True)
    (run_dir / "parity").mkdir()
    validations = {
        method: {"method_id": method, "PORT_VALID": True}
        for method in LOCKED.PUBLISHED_METHODS
    }
    (run_dir / "pilot/port_validation.json").write_text(
        json.dumps({"validations": validations}))
    (run_dir / "parity/waveform_frozen_parity.json").write_text(
        json.dumps({"all_within_project_tolerance": True}))

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.rows = [{"sample_id": "expected"}]

    class Cache:
        def __init__(self, *args, **kwargs):
            pass

        def load_by_id(self):
            return {"expected": object()}

    monkeypatch.setattr(PREFLIGHT, "TargetWaveformDataset", Dataset)
    monkeypatch.setattr(PREFLIGHT, "FeatureCache", Cache)
    monkeypatch.setattr(PREFLIGHT, "get_split", lambda split_id: {
        "manifest_ref": "manifest", "data_roots": {}, "role": "target_test",
        "feature_cache_ref": "cache"})
    monkeypatch.setattr(
        PREFLIGHT, "load_split_sample_ids",
        lambda split_id: (["target10-only"], 0) if split_id == "target10"
        else (["expected"], 0))
    report = PREFLIGHT.run_preflight(
        run_dir, BASELINES / "locked_port_configs.json", gpu_count=4)
    assert report["status"] == "PASS"
    assert report["output_mode"] == "NEW"


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
