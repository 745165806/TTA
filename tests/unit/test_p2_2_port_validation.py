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

GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "p2_generate_validation_evidence",
    ROOT / "experiments/p2_calibration_baselines/scripts/generate_validation_evidence.py")
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(GENERATOR)

VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "p2_validate_existing_pilot",
    ROOT / "experiments/p2_calibration_baselines/scripts/validate_existing_pilot.py")
VALIDATE = importlib.util.module_from_spec(VALIDATE_SPEC)
VALIDATE_SPEC.loader.exec_module(VALIDATE)


def _evidence(path, commit, **overrides):
    record = {"method_id": "tent_audio_ep", "parameter_scope_verified": True,
              "reset_contract_pass": True, "label_isolation_pass": True,
              "audit_verified": True, "official_commit_pinned": True,
              "official_repo_commit": "abc"}
    record.update(overrides)
    path.write_text(json.dumps({
        "schema_version": "0.2.0", "validated_git_commit": commit,
        "validated_tree_or_code_version": "clean_git_tree_at_exact_commit",
        "generated_at": "2026-09-22T00:00:00+00:00",
        "methods": {"tent_audio_ep": record}}))
    return path


def _run_config(path, commit, locked_content, method="method", split="split"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "0.2.0", "git_commit": commit,
                                "method": method, "split": split,
                                "locked_config_content": locked_content,
                                "parameters": {}}))


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
        True, 1.0, 0, 0, "a" * 40)
    assert result["PORT_VALID"] is False


@pytest.mark.parametrize("field", ["parameter_scope_verified", "reset_contract_pass",
                                    "label_isolation_pass"])
def test_false_contract_evidence_fails_closed(tmp_path, field):
    path = tmp_path / "evidence.json"
    _evidence(path, "a" * 40, **{field: False})
    result = PV.compute_evidence_backed_validation(
        "tent_audio_ep", {"repo_commit": "abc", "repo_commit_pinned": True,
                          "algorithm_audit_status": "VERIFIED"},
        path, True, 1.0, 0, 0, "a" * 40)
    assert result["PORT_VALID"] is False


def test_validation_evidence_requires_exact_git_commit(tmp_path):
    path = _evidence(tmp_path / "evidence.json", "a" * 40)
    audit = {"repo_commit": "abc", "repo_commit_pinned": True,
             "algorithm_audit_status": "VERIFIED"}
    valid = PV.compute_evidence_backed_validation(
        "tent_audio_ep", audit, path, True, 1.0, 0, 0, "a" * 40)
    stale = PV.compute_evidence_backed_validation(
        "tent_audio_ep", audit, path, True, 1.0, 0, 0, "b" * 40)
    assert valid["PORT_VALID"] is True
    assert stale["PORT_VALID"] is False


def test_validation_only_writes_new_run_without_recomputing_results(tmp_path, monkeypatch):
    commit = "a" * 40
    historical = tmp_path / "historical"
    (historical / "pilot").mkdir(parents=True)
    (historical / "parity").mkdir()
    (historical / "pilot/p2_1_pilot_summary.json").write_text("{}")
    (historical / "pilot/p2_1_pilot_metrics.csv").write_text("method,status\n")
    (historical / "parity/waveform_frozen_parity.json").write_text(
        json.dumps({"all_within_project_tolerance": True}))
    for directory in ("tent", "sar", "memo"):
        path = historical / "pilot" / directory
        path.mkdir()
        (path / "scores.jsonl").write_text(json.dumps({
            "sample_id": "sample", "numeric_failure": False,
            "resource_failure": False}) + "\n")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(GENERATOR.build_evidence(
        commit, {"pytest_selected_status": "PASS", "test_files": []})))
    parity = tmp_path / "current_parity.json"
    parity.write_text(json.dumps({"git_commit": commit,
                                  "all_within_project_tolerance": True}))
    monkeypatch.setattr(VALIDATE, "load_split_sample_ids",
                        lambda split: (["sample"], 0))
    monkeypatch.setattr(VALIDATE, "path_introduction_commit", lambda path: "7" * 40)
    output = tmp_path / "validation"
    summary = VALIDATE.validate_existing_pilot(
        historical, output, evidence_path, BASELINES / "locked_port_configs.json",
        parity, current_commit=commit)
    assert summary["result_values_recomputed"] is False
    assert summary["validation_status"] == "PASS"
    assert (output / "pilot/port_validation.json").is_file()
    assert not (historical / "pilot/port_validation.json").exists()


def test_failed_confirmatory_job_causes_nonzero_final_status(tmp_path, monkeypatch):
    class Failed:
        returncode = 3

    monkeypatch.setattr(QUEUE.subprocess, "run", lambda *args, **kwargs: Failed())
    store = QUEUE.StatusStore(tmp_path)
    QUEUE.run_job("split", "method", "0", tmp_path, tmp_path / "locked.json",
                  store, "a" * 40, {"methods": {}})
    assert store.records["split/method"]["status"] == "FAILED"
    assert QUEUE.final_exit_code(store.records) != 0


def test_successful_confirmatory_job_can_resume_without_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "split/method"
    output.mkdir(parents=True)
    scores = output / "scores.jsonl"
    scores.write_text("original\n")
    current_commit = "a" * 40
    locked_content = {"methods": {"method": {"lr": 1}}}
    _run_config(output / "run_config.json", current_commit, locked_content)
    store = QUEUE.StatusStore(tmp_path)
    store.put("split/method", {"dataset": "split", "method": "method",
                               "status": "SUCCESS", "returncode": 0, "gpu": "0",
                               "resumed_skipped": False, "output": str(output),
                               "git_commit": current_commit,
                               "locked_config_content": locked_content,
                               "run_config_ref": str(output / "run_config.json")})

    def must_not_run(*args, **kwargs):
        raise AssertionError("successful job was dispatched again")

    monkeypatch.setattr(QUEUE.subprocess, "run", must_not_run)
    QUEUE.run_job("split", "method", "0", tmp_path, tmp_path / "locked.json",
                  store, current_commit, locked_content)
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


@pytest.mark.parametrize("change", ["config", "commit"])
def test_resume_changed_provenance_fails_closed(tmp_path, monkeypatch, change):
    output = tmp_path / "split/method"
    scores = output / "scores.jsonl"
    output.mkdir(parents=True)
    scores.write_text("original\n")
    old_commit, new_commit = "a" * 40, "b" * 40
    old_config, new_config = {"value": 1}, {"value": 2}
    _run_config(output / "run_config.json", old_commit, old_config)
    store = QUEUE.StatusStore(tmp_path)
    store.put("split/method", {
        "dataset": "split", "method": "method", "status": "SUCCESS",
        "returncode": 0, "gpu": "0", "resumed_skipped": False,
        "output": str(output), "git_commit": old_commit,
        "locked_config_content": old_config,
        "run_config_ref": str(output / "run_config.json")})
    monkeypatch.setattr(QUEUE.subprocess, "run", lambda *a, **k: pytest.fail(
        "mismatched output must not run"))
    current_commit = new_commit if change == "commit" else old_commit
    current_config = new_config if change == "config" else old_config
    QUEUE.run_job("split", "method", "0", tmp_path, "locked", store,
                  current_commit, current_config)
    assert store.records["split/method"]["status"] == "CONFIG_MISMATCH"
    assert scores.read_text() == "original\n"


def test_existing_scores_without_matching_run_config_fails(tmp_path, monkeypatch):
    output = tmp_path / "split/method"
    output.mkdir(parents=True)
    (output / "scores.jsonl").write_text("original\n")
    store = QUEUE.StatusStore(tmp_path)
    monkeypatch.setattr(QUEUE.subprocess, "run", lambda *a, **k: pytest.fail(
        "existing output must not run"))
    QUEUE.run_job("split", "method", "0", tmp_path, "locked", store,
                  "a" * 40, {"value": 1})
    assert store.records["split/method"]["status"] == "CONFIG_MISMATCH"


def test_locked_config_binding_is_content_based(tmp_path):
    content = (BASELINES / "locked_port_configs.json").read_text()
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(content)
    second.write_text(content)
    assert LOCKED.load_locked_document(first) == LOCKED.load_locked_document(second)
    changed = json.loads(content)
    changed["methods"]["tent_audio_ep"]["lr"] = 0.5
    second.write_text(json.dumps(changed))
    assert LOCKED.load_locked_document(first) != LOCKED.load_locked_document(second)


def test_preflight_blocks_invalid_port(tmp_path):
    path = tmp_path / "port_validation.json"
    path.write_text(json.dumps({"validations": {
        "TENT": {"method_id": "tent_audio_ep", "PORT_VALID": False},
        "SAR": {"method_id": "sar_audio_ep", "PORT_VALID": True},
        "MEMO": {"method_id": "memo_audio_ep_full", "PORT_VALID": True}}}))
    with pytest.raises(ValueError, match="not PORT_VALID"):
        PREFLIGHT.validate_port_results(path)


def _validation_run(path, commit, locked_content):
    (path / "pilot").mkdir(parents=True)
    (path / "parity").mkdir()
    validations = {
        method: {"method_id": method, "PORT_VALID": True}
        for method in LOCKED.PUBLISHED_METHODS
    }
    evidence = path / "validation_evidence.json"
    evidence.write_text(json.dumps(GENERATOR.build_evidence(
        commit, {"pytest_selected_status": "PASS", "test_files": []})))
    (path / "validation_summary.json").write_text(json.dumps({
        "validation_git_commit": commit, "validation_status": "PASS",
        "published_ports_valid": True}))
    (path / "pilot/port_validation.json").write_text(json.dumps({
        "validation_git_commit": commit, "locked_config_content": locked_content,
        "validation_evidence_ref": str(evidence), "validations": validations}))
    (path / "parity/waveform_frozen_parity.json").write_text(json.dumps({
        "git_commit": commit, "all_within_project_tolerance": True}))


def test_validation_run_commit_mismatch_blocks_confirmatory(tmp_path):
    locked = BASELINES / "locked_port_configs.json"
    validation_run = tmp_path / "validation"
    _validation_run(validation_run, "a" * 40, LOCKED.load_locked_document(locked))
    with pytest.raises(ValueError, match="current git commit"):
        PREFLIGHT.validate_validation_run(validation_run, locked, "b" * 40)


def test_validation_run_current_commit_passes(tmp_path):
    locked = BASELINES / "locked_port_configs.json"
    validation_run = tmp_path / "validation"
    _validation_run(validation_run, "a" * 40, LOCKED.load_locked_document(locked))
    result = PREFLIGHT.validate_validation_run(validation_run, locked, "a" * 40)
    assert result["validation_status"] == "PASS"


def test_confirmatory_preflight_passes_all_hard_gates(tmp_path, monkeypatch):
    run_dir = tmp_path / "confirmatory"
    validation_run = tmp_path / "validation"
    current_commit = "a" * 40
    locked = BASELINES / "locked_port_configs.json"
    _validation_run(validation_run, current_commit,
                    LOCKED.load_locked_document(locked))

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
        run_dir, validation_run, locked, gpu_count=4,
        current_commit=current_commit)
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
