import json

import pytest

from eptta.data.io import sha256_file
from eptta.errors import ContractError, DataError
from eptta.execution.extract import compile_inference_job
from eptta.offline.artifacts import build_source_resources


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def extraction_plan(tmp_path, with_label=False, purpose="confirmatory", input_role="target_test",
                    row_role=None):
    export = tmp_path / "export"
    export.mkdir()
    head = export / "linear_head.pt"
    head.write_bytes(b"head")
    state = export / "detector_state.pt"
    state.write_bytes(b"state")
    parity = export / "parity.json"
    parity.write_text(json.dumps({"schema_version": "0.1.0", "status": "PASS",
                                  "module_modes_stable": True, "buffers_stable": True}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "FINALIZED", "training_phase": "full",
                                     "task_weight_origin": "trained_in_project",
                                     "selected_checkpoint_sha256": "a" * 64,
                                     "training_run_id": "run", "fit_snapshot_hash": "c" * 64,
                                     "source_val_snapshot_hash": "d" * 64,
                                     "recipe_hash": "e" * 64}))
    bundle = {"schema_version": "0.1.0", "model_id": "aasist_source", "baseline_id": "baseline",
              "selected_checkpoint_sha256": "a" * 64, "training_run_id": "run",
              "fit_snapshot_hash": "c" * 64, "source_val_snapshot_hash": "d" * 64,
              "recipe_hash": "e" * 64, "init_provenance": {"scope": "native_initialization"},
              "task_training_provenance": {"task_weight_origin": "trained_in_project",
                                             "source_val_selection_sha256": sha256_file(selection),
                                             "migration": {"exact_resume_claim": False},
                                             "training_endpoint": {"last_epoch": 79,
                                               "completed_epoch_count": 80, "scheduler_horizon_epochs": 100}},
              "eval_preprocess_hash": "b" * 64, "class_index_map": {"bonafide": 1, "spoof": 0},
              "head_ref": "linear_head.pt", "embedding_dim": 160,
              "training_status": "FINALIZED", "training_phase": "full",
              "task_weight_origin": "trained_in_project", "source_val_selection_ref": str(selection),
              "parity_report_ref": "parity.json",
              "model_contract": {"embedding_point": "native_out_layer_input", "freq_aug": False},
              "score_contract": {"formula": "native_logits[spoof]-native_logits[bonafide]",
                 "direction": "larger_is_spoof", "output_type": "logit_difference", "unit": "dimensionless"},
              "numerical_contract": {"atol": 1e-6, "rtol": 1e-5},
              "r4_validation": {"status": "PASS", "fit_count": 128, "source_val_count": 5654},
              "export_code_sha256": "f" * 64}
    for name in ("parity_per_sample.jsonl", "source_val_recompute.json", "fit128_uids.json",
                 "baseline_bridge.py", "author_training.py"):
        (export / name).write_bytes(name.encode())
    (export / "source_val_recompute.json").write_text(json.dumps({
        "schema_version": "0.1.0", "status": "PASS", "source_val_count": 5654,
        "reference_vs_export_eer_abs": 0.0, "historical_vs_reference_abs": 0.0,
        "eer_atol": 1e-12}))
    (export / "fit128_uids.json").write_text(json.dumps({"count": 128,
        "class_counts": {"bonafide": 64, "spoof": 64},
        "attack_ids": ["A01", "A02", "A03", "A04", "A05", "A06"]}))
    bundle["r4_validation"].update({
        "fit_uids_sha256": sha256_file(export / "fit128_uids.json"),
        "parity_sha256": sha256_file(parity),
        "per_sample_sha256": sha256_file(export / "parity_per_sample.jsonl"),
        "source_val_recompute_sha256": sha256_file(export / "source_val_recompute.json")})
    (export / "bundle.json").write_text(json.dumps(bundle))
    manifest = tmp_path / "manifest.jsonl"
    row = {"schema_version": "0.1.0", "sample_id": "sample", "root_key": "data",
           "audio_relpath": "sample.wav", "input_sha256": "c" * 64,
           "split_role": row_role or input_role}
    if with_label:
        row["canonical_label"] = 1
    write_jsonl(manifest, [row])
    export_manifest = {"schema_version": "0.1.0", "status": "LOCKED", "immutable": True,
                       "r5_eligible": True,
                       "files": {name: sha256_file(export / name) for name in
                                 ("bundle.json", "linear_head.pt", "detector_state.pt", "parity.json",
                                  "parity_per_sample.jsonl", "source_val_recompute.json", "fit128_uids.json",
                                  "baseline_bridge.py", "author_training.py")}}
    (export / "export_manifest.json").write_text(json.dumps(export_manifest))
    plan = {"schema_version": "0.1.0", "status": "LOCKED",
            "purpose": purpose, "input_role": input_role,
            "frozen_bundle_ref": str(export / "bundle.json"), "manifest_ref": str(manifest),
            "manifest_sha256": sha256_file(manifest), "data_roots": {"data": str(tmp_path)},
            "probe": {"num_views": 3, "seed": 13, "noise_snr_db": 30.0, "fir_side_gain": .05},
            "numerical_mode": {"dtype": "float32", "tf32": False, "block_units": 4},
            "output_root": str(tmp_path / "cache")}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path


def test_inference_job_is_label_free_and_deterministically_sharded(tmp_path):
    job = compile_inference_job(extraction_plan(tmp_path), 0, 1)
    assert job["job_type"] == "inference" and job["expected_ids"] == ["sample"]
    assert job["cache_identity"]["wrapper_numerical_version"] == "baseline_bridge_sha256:" + job["worker_sha256"]
    assert "canonical_label" not in json.dumps(job)


def test_inference_job_rejects_target_label_leak(tmp_path):
    with pytest.raises(DataError, match="labels"):
        compile_inference_job(extraction_plan(tmp_path, with_label=True), 0, 1)


def test_r5_inference_rejects_failed_frozen_parity(tmp_path):
    plan = extraction_plan(tmp_path)
    export = tmp_path / "export"
    parity = export / "parity.json"
    parity.write_text(json.dumps({"schema_version": "0.1.0", "status": "FAIL",
                                  "module_modes_stable": True, "buffers_stable": True}))
    manifest = json.loads((export / "export_manifest.json").read_text())
    manifest["files"]["parity.json"] = sha256_file(parity)
    (export / "export_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="parity PASS"):
        compile_inference_job(plan, 0, 1)


def test_r5_source_prepare_allows_only_explicit_fit_or_cal0_role(tmp_path):
    job = compile_inference_job(extraction_plan(tmp_path, purpose="source_prepare", input_role="fit"), 0, 1)
    assert job["purpose"] == "source_prepare" and job["input_role"] == "fit"


def test_inference_rejects_manifest_role_different_from_locked_role(tmp_path):
    with pytest.raises(DataError, match="role differs"):
        compile_inference_job(extraction_plan(tmp_path, purpose="select", input_role="select",
                                              row_role="target_test"), 0, 1)


def test_r5_source_artifacts_reject_non_fit_or_non_cal0_labels(tmp_path):
    plan = {"schema_version": "0.1.0", "status": "LOCKED", "frozen_bundle_ref": "unused",
            "fit_role": "select", "fit_manifest_sha256": "a" * 64, "fit_labels_ref": "unused",
            "fit_labels_sha256": "a" * 64, "fit_groups_ref": "unused",
            "fit_groups_sha256": "a" * 64, "calibration_role": "cal0",
            "calibration_manifest_sha256": "b" * 64, "calibration_cache_ref": "unused",
            "calibration_labels_ref": "unused", "calibration_labels_sha256": "b" * 64,
            "source_snapshot_hash": "c" * 64, "rank": 2,
            "alpha_cal": .05, "anchor_per_class": 2, "seed": 13, "random_seeds": [13, 29, 47],
            "treatment_families": ["noise", "fir"], "samples_per_group": 2, "pair_seed": 13,
            "margin_bins": 2, "margin_epsilon": 1e-6, "minimum_cal0_bonafide": 2,
            "fixed_adapter": {"steps": 1, "lr": .01, "rho": .2, "gamma": .1,
                              "lambda_keep": 1.0}}
    path = tmp_path / "source-artifacts.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ContractError, match="only from fit and cal0"):
        build_source_resources(path, "unused", tmp_path / "output")


def test_r5_source_artifacts_reject_changed_label_or_group_sidecar(tmp_path):
    fit_labels = tmp_path / "fit.labels.jsonl"
    fit_groups = tmp_path / "fit.groups.jsonl"
    cal_labels = tmp_path / "cal.labels.jsonl"
    write_jsonl(fit_labels, [{"schema_version": "0.1.0", "sample_id": "fit",
                              "canonical_label": 0}])
    write_jsonl(fit_groups, [{"schema_version": "0.1.0", "sample_id": "fit",
                              "source_group_id": "speaker"}])
    write_jsonl(cal_labels, [{"schema_version": "0.1.0", "sample_id": "cal",
                              "canonical_label": 0}])
    plan = {"schema_version": "0.1.0", "status": "LOCKED", "frozen_bundle_ref": "unused",
            "fit_role": "fit", "fit_manifest_sha256": "a" * 64,
            "fit_labels_ref": str(fit_labels), "fit_labels_sha256": sha256_file(fit_labels),
            "fit_groups_ref": str(fit_groups), "fit_groups_sha256": sha256_file(fit_groups),
            "calibration_role": "cal0", "calibration_manifest_sha256": "b" * 64,
            "calibration_cache_ref": "unused", "calibration_labels_ref": str(cal_labels),
            "calibration_labels_sha256": sha256_file(cal_labels),
            "source_snapshot_hash": "c" * 64, "rank": 2, "alpha_cal": .05,
            "anchor_per_class": 2, "seed": 13, "random_seeds": [13, 29, 47],
            "treatment_families": ["noise", "fir"], "samples_per_group": 2,
            "pair_seed": 13, "margin_bins": 2, "margin_epsilon": 1e-6,
            "minimum_cal0_bonafide": 2,
            "fixed_adapter": {"steps": 1, "lr": .01, "rho": .2, "gamma": .1,
                              "lambda_keep": 1.0}}
    plan_path = tmp_path / "source-artifacts.json"
    plan_path.write_text(json.dumps(plan))
    fit_labels.write_text(fit_labels.read_text().replace('"canonical_label": 0',
                                                         '"canonical_label": 1'))
    with pytest.raises(DataError, match="sidecar changed"):
        build_source_resources(plan_path, "unused", tmp_path / "output")
