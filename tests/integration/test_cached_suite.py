import json

import numpy as np

from eptta.cache import CacheIdentity, FeatureCacheWriter
from eptta.data.io import sha256_file
from eptta.evaluation.seal import seal_scores
from eptta.execution.suite import run_suite
from eptta.offline.artifacts import write_frozen_resources


def test_cache_to_mechanism_scores_to_seal(tmp_path):
    rng = np.random.default_rng(3)
    d, r = 6, 2
    U, _ = np.linalg.qr(rng.normal(size=(d, r)))
    w = rng.normal(size=d)
    labels = np.arange(8) % 2
    anchors = rng.normal(size=(8, d))
    b, tau = .1, -.2
    signs = 2 * labels - 1
    desired = tau + signs * np.linspace(.2, 1.0, 8)
    anchors += ((desired - anchors @ w - b) / (w @ w))[:, None] * w
    scores = anchors @ w + b
    margins = signs * (scores - tau)
    checkpoint = "a" * 64
    export = tmp_path / "export"
    export.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "FINALIZED", "training_phase": "full",
                                     "task_weight_origin": "trained_in_project",
                                     "selected_checkpoint_sha256": checkpoint,
                                     "training_run_id": "run", "fit_snapshot_hash": "b" * 64,
                                     "source_val_snapshot_hash": "c" * 64,
                                     "recipe_hash": "d" * 64}))
    bundle = {"schema_version": "0.1.0", "model_id": "aasist_source", "baseline_id": "baseline",
              "selected_checkpoint_sha256": checkpoint, "training_run_id": "run",
              "fit_snapshot_hash": "b" * 64, "source_val_snapshot_hash": "c" * 64,
              "recipe_hash": "d" * 64, "init_provenance": {"scope": "native_initialization"},
              "task_training_provenance": {"task_weight_origin": "trained_in_project",
                                             "source_val_selection_sha256": sha256_file(selection),
                                             "migration": {"exact_resume_claim": False},
                                             "training_endpoint": {"last_epoch": 79,
                                               "completed_epoch_count": 80, "scheduler_horizon_epochs": 100}},
              "eval_preprocess_hash": "e" * 64, "class_index_map": {"bonafide": 1, "spoof": 0},
              "head_ref": "linear_head.pt", "embedding_dim": d, "training_status": "FINALIZED",
              "training_phase": "full", "task_weight_origin": "trained_in_project",
              "source_val_selection_ref": str(selection), "parity_report_ref": "parity.json",
              "model_contract": {"embedding_point": "native_out_layer_input", "freq_aug": False},
              "score_contract": {"formula": "native_logits[spoof]-native_logits[bonafide]",
                 "direction": "larger_is_spoof", "output_type": "logit_difference", "unit": "dimensionless"},
              "numerical_contract": {"atol": 1e-6, "rtol": 1e-5},
              "r4_validation": {"status": "PASS", "fit_count": 128, "source_val_count": 5654},
              "export_code_sha256": "f" * 64}
    bundle_path = export / "bundle.json"
    (export / "linear_head.pt").write_bytes(b"head")
    (export / "detector_state.pt").write_bytes(b"state")
    (export / "parity.json").write_text(json.dumps({"schema_version": "0.1.0", "status": "PASS",
                                                     "module_modes_stable": True,
                                                     "buffers_stable": True}))
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
        "parity_sha256": sha256_file(export / "parity.json"),
        "per_sample_sha256": sha256_file(export / "parity_per_sample.jsonl"),
        "source_val_recompute_sha256": sha256_file(export / "source_val_recompute.json")})
    bundle_path.write_text(json.dumps(bundle))
    (export / "export_manifest.json").write_text(json.dumps({"schema_version": "0.1.0",
        "status": "LOCKED", "immutable": True, "r5_eligible": True, "files": {
            name: sha256_file(export / name) for name in
            ("detector_state.pt", "linear_head.pt", "parity.json", "bundle.json",
             "parity_per_sample.jsonl", "source_val_recompute.json", "fit128_uids.json",
             "baseline_bridge.py", "author_training.py")}}))
    resources_dir = tmp_path / "resources"
    write_frozen_resources(resources_dir, "baseline", checkpoint,
        {"U": U.astype("float32"), "w": w.astype("float32"),
         "anchors_z": anchors.astype("float32"), "anchors_y": labels.astype("int64"),
         "anchors_m0": margins.astype("float32"), "anchors_s0": scores.astype("float32")},
        {"b": b, "tau0": tau}, "b" * 64)
    identity = CacheIdentity("c" * 64, "baseline", checkpoint, "d" * 64,
                             "wrapper-v1", "e" * 64, "f" * 64, 13, "float32", {"dtype": "float32"})
    cache_dir = tmp_path / "cache"
    with FeatureCacheWriter(cache_dir, identity, ["x", "y"], 3, d) as writer:
        writer.add(["x", "y"], rng.normal(size=(2, 3, d)).astype("float32"))
    plan = {"schema_version": "0.1.0", "status": "LOCKED", "suite_id": "fixture",
            "feature_cache_ref": str(cache_dir), "resources_ref": str(resources_dir),
            "frozen_bundle_ref": str(bundle_path),
            "methods": [{"method_id": "frozen", "config": {"steps": 0, "lr": .01,
                         "rho": .2, "gamma": .1, "lambda_keep": 1.0}, "params": {}},
                        {"method_id": "ep_tta", "config": {"steps": 2, "lr": .01,
                         "rho": .2, "gamma": .1, "lambda_keep": 1.0}, "params": {}},
                        {"method_id": "tent_audio_ep", "config": {"steps": 1, "lr": .01,
                         "rho": .2, "gamma": .1, "lambda_keep": 0.0}, "params": {}}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = run_suite(plan_path, "fixture", "select", tmp_path / "run")
    assert result["target_labels_read"] is False and len(result["methods"]) == 3
    assert result["status"] == "PARTIAL"
    assert json.loads((tmp_path / "run" / "tent_audio_ep" / "run.json").read_text())["status"] == "NOT_RUN"
    for method in ("frozen", "ep_tta"):
        seal = seal_scores(tmp_path / "run" / method)
        assert seal["sample_count"] == 2 and seal["labels_read"] is False
        assert seal["coverage_report_ref"] == "coverage.json"
        assert seal["runtime_report_ref"] == "runtime.jsonl"
        assert seal["runtime_report_sha256"] == sha256_file(tmp_path / "run" / method / "runtime.jsonl")
    frozen_spec = {"schema_version": "0.1.0", "status": "LOCKED", "suite_id": "fixture",
                   "phase": "confirmatory", "selected_on_role": "select",
                   "plan_ref": str(plan_path.resolve()), "plan_sha256": sha256_file(plan_path),
                   "methods": plan["methods"]}
    frozen_spec_path = tmp_path / "confirmatory.lock.json"
    frozen_spec_path.write_text(json.dumps(frozen_spec))
    confirmatory = run_suite(plan_path, "fixture", "confirmatory", tmp_path / "confirmatory",
                             frozen_spec_path)
    assert confirmatory["status"] == "PARTIAL"
