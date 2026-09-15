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
                                             "source_val_selection_sha256": sha256_file(selection)},
              "eval_preprocess_hash": "e" * 64, "class_index_map": {"bonafide": 1, "spoof": 0},
              "head_ref": "linear_head.pt", "embedding_dim": d, "training_status": "FINALIZED",
              "training_phase": "full", "task_weight_origin": "trained_in_project",
              "source_val_selection_ref": str(selection), "parity_report_ref": "parity.json"}
    bundle_path = export / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    (export / "linear_head.pt").write_bytes(b"head")
    (export / "detector_state.pt").write_bytes(b"state")
    (export / "parity.json").write_text(json.dumps({"schema_version": "0.1.0", "status": "PASS",
                                                     "module_modes_stable": True,
                                                     "buffers_stable": True}))
    (export / "export_manifest.json").write_text(json.dumps({"schema_version": "0.1.0",
        "status": "LOCKED", "immutable": True, "files": {
            name: sha256_file(export / name) for name in
            ("detector_state.pt", "linear_head.pt", "parity.json", "bundle.json")}}))
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
