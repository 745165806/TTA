import json

import numpy as np

from eptta.cache import CacheIdentity, FeatureCacheWriter
from eptta.execution.suite import run_suite
from eptta.offline.artifacts import write_frozen_resources


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cache_to_label_free_ep_suite(tmp_path):
    rng = np.random.default_rng(3)
    d, rank = 6, 2
    U, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    w = rng.normal(size=d)
    labels = np.arange(8) % 2
    anchors = rng.normal(size=(8, d))
    bias, threshold = .1, -.2
    signs = 2 * labels - 1
    desired = threshold + signs * np.linspace(.2, 1.0, 8)
    anchors += ((desired - anchors @ w - bias) / (w @ w))[:, None] * w
    scores = anchors @ w + bias
    margins = signs * (scores - threshold)

    checkpoint = tmp_path / "source" / "checkpoints" / "epoch_0001.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint reference")
    selection = tmp_path / "selection.json"
    _json(selection, {"status": "SELECTED", "selected_epoch": 1})
    export = tmp_path / "export"
    export.mkdir()
    bundle = {"schema_version": "0.3.0", "model_id": "aasist_source",
              "baseline_id": "source-run:epoch_0001", "source_run_id": "source-run",
              "checkpoint_ref": str(checkpoint), "epoch": 1,
              "class_index_map": {"bonafide": 1, "spoof": 0}, "head_ref": "linear_head.pt",
              "detector_state_ref": "detector_state.pt", "embedding_dim": d,
              "task_weight_origin": "trained_in_project", "training_phase": "full",
              "source_val_selection_ref": str(selection), "preprocess": {"sample_rate": 16000},
              "initialization": None, "parity_report_ref": "parity.json"}
    _json(export / "bundle.json", bundle)
    (export / "linear_head.pt").write_bytes(b"head")
    (export / "detector_state.pt").write_bytes(b"state")
    _json(export / "parity.json", {"status": "PASS", "module_modes_stable": True,
                                    "buffers_stable": True})

    resources_dir = tmp_path / "resources"
    write_frozen_resources(resources_dir, bundle,
        {"U": U.astype("float32"), "w": w.astype("float32"),
         "anchors_z": anchors.astype("float32"), "anchors_y": labels.astype("int64"),
         "anchors_m0": margins.astype("float32"), "anchors_s0": scores.astype("float32")},
        {"b": bias, "tau0": threshold})

    manifest = tmp_path / "select.jsonl"
    rows = [{"schema_version": "0.2.0", "sample_id": sample_id, "sample_index": index,
             "root_key": "fixture", "audio_relpath": sample_id + ".wav", "split_role": "select"}
            for index, sample_id in enumerate(("x", "y"))]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    identity = CacheIdentity("cache-select", "source-run", str(checkpoint), "fixture", "select",
                             str(manifest.resolve()), {"sample_rate": 16000}, {"num_views": 3},
                             13, "float32", {"dtype": "float32"})
    cache_dir = tmp_path / "cache"
    with FeatureCacheWriter(cache_dir, identity, ["x", "y"], 3, d) as writer:
        writer.add(["x", "y"], rng.normal(size=(2, 3, d)).astype("float32"))
    methods = [{"method_id": "frozen", "config": {"steps": 0, "lr": .01, "rho": .2,
                "gamma": .1, "lambda_keep": 1.0}, "params": {}},
               {"method_id": "ep_tta", "config": {"steps": 2, "lr": .01, "rho": .2,
                "gamma": .1, "lambda_keep": 1.0}, "params": {}},
               {"method_id": "tent_audio_ep", "config": {"steps": 1, "lr": .01, "rho": .2,
                "gamma": .1, "lambda_keep": 0.0}, "params": {}}]
    plan = {"schema_version": "0.3.0", "status": "READY", "suite_id": "fixture",
            "feature_cache_ref": str(cache_dir), "resources_ref": str(resources_dir),
            "frozen_bundle_ref": str(export / "bundle.json"), "input_manifest_ref": str(manifest),
            "input_role": "select", "scope_id": "fixture/select", "methods": methods}
    plan_path = tmp_path / "plan.json"
    _json(plan_path, plan)
    result = run_suite(plan_path, "fixture", "select", tmp_path / "run")
    assert result["target_labels_read"] is False and result["status"] == "PARTIAL"
    assert (tmp_path / "run" / "ep_tta" / "scores.jsonl").is_file()
    assert json.loads((tmp_path / "run" / "tent_audio_ep" / "run.json").read_text())["status"] == "NOT_RUN"
