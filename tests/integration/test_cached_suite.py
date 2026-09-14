import json

import numpy as np

from eptta.cache import CacheIdentity, FeatureCacheWriter
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
    bundle = {"baseline_id": "baseline", "selected_checkpoint_sha256": checkpoint}
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
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
                         "rho": .2, "gamma": .1, "lambda_keep": 1.0}, "params": {}}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = run_suite(plan_path, "fixture", "select", tmp_path / "run")
    assert result["target_labels_read"] is False and len(result["methods"]) == 2
    for method in ("frozen", "ep_tta"):
        seal = seal_scores(tmp_path / "run" / method)
        assert seal["sample_count"] == 2 and seal["labels_read"] is False
