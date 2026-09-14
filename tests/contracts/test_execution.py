import json

import pytest

from eptta.data.io import sha256_file
from eptta.errors import DataError
from eptta.execution.extract import compile_inference_job


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def extraction_plan(tmp_path, with_label=False):
    export = tmp_path / "export"
    export.mkdir()
    head = export / "linear_head.pt"
    head.write_bytes(b"head")
    bundle = {"baseline_id": "baseline", "selected_checkpoint_sha256": "a" * 64,
              "head_ref": "linear_head.pt", "eval_preprocess_hash": "b" * 64}
    (export / "bundle.json").write_text(json.dumps(bundle))
    manifest = tmp_path / "manifest.jsonl"
    row = {"schema_version": "0.1.0", "sample_id": "sample", "root_key": "data",
           "audio_relpath": "sample.wav", "input_sha256": "c" * 64, "split_role": "target_test"}
    if with_label:
        row["canonical_label"] = 1
    write_jsonl(manifest, [row])
    export_manifest = {"files": {"bundle.json": sha256_file(export / "bundle.json"),
                                  "linear_head.pt": sha256_file(head)}}
    (export / "export_manifest.json").write_text(json.dumps(export_manifest))
    plan = {"schema_version": "0.1.0", "status": "LOCKED",
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
    assert "canonical_label" not in json.dumps(job)


def test_inference_job_rejects_target_label_leak(tmp_path):
    with pytest.raises(DataError, match="labels"):
        compile_inference_job(extraction_plan(tmp_path, with_label=True), 0, 1)
