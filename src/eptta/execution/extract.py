"""Compile and launch label-free frozen feature extraction shards."""
import json
from pathlib import Path
import subprocess
import sys

from eptta.cache.keys import CacheIdentity
from eptta.config.schema import read_document
from eptta.config.validate import content_hash
from eptta.data.io import iter_jsonl, sha256_file, write_json_new
from eptta.errors import ContractError, DataError, ResourceError


def compile_inference_job(plan_ref, worker_slot, worker_count):
    plan = read_document(plan_ref)
    required = {"schema_version", "status", "frozen_bundle_ref", "manifest_ref", "manifest_sha256",
                "data_roots", "probe", "numerical_mode", "output_root"}
    if set(plan) != required or plan["schema_version"] != "0.1.0" or plan["status"] != "LOCKED":
        raise ContractError("inference plan must be a strict LOCKED v0.1.0 document")
    if type(worker_slot) is not int or type(worker_count) is not int or not 0 <= worker_slot < worker_count:
        raise ContractError("invalid extraction worker shard")
    manifest = Path(plan["manifest_ref"])
    if not manifest.is_file() or sha256_file(manifest) != plan["manifest_sha256"]:
        raise DataError("inference manifest is missing or changed")
    ids = []
    for row in iter_jsonl(manifest):
        allowed = {"schema_version", "sample_id", "root_key", "audio_relpath", "input_sha256", "split_role"}
        if set(row) != allowed:
            raise DataError("inference manifest contains labels or unrecognized fields")
        if row["split_role"] in ("fit", "source_val", "unassigned", "quarantine"):
            raise DataError("inference plan requires an assigned non-training role")
        ids.append(row["sample_id"])
    if len(ids) != len(set(ids)) or not ids:
        raise DataError("inference manifest IDs must be unique and nonempty")
    bundle_ref = Path(plan["frozen_bundle_ref"])
    bundle = read_document(bundle_ref)
    export_manifest = read_document(bundle_ref.parent / "export_manifest.json")
    if export_manifest.get("files", {}).get("bundle.json") != sha256_file(bundle_ref):
        raise DataError("frozen bundle changed after export")
    head_hash = export_manifest["files"][bundle["head_ref"]]
    probe_hash = content_hash(plan["probe"])
    identity = CacheIdentity(plan["manifest_sha256"], bundle["baseline_id"],
                             bundle["selected_checkpoint_sha256"], head_hash,
                             "author_wrapper_v1", bundle["eval_preprocess_hash"], probe_hash,
                             plan["probe"]["seed"], plan["numerical_mode"]["dtype"], plan["numerical_mode"])
    shard_ids = ids[worker_slot::worker_count]
    output = Path(plan["output_root"]) / ("shard-%05d-of-%05d" % (worker_slot, worker_count))
    return {"schema_version": "0.1.0", "job_type": "inference", "bundle_ref": str(bundle_ref.resolve()),
            "manifest_ref": str(manifest.resolve()), "manifest_sha256": plan["manifest_sha256"],
            "data_roots": plan["data_roots"], "probe": plan["probe"],
            "numerical_mode": plan["numerical_mode"], "worker_slot": worker_slot,
            "worker_count": worker_count, "expected_ids": shard_ids,
            "cache_identity": identity.as_dict(), "cache_key": identity.cache_key,
            "output_dir": str(output.resolve())}


def launch_extraction(job, worker_ref=None, python_executable=None):
    output = Path(job["output_dir"])
    if output.exists():
        raise ContractError("extraction shard output exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = output.parent / (output.name + ".job.json")
    write_json_new(job_path, job)
    worker = Path(worker_ref) if worker_ref else Path(__file__).parents[3] / "workers/baseline_bridge.py"
    completed = subprocess.run([python_executable or sys.executable, str(worker), "extract", "--job", str(job_path)],
                               check=False)
    if completed.returncode:
        raise ResourceError("feature extraction worker failed with exit code %d" % completed.returncode)
    return read_document(output / "index.json")
