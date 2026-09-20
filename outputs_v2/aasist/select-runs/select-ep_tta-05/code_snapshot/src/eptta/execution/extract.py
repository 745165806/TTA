"""Compile and launch label-free frozen feature extraction shards."""
from pathlib import Path
import subprocess
import sys

from eptta.cache.keys import CacheIdentity
from eptta.config.schema import read_document
from eptta.data.io import iter_jsonl, write_json_new
from eptta.errors import ContractError, DataError, ResourceError
from eptta.models.frozen import verify_frozen_export


def compile_inference_job(plan_ref, worker_slot, worker_count):
    plan = read_document(plan_ref)
    required = {"schema_version", "status", "purpose", "input_role", "frozen_bundle_ref",
                "manifest_ref", "dataset_id", "data_roots", "probe", "numerical_mode",
                "cache_id", "output_root"}
    if set(plan) != required or plan.get("schema_version") != "0.3.0" or plan.get("status") != "READY":
        raise ContractError("inference plan must be a strict READY document")
    if type(worker_slot) is not int or type(worker_count) is not int or not 0 <= worker_slot < worker_count:
        raise ContractError("invalid extraction worker shard")
    manifest = Path(plan["manifest_ref"])
    if not manifest.is_file():
        raise DataError("inference manifest is missing")
    role_policy = {"source_prepare": {"fit", "cal0"}, "select": {"select"},
                   "audit": {"audit", "cal1"},
                   "confirmatory": {"control_test", "target_test"}}
    if plan["purpose"] not in role_policy or plan["input_role"] not in role_policy[plan["purpose"]]:
        raise ContractError("inference purpose/input_role violates data-role policy")
    ids = []
    required_row = {"schema_version", "sample_id", "root_key", "audio_relpath", "split_role"}
    allowed_row = required_row | {"sample_index"}
    for row in iter_jsonl(manifest):
        if not required_row.issubset(row) or set(row) - allowed_row:
            raise DataError("inference manifest contains labels or forbidden metadata")
        if row["split_role"] != plan["input_role"]:
            raise DataError("inference manifest role differs from requested role")
        ids.append(row["sample_id"])
    if not ids or len(ids) != len(set(ids)):
        raise DataError("inference manifest IDs must be unique and nonempty")
    bundle_ref = Path(plan["frozen_bundle_ref"])
    bundle, _export, _parity, _selection = verify_frozen_export(bundle_ref)
    identity = CacheIdentity(
        cache_id=plan["cache_id"], source_run_id=bundle["source_run_id"],
        checkpoint_ref=bundle["checkpoint_ref"], dataset_id=plan["dataset_id"],
        split_role=plan["input_role"], manifest_ref=str(manifest.resolve()),
        preprocess=bundle["preprocess"], views=plan["probe"], seed=plan["probe"]["seed"],
        dtype=plan["numerical_mode"]["dtype"], numerical_mode=plan["numerical_mode"])
    shard_ids = ids[worker_slot::worker_count]
    output = Path(plan["output_root"]) / ("shard-%05d-of-%05d" % (worker_slot, worker_count))
    return {"schema_version": "0.3.0", "job_type": "inference",
            "bundle_ref": str(bundle_ref.resolve()), "purpose": plan["purpose"],
            "input_role": plan["input_role"], "manifest_ref": str(manifest.resolve()),
            "data_roots": plan["data_roots"], "probe": plan["probe"],
            "numerical_mode": plan["numerical_mode"], "worker_slot": worker_slot,
            "worker_count": worker_count, "expected_ids": shard_ids,
            "cache_identity": identity.as_dict(), "output_dir": str(output.resolve())}


def launch_extraction(job):
    output = Path(job["output_dir"])
    if output.exists():
        raise ContractError("extraction shard output exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = output.parent / (output.name + ".job.json")
    write_json_new(job_path, job)
    worker = Path(__file__).parents[3] / "workers/baseline_bridge.py"
    if not worker.is_file():
        raise ContractError("feature extraction worker is missing")
    completed = subprocess.run([sys.executable, str(worker), "extract", "--job", str(job_path)],
                               check=False)
    if completed.returncode:
        raise ResourceError("feature extraction worker failed with exit code %d" % completed.returncode)
    return read_document(output / "index.json")
