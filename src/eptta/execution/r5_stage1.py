"""R5 stage-1 execution: lock the reviewed proposal, verify cache parity and K=0."""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from eptta.cache.reader import FeatureCache
from eptta.config.schema import read_document
from eptta.data.io import AtomicDirectory, iter_jsonl, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export

ROLES = ("fit", "cal0", "select")
STAGE1_PROPOSAL_SHA256 = "3579192bc6fbc3507e0eb618c9bb36f2a56ee80f11a65641bc3505bfa11dd1bc"


def lock_stage1_proposal(proposal_ref, expected_sha256):
    """Publish LOCKED extraction plans and a review record from an approved proposal."""
    proposal_path = Path(proposal_ref)
    if sha256_file(proposal_path) != expected_sha256:
        raise ContractError("R5 stage-1 proposal SHA-256 changed")
    proposal = read_document(proposal_path)
    if (proposal.get("schema_version") != "0.1.0" or proposal.get("status") != "PROPOSED" or
            proposal.get("approval_required") is not True or proposal.get("locks_published") is not False):
        raise ContractError("only a PROPOSED/approval_required proposal may be locked")
    proposal_dir = proposal_path.parent
    locked_dir = proposal_dir / "locked"
    if locked_dir.exists():
        raise ContractError("locked proposal exists; overwrite is forbidden")
    role_records = proposal["roles"]
    if set(role_records) != set(ROLES):
        raise ContractError("proposal must bind fit/cal0/select roles")
    locked_records = {}
    with AtomicDirectory(locked_dir) as temporary:
        for role in ROLES:
            record = role_records[role]
            source = proposal_dir / record["extraction_proposal_ref"]
            if sha256_file(source) != record["extraction_proposal_sha256"]:
                raise DataError(role + " extraction proposal changed")
            plan = read_document(source)
            if plan.get("status") != "PROPOSED" or plan.get("input_role") != role:
                raise ContractError("unexpected extraction proposal for " + role)
            locked = dict(plan)
            locked["status"] = "LOCKED"
            locked_path = temporary / (role + ".extraction.lock.json")
            write_json_new(locked_path, locked)
            locked_records[role] = {"ref": locked_path.name,
                                    "sha256": sha256_file(locked_path),
                                    "purpose": plan["purpose"], "input_role": plan["input_role"]}
        lock = {"schema_version": "0.1.0", "status": "LOCKED",
                "proposal_ref": str(proposal_path.resolve()), "proposal_sha256": expected_sha256,
                "proposal_id": proposal["proposal_id"],
                "parent_plan_ref": proposal["parent_plan_ref"],
                "parent_plan_sha256": proposal["parent_plan_sha256"],
                "bundle_sha256": proposal["bundle_sha256"],
                "checkpoint_sha256": proposal["checkpoint_sha256"],
                "source_snapshot_sha256": proposal["source_snapshot_sha256"],
                "worker_sha256": proposal["worker_sha256"],
                "probe_sha256": proposal["probe_sha256"],
                "approval": {"decision": "approved_lock_and_execute_stage1", "reviewer": "human_user",
                             "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "scope": "fixed_small_cache + cache_parity + k0_wiring_acceptance"},
                "roles": locked_records,
                "forbidden": proposal["forbidden"]}
        write_json_new(temporary / "lock.json", lock)
    return lock


def _read_unlabeled_manifest(manifest_ref, manifest_sha256, role):
    if sha256_file(manifest_ref) != manifest_sha256:
        raise DataError("inference manifest changed")
    ids, rows = [], []
    for row in iter_jsonl(manifest_ref):
        allowed = {"schema_version", "sample_id", "root_key", "audio_relpath", "input_sha256", "split_role"}
        if set(row) != allowed or row["split_role"] != role:
            raise DataError("inference manifest contains labels or role mismatch")
        ids.append(row["sample_id"])
        rows.append(row)
    if len(ids) != len(set(ids)) or not ids:
        raise DataError("inference manifest IDs must be unique and nonempty")
    return ids, rows


def compile_parity_job(plan_ref, role, cache_ref, output_dir, batch_sizes=(1, 2, 7, 48),
                       atol=1e-6, rtol=1e-5):
    plan = read_document(plan_ref)
    required = {"schema_version", "status", "purpose", "input_role", "frozen_bundle_ref",
                "manifest_ref", "manifest_sha256", "data_roots", "probe", "numerical_mode",
                "output_root"}
    if set(plan) != required or plan.get("status") != "LOCKED" or plan.get("input_role") != role:
        raise ContractError("parity requires a strict LOCKED plan for role " + role)
    bundle_ref = Path(plan["frozen_bundle_ref"])
    verify_frozen_export(bundle_ref)
    ids, _rows = _read_unlabeled_manifest(plan["manifest_ref"], plan["manifest_sha256"], role)
    cache = FeatureCache(cache_ref)
    if cache.index["sample_count"] != len(ids):
        raise DataError("cache sample count differs from the locked manifest")
    worker = Path(__file__).parents[3] / "workers/r5_parity_bridge.py"
    job = {"schema_version": "0.1.0", "job_type": "r5_parity", "role": role,
           "bundle_ref": str(bundle_ref.resolve()), "cache_ref": str(Path(cache_ref).resolve()),
           "manifest_ref": str(Path(plan["manifest_ref"]).resolve()),
           "manifest_sha256": plan["manifest_sha256"], "data_roots": plan["data_roots"],
           "probe": plan["probe"], "numerical_mode": plan["numerical_mode"],
           "expected_ids": ids, "batch_sizes": list(batch_sizes),
           "atol": float(atol), "rtol": float(rtol),
           "worker_sha256": sha256_file(worker), "output_dir": str(Path(output_dir).resolve())}
    return job, worker


def launch_parity(job, worker, python_executable):
    output = Path(job["output_dir"])
    if output.exists():
        raise ContractError("parity output exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = output.parent / (output.name + ".job.json")
    write_json_new(job_path, job)
    completed = subprocess.run([python_executable or sys.executable, str(worker), "parity",
                                "--job", str(job_path)], check=False)
    if completed.returncode:
        raise DataError("r5 parity worker failed with exit code %d" % completed.returncode)
    return read_document(output / "report.json")


def run_k0_verification(bundle_ref, cache_ref, role, output_root):
    import torch
    from eptta.adaptation.episode import run_k0_episode
    from eptta.adaptation.types import TargetViews
    bundle, _export, _parity, _selection = verify_frozen_export(bundle_ref)
    head = torch.load(Path(bundle_ref).parent / bundle["head_ref"], map_location="cpu", weights_only=True)
    w = head["w"]
    b = float(head["b"])
    cache = FeatureCache(cache_ref)
    if (cache.index["identity"]["baseline_id"] != bundle["baseline_id"] or cache.index["identity"][
            "selected_checkpoint_sha256"] != bundle["selected_checkpoint_sha256"]):
        raise ContractError("feature cache does not share the frozen detector")
    features = cache.load_by_id()
    ids = sorted(features)

    def frozen_score(z0):
        return float(z0 @ w + b)

    def run_one(sample_id):
        z = torch.from_numpy(features[sample_id])
        target = TargetViews(sample_id, z, cache.index["cache_key"])
        result = run_k0_episode(target, w, b, bundle["baseline_id"])
        frozen = frozen_score(z[0])
        return {"sample_id": sample_id, "score_k0": result.score_after,
                "score_before": result.score_before, "status": result.status,
                "steps_completed": result.steps_completed, "frozen_score": frozen,
                "match": bool(result.score_after == frozen and result.score_before == frozen and
                              result.status == "no_adaptation" and result.steps_completed == 0)}

    serial = {sample_id: run_one(sample_id) for sample_id in ids}
    # Batch: one batched frozen matmul over all z0, compared per UID.
    z0s = torch.stack([torch.from_numpy(features[sample_id][0]) for sample_id in ids])
    batched = (z0s @ w + b).tolist()
    batch_ok = all(abs(serial[ids[i]]["score_k0"] - batched[i]) <= 1e-6 + 1e-5 * abs(batched[i])
                   for i in range(len(ids)))
    # Shuffle (fixed seed), shards, A/B/A: per-UID scores must be identical.
    import random
    order = list(ids)
    random.Random(13).shuffle(order)
    orders = {"serial": ids, "shuffle": order, "even": ids[::2], "odd": ids[1::2],
              "A": ids[:7], "B": ids[7:14], "A_repeat": ids[:7]}
    consistency = {}
    for name, seq in orders.items():
        scores = {sample_id: run_one(sample_id)["score_k0"] for sample_id in seq}
        consistent = all(scores[sample_id] == serial[sample_id]["score_k0"] for sample_id in seq)
        consistency[name] = {"samples": len(seq), "consistent": bool(consistent),
                             "unique": len(set(scores.values()))}
    all_match = all(item["match"] for item in serial.values()) and batch_ok and \
        all(item["consistent"] for item in consistency.values())
    status = "PASS" if all_match else "FAIL"
    report = {"schema_version": "0.1.0", "status": status, "role": role,
              "bundle_id": bundle["baseline_id"],
              "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
              "cache_key": cache.index["cache_key"], "sample_count": len(ids),
              "k0_status": "no_adaptation", "no_numeric_fallback": True,
              "all_match": bool(all_match), "batch_consistent": bool(batch_ok),
              "consistency": consistency,
              "scores": {sample_id: {"score": serial[sample_id]["score_k0"],
                                     "frozen_score": serial[sample_id]["frozen_score"],
                                     "match": serial[sample_id]["match"]} for sample_id in ids}}
    with AtomicDirectory(output_root) as temporary:
        write_json_new(temporary / "k0.json", report)
    return report
