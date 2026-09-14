"""Group-preserving split proposals. Proposals never become approvals automatically."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from eptta.data.io import iter_jsonl, sha256_file, write_json_new
from eptta.data.roles import ROLES
from eptta.errors import ContractError, DataError


def _staging_paths(staging_ref):
    path = Path(staging_ref)
    manifest_path = path / "staging.json" if path.is_dir() else path
    from eptta.data.io import read_json
    manifest = read_json(manifest_path)
    records = manifest_path.parent / manifest["records_ref"]
    if manifest["status"] != "STAGED" or sha256_file(records) != manifest["records_sha256"]:
        raise DataError("staging snapshot is blocked or its records changed")
    return manifest, records


def propose_splits(staging_ref, policy, output):
    manifest, records_path = _staging_paths(staging_ref)
    payload = policy["payload"] if "payload" in policy else policy
    ratios = payload.get("ratios")
    pool_ratios = payload.get("pool_ratios")
    if bool(ratios) == bool(pool_ratios):
        raise ContractError("provide exactly one of ratios or pool_ratios")
    ratio_sets = [ratios] if ratios else list(pool_ratios.values())
    for ratio_set in ratio_sets:
        if not isinstance(ratio_set, dict) or not ratio_set:
            raise ContractError("every split pool requires explicit nonempty ratios")
        if set(ratio_set) - ROLES or any(role in ("unassigned", "quarantine") for role in ratio_set):
            raise ContractError("invalid assignable split role")
        if any(type(value) not in (int, float) or isinstance(value, bool) or value <= 0
               for value in ratio_set.values()):
            raise ContractError("split ratios must be positive numbers")
    seed = payload.get("seed", 13)
    groups = defaultdict(lambda: {"sample_ids": [], "pools": set()})
    for record in iter_jsonl(records_path):
        if record["split_role"] == "quarantine" or not record.get("source_group_id"):
            continue
        else:
            group = groups[record["source_group_id"]]
            group["sample_ids"].append(record["sample_id"])
            group["pools"].add(record.get("official_split"))
    group_roles = {}
    counts = Counter()
    for group_id, group in sorted(groups.items()):
        if pool_ratios:
            pools = group["pools"]
            if None in pools or any(pool not in pool_ratios for pool in pools):
                raise ContractError(f"group has official_split outside reviewed pools: {group_id}")
            allowed = set.intersection(*(set(pool_ratios[pool]) for pool in pools))
            if not allowed:
                raise DataError(f"source group spans incompatible official split pools: {group_id}")
            ratio_set = {role: sum(pool_ratios[pool].get(role, 0) for pool in pools)
                         for role in sorted(allowed)}
        else:
            ratio_set = ratios
        total = sum(ratio_set.values())
        normalized = {role: value / total for role, value in ratio_set.items()}
        cumulative, running = [], 0.0
        for role, value in normalized.items():
            running += value
            cumulative.append((running, role))
        key = hashlib.sha256(f"{seed}\0{group_id}".encode()).digest()
        point = int.from_bytes(key[:8], "big") / 2**64
        role = next(role for edge, role in cumulative if point < edge)
        group_roles[group_id] = role
        counts[role] += len(group["sample_ids"])
    assignment_path = Path(output).with_name(Path(output).stem + ".assignments.jsonl")
    if assignment_path.exists():
        raise ContractError(f"assignment output exists: {assignment_path}")
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("x", encoding="utf-8") as stream:
        for group_id, group in sorted(groups.items()):
            for sample_id in group["sample_ids"]:
                value = {"schema_version": "0.1.0", "sample_id": sample_id,
                         "source_group_id": group_id, "split_role": group_roles[group_id]}
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    result_payload = {
        "assignments_ref": assignment_path.name,
        "group_policy_hash": manifest["contract_hashes"]["group"],
        "ratios": ratios,
        "counts": dict(sorted(counts.items())),
        "preserve_existing_assignments": True,
        "new_group_default_role": payload.get("new_group_default_role", "unassigned"),
        "automatic_resplit": False,
        "automatic_retrain": False,
        "seed": seed,
        "staging_id": manifest["staging_id"],
        "staging_records_sha256": manifest["records_sha256"],
        "assignments_sha256": sha256_file(assignment_path),
        "assignment_method": "sha256_group_threshold_v1",
        "pool_ratios": pool_ratios,
    }
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval": None,
                "payload": result_payload}
    write_json_new(output, proposal)
    return proposal
