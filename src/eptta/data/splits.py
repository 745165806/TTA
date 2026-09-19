"""Stable, group-preserving split generation without content digests.

Existing assignment files are consumed as data and never regenerated.  For a new
dataset, callers pass the complete group set: groups are sorted once, numbered,
then assigned by an independent PRNG seeded from the explicit integer seed.  The
saved assignment file is the authority for later runs.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from eptta.data.io import iter_jsonl, read_json, write_json_new
from eptta.data.roles import ROLES
from eptta.errors import ContractError, DataError


def _validated_ratios(ratios):
    if (not isinstance(ratios, dict) or not ratios or set(ratios) - ROLES or
            any(role in ("unassigned", "quarantine") for role in ratios) or
            any(type(value) not in (int, float) or isinstance(value, bool) or value <= 0
                for value in ratios.values())):
        raise ContractError("split ratios must be explicit positive values for known roles")
    return [(role, float(value)) for role, value in sorted(ratios.items())]


def stable_group_roles(seed, group_ids, ratios):
    """Return order-independent assignments for one complete new group set."""
    if type(seed) is not int:
        raise ContractError("split seed must be an integer")
    groups = sorted(group_ids)
    if not groups or any(not isinstance(group, str) or not group for group in groups):
        raise ContractError("split groups must be nonempty strings")
    if len(groups) != len(set(groups)):
        raise ContractError("split group IDs must be unique")
    weights = _validated_ratios(ratios)
    total = sum(value for _role, value in weights)
    rng = random.Random(seed)
    result = {}
    for sample_index, group_id in enumerate(groups):
        point, running = rng.random(), 0.0
        for role, value in weights:
            running += value / total
            if point < running:
                result[group_id] = role
                break
        else:
            result[group_id] = weights[-1][0]
        # sample_index is deliberately implicit in sorted order and recorded by
        # prepare-data in the assignment table.
        assert sample_index >= 0
    return result


def propose_splits(staging_ref, policy, output):
    """Legacy-callable ordinary split writer; no proposal/approval transition."""
    root = Path(staging_ref)
    staging_path = root / "staging.json" if root.is_dir() else root
    staging = read_json(staging_path)
    records_ref = staging.get("records_ref")
    if not records_ref:
        raise DataError("staging document does not name records_ref")
    records_path = staging_path.parent / records_ref
    if not records_path.is_file():
        raise DataError("staging records are missing")
    payload = policy.get("payload", policy)
    ratios, pool_ratios = payload.get("ratios"), payload.get("pool_ratios")
    if bool(ratios) == bool(pool_ratios):
        raise ContractError("provide exactly one of ratios or pool_ratios")
    seed = payload.get("seed", 13)
    groups = defaultdict(lambda: {"sample_ids": [], "pools": set()})
    for record in iter_jsonl(records_path):
        group_id = record.get("source_group_id")
        if record.get("split_role") == "quarantine" or not group_id:
            continue
        groups[group_id]["sample_ids"].append(record["sample_id"])
        groups[group_id]["pools"].add(record.get("official_split"))
    if ratios:
        group_roles = stable_group_roles(seed, groups, ratios)
    else:
        group_roles = {}
        for pool in sorted(pool_ratios):
            members = sorted(group_id for group_id, item in groups.items() if item["pools"] == {pool})
            if not members:
                continue
            group_roles.update(stable_group_roles(seed, members, pool_ratios[pool]))
        if set(group_roles) != set(groups):
            raise DataError("groups spanning/missing official split pools require explicit resolution")
    assignment_path = Path(output).with_name(Path(output).stem + ".assignments.jsonl")
    if Path(output).exists() or assignment_path.exists():
        raise ContractError("split output already exists")
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    sample_index_by_id = {sample_id: index for index, sample_id in enumerate(sorted(
        sample_id for group in groups.values() for sample_id in group["sample_ids"]))}
    with assignment_path.open("x", encoding="utf-8") as stream:
        for group_id, group in sorted(groups.items()):
            role = group_roles[group_id]
            for sample_id in sorted(group["sample_ids"]):
                counts[role] += 1
                stream.write(json.dumps({"schema_version": "0.2.0", "sample_id": sample_id,
                                         "sample_index": sample_index_by_id[sample_id],
                                         "source_group_id": group_id,
                                         "split_role": role}, sort_keys=True,
                                        separators=(",", ":")) + "\n")
    result = {"schema_version": "0.2.0", "status": "READY",
              "assignments_ref": assignment_path.name, "ratios": ratios,
              "pool_ratios": pool_ratios, "counts": dict(sorted(counts.items())),
              "seed": seed, "assignment_method": "sorted_group_prng_v1",
              "preserve_existing_assignments": True}
    write_json_new(output, result)
    return result
