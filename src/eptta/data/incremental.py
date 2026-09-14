"""Append-only snapshot reconciliation; parent records are never rewritten."""
import json
import shutil
from collections import Counter
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.data.splits import _staging_paths
from eptta.errors import ContractError, DataError


IDENTITY_FIELDS = ("canonical_label", "source_group_id", "group_quality", "audio_relpath",
                   "input_sha256", "source_release", "raw_contract_hash")


def build_delta(parent_ref, candidate_staging_ref, output):
    parent_path = Path(parent_ref)
    parent_meta_path = parent_path / "snapshot.json" if parent_path.is_dir() else parent_path
    parent = read_json(parent_meta_path)
    parent_records_path = parent_meta_path.parent / parent["canonical_ref"]
    if parent["status"] != "LOCKED" or sha256_file(parent_records_path) != parent["canonical_sha256"]:
        raise DataError("parent snapshot is not locked or has changed")
    candidate, candidate_path = _staging_paths(candidate_staging_ref)
    old = {record["sample_id"]: record for record in iter_jsonl(parent_records_path)}
    old_path_ids = {record["audio_relpath"]: record["sample_id"] for record in old.values()}
    old_group_roles = {}
    for record in old.values():
        group = record.get("source_group_id")
        role = record["split_role"]
        if group and role not in ("quarantine", "unassigned"):
            if group in old_group_roles and old_group_roles[group] != role:
                raise DataError(f"parent source group already crosses roles: {group}")
            old_group_roles[group] = role
    rows = []
    counts = Counter()
    added_role_counts = Counter()
    for record in iter_jsonl(candidate_path):
        previous = old.get(record["sample_id"])
        if previous:
            changed = [field for field in IDENTITY_FIELDS if previous.get(field) != record.get(field)]
            status = "unchanged" if not changed else "relabeled" if changed == ["canonical_label"] else "conflict"
            rows.append({"status": status, "sample_id": record["sample_id"], "changed_fields": changed,
                         "candidate": record if status != "unchanged" else None})
        elif record["audio_relpath"] in old_path_ids:
            rows.append({"status": "duplicate", "sample_id": record["sample_id"],
                         "existing_sample_id": old_path_ids[record["audio_relpath"]], "candidate": record})
            status = "duplicate"
        else:
            group = record.get("source_group_id")
            related = {}
            if group in old_group_roles:
                related[f"group:{group}"] = old_group_roles[group]
            parent_record = old.get(record.get("parent_id"))
            if parent_record and parent_record["split_role"] not in ("unassigned", "quarantine"):
                related[f"parent:{parent_record['sample_id']}"] = parent_record["split_role"]
            if len(set(related.values())) > 1:
                record["split_role"] = "quarantine"
                record["status"] = "lineage_conflict"
                rows.append({"status": "lineage_conflict", "sample_id": record["sample_id"],
                             "related_roles": related, "candidate": record})
                counts["lineage_conflict"] += 1
                continue
            inherited_role = next(iter(related.values()), None)
            if inherited_role:
                record["split_role"] = inherited_role
                record["status"] = "incremental_inherited_role"
            else:
                record["split_role"] = "unassigned"
                record["status"] = "incremental_unassigned"
            rows.append({"status": "added", "sample_id": record["sample_id"], "candidate": record,
                         "inherited_role": inherited_role})
            status = "added"
            added_role_counts[record["split_role"]] += 1
        counts[status] += 1
    with AtomicDirectory(output) as temporary:
        delta_path = temporary / "delta.jsonl"
        with delta_path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        metadata = {"schema_version": "0.1.0", "status": "PROPOSED", "delta_id": "delta-" +
                    content_hash({"parent": parent["canonical_sha256"], "candidate": candidate["records_sha256"]})[:20],
                    "parent_snapshot_ref": str(parent_meta_path), "parent_snapshot_id": parent["snapshot_id"],
                    "candidate_staging_ref": str(candidate_staging_ref), "delta_ref": "delta.jsonl",
                    "delta_sha256": sha256_file(delta_path), "counts": dict(sorted(counts.items())),
                    "added_role_counts": dict(sorted(added_role_counts.items())),
                    "automatic_resplit": False, "automatic_retrain": False}
        write_json_new(temporary / "delta.json", metadata)
    return metadata


def commit_delta(delta_ref, review, output):
    delta_path = Path(delta_ref)
    metadata_path = delta_path / "delta.json" if delta_path.is_dir() else delta_path
    delta = read_json(metadata_path)
    required = {"accepted", "reviewer", "approved_at", "report_ref", "sample_evidence_ref"}
    if set(review) != required or review["accepted"] is not True:
        raise ContractError("delta review must explicitly accept and include reviewer/time/evidence")
    rows_path = metadata_path.parent / delta["delta_ref"]
    if sha256_file(rows_path) != delta["delta_sha256"]:
        raise DataError("delta changed after proposal")
    parent_meta_path = Path(delta["parent_snapshot_ref"])
    parent = read_json(parent_meta_path)
    parent_records_path = parent_meta_path.parent / parent["canonical_ref"]
    if sha256_file(parent_records_path) != parent["canonical_sha256"]:
        raise DataError("parent snapshot changed")
    accepted = [row["candidate"] for row in iter_jsonl(rows_path) if row["status"] == "added"]
    snapshot_id = "snapshot-" + content_hash({"parent": parent["snapshot_id"], "delta": delta["delta_sha256"],
                                               "review": review})[:20]
    with AtomicDirectory(output) as temporary:
        canonical = temporary / "canonical.jsonl"
        with canonical.open("x", encoding="utf-8") as stream:
            for record in iter_jsonl(parent_records_path):
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            for record in accepted:
                record["snapshot_id"] = snapshot_id
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        parent_manifests = parent_meta_path.parent / "manifests"
        parent_labels = parent_meta_path.parent / "evaluation_labels"
        if not parent_manifests.is_dir() or not parent_labels.is_dir():
            raise DataError("parent role/label manifests are unavailable")
        shutil.copytree(parent_manifests, temporary / "manifests")
        shutil.copytree(parent_labels, temporary / "evaluation_labels")
        for record in accepted:
            role = record["split_role"]
            if role in ("unassigned", "quarantine"):
                continue
            manifest_path = temporary / "manifests" / f"{role}.jsonl"
            label_path = temporary / "evaluation_labels" / f"{role}.jsonl"
            item = {"schema_version": "0.1.0", "sample_id": record["sample_id"],
                    "root_key": record["root_key"], "audio_relpath": record["audio_relpath"],
                    "input_sha256": record["input_sha256"], "split_role": role}
            if role in ("fit", "source_val"):
                item["canonical_label"] = record["canonical_label"]
            with manifest_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            with label_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"schema_version": "0.1.0", "sample_id": record["sample_id"],
                                         "canonical_label": record["canonical_label"]},
                                        sort_keys=True, separators=(",", ":")) + "\n")
        role_hashes = {path.name: sha256_file(path) for path in sorted((temporary / "manifests").glob("*.jsonl"))}
        label_hashes = {path.name: sha256_file(path) for path in sorted((temporary / "evaluation_labels").glob("*.jsonl"))}
        role_counts = Counter(parent.get("role_counts", {}))
        role_counts.update(record["split_role"] for record in accepted)
        result = {"schema_version": "0.1.0", "status": "LOCKED", "snapshot_id": snapshot_id,
                  "parent_snapshot_id": parent["snapshot_id"], "dataset_id": parent["dataset_id"],
                  "record_count": parent["record_count"] + len(accepted), "canonical_ref": "canonical.jsonl",
                  "canonical_sha256": sha256_file(canonical), "delta_id": delta["delta_id"],
                  "delta_sha256": delta["delta_sha256"], "review": review, "immutable": True,
                  "role_counts": dict(sorted(role_counts.items())), "role_manifest_hashes": role_hashes,
                  "evaluation_label_hashes": label_hashes, "role_manifests_require_rebuild": False}
        result["fixture_only"] = False
        write_json_new(temporary / "snapshot.json", result)
    return result


def refresh_plan(delta, artifacts):
    counts = delta["counts"]
    has_added = counts.get("added", 0) > 0
    has_source_change = delta.get("added_role_counts", {}).get("fit", 0) > 0
    checkpoint_changed = delta.get("checkpoint_changed", False)
    rows = []
    current_hashes = artifacts.get("current_dependency_hashes", {})
    for item in artifacts.get("artifacts", []):
        bound_hashes = item.get("dependency_hashes", {})
        changed_dependencies = sorted(key for key, old_hash in bound_hashes.items()
                                      if key in current_hashes and current_hashes[key] != old_hash)
        if changed_dependencies:
            action = "rebuild_dependency_hash_changed"
        elif checkpoint_changed and item.get("model_dependent", item.get("kind") not in ("raw_audio", "raw_decode")):
            action = "rebuild_new_checkpoint_identity"
        elif not has_added:
            action = "reuse"
        elif item.get("kind") in ("raw_decode", "frozen_model"):
            action = "reuse"
        elif item.get("depends_on_fit"):
            action = "blocked_pending_retrain_decision"
        elif item.get("kind") in ("target_features", "scores"):
            action = "append_new_inputs"
        else:
            action = "review"
        row = {"artifact_id": item.get("artifact_id"), "action": action}
        if changed_dependencies:
            row["changed_dependencies"] = changed_dependencies
        rows.append(row)
    return {"schema_version": "0.1.0", "status": "PREVIEW_ONLY", "automatic_retrain": False,
            "automatic_resplit": False, "source_change_requires_new_model_identity": has_source_change, "artifacts": rows}
