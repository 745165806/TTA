"""Validation and sealing of the two source-training role manifests.

This module deliberately accepts a published snapshot, not arbitrary CSV files.
It never assigns roles: the reviewed split lock has already done that work.
"""
from collections import Counter
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import DataError
from eptta.data.roles import ROLES
import json


SOURCE_ROLES = ("fit", "source_val")


def _snapshot_paths(snapshot_ref):
    path = Path(snapshot_ref)
    metadata_path = path / "snapshot.json" if path.is_dir() else path
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != "0.1.0" or metadata.get("status") != "LOCKED":
        raise DataError("source training requires a LOCKED v0.1.0 snapshot")
    if metadata.get("fixture_only") is not False:
        raise DataError("production source manifests reject fixture/unspecified snapshots")
    canonical = metadata_path.parent / metadata.get("canonical_ref", "canonical.jsonl")
    if not canonical.is_file() or sha256_file(canonical) != metadata.get("canonical_sha256"):
        raise DataError("snapshot canonical manifest is missing or changed")
    return metadata_path, metadata


def validate_source_snapshot(snapshot_ref):
    """Return hash-bound fit/source_val references after exact row validation."""
    metadata_path, metadata = _snapshot_paths(snapshot_ref)
    result = {}
    all_ids = set()
    for role in SOURCE_ROLES:
        manifest = metadata_path.parent / "manifests" / (role + ".jsonl")
        expected_hash = metadata.get("role_manifest_hashes", {}).get(manifest.name)
        if not manifest.is_file() or not expected_hash or sha256_file(manifest) != expected_hash:
            raise DataError("source role manifest is missing or changed: %s" % role)
        ids = set()
        root_keys = set()
        labels = Counter()
        for row_number, row in enumerate(iter_jsonl(manifest), 1):
            required = {"schema_version", "sample_id", "root_key", "audio_relpath",
                        "input_sha256", "split_role", "canonical_label"}
            if set(row) != required:
                raise DataError("%s row %d has an invalid/unsafe field set" % (role, row_number))
            if row["schema_version"] != "0.1.0" or row["split_role"] != role:
                raise DataError("%s row %d has a mismatched role/version" % (role, row_number))
            if type(row["canonical_label"]) is not int or row["canonical_label"] not in (0, 1):
                raise DataError("%s row %d has no canonical binary label" % (role, row_number))
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids or sample_id in all_ids:
                raise DataError("duplicate/empty source sample id: %r" % sample_id)
            ids.add(sample_id)
            root_keys.add(row["root_key"])
            labels[row["canonical_label"]] += 1
        if not ids:
            raise DataError("source role manifest is empty: %s" % role)
        if set(labels) != {0, 1}:
            raise DataError("source role %s must contain both canonical classes" % role)
        all_ids.update(ids)
        result[role] = {"manifest_ref": str(manifest.resolve()), "manifest_sha256": expected_hash,
                        "snapshot_hash": metadata["canonical_sha256"], "snapshot_id": metadata["snapshot_id"],
                        "sample_count": len(ids), "label_counts": {str(k): labels[k] for k in sorted(labels)},
                        "root_keys": sorted(root_keys)}
    return {"schema_version": "0.1.0", "status": "VALID", "snapshot_ref": str(metadata_path.resolve()),
            "snapshot_id": metadata["snapshot_id"], "snapshot_hash": metadata["canonical_sha256"],
            "roles": result}


def seal_source_manifests(snapshot_ref, output):
    """Publish an immutable small index; source rows stay in their snapshot."""
    validated = validate_source_snapshot(snapshot_ref)
    identity = {"snapshot_hash": validated["snapshot_hash"],
                "role_hashes": {k: v["manifest_sha256"] for k, v in validated["roles"].items()}}
    validated["source_manifest_set_id"] = "source-manifests-" + content_hash(identity)[:20]
    validated["status"] = "LOCKED"
    validated["immutable"] = True
    with AtomicDirectory(output) as temporary:
        write_json_new(temporary / "source_manifests.json", validated)
    return validated


def publish_label_free_manifest(snapshot_ref, role, output, status="LOCKED"):
    """Create an immutable inference view; labels remain only in sidecars."""
    if role not in ROLES or role in ("unassigned", "quarantine"):
        raise DataError("invalid inference role")
    metadata_path, metadata = _snapshot_paths(snapshot_ref)
    source = metadata_path.parent / "manifests" / (role + ".jsonl")
    expected = metadata.get("role_manifest_hashes", {}).get(source.name)
    if not source.is_file() or sha256_file(source) != expected:
        raise DataError("role manifest is missing or changed: %s" % role)
    rows = []
    seen = set()
    for row in iter_jsonl(source):
        if row.get("split_role") != role or row["sample_id"] in seen:
            raise DataError("role manifest mismatch or duplicate ID")
        seen.add(row["sample_id"])
        rows.append({key: row[key] for key in ("schema_version", "sample_id", "root_key",
                                                "audio_relpath", "input_sha256", "split_role")})
    if status not in ("PROPOSED", "LOCKED"):
        raise DataError("inference manifest status must be PROPOSED or LOCKED")
    with AtomicDirectory(output) as temporary:
        manifest = temporary / (role + ".jsonl")
        with manifest.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        result = {"schema_version": "0.1.0", "status": status, "role": role,
                  "source_snapshot_hash": metadata["canonical_sha256"],
                  "source_role_manifest_sha256": expected, "manifest_ref": manifest.name,
                  "manifest_sha256": sha256_file(manifest), "sample_count": len(rows),
                  "labels_in_manifest": False, "immutable": status == "LOCKED"}
        write_json_new(temporary / "inference_manifest.json", result)
    return result
