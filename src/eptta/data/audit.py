"""Read-only audit of an already published dataset snapshot."""
from collections import Counter, defaultdict
from pathlib import Path

from eptta.config.schema import check
from eptta.data.io import iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import DataError


EXPECTED_BOUNDARIES = {
    "fit": "train", "source_val": "dev", "select": "dev", "cal0": "dev",
    "audit": "dev", "control_test": "eval",
}


def audit_source_snapshot(snapshot_ref, output=None):
    root = Path(snapshot_ref).resolve()
    metadata_path = root / "snapshot.json" if root.is_dir() else root
    root = metadata_path.parent
    metadata = read_json(metadata_path)
    canonical = root / metadata.get("canonical_ref", "canonical.jsonl")
    if sha256_file(canonical) != metadata.get("canonical_sha256"):
        raise DataError("snapshot canonical hash mismatch")

    matrix = Counter()
    groups = defaultdict(Counter)
    speakers = defaultdict(set)
    attacks = defaultdict(set)
    ids = set()
    audio_paths = set()
    input_hashes = Counter()
    parent_known = 0
    cross_parent = 0
    boundary_errors = []
    role_ids = defaultdict(set)
    count = 0
    for row_number, row in enumerate(iter_jsonl(canonical), 1):
        count += 1
        sample_id = row.get("sample_id")
        role = row.get("split_role")
        partition = row.get("official_split")
        label = row.get("canonical_label")
        if sample_id in ids:
            raise DataError(f"duplicate canonical sample_id at row {row_number}")
        ids.add(sample_id)
        role_ids[role].add(sample_id)
        matrix[(partition, role, label)] += 1
        groups[role][row.get("source_group_id")] += 1
        if row.get("speaker_id") is not None:
            speakers[role].add(row["speaker_id"])
        if row.get("generator_id") not in (None, "-"):
            attacks[role].add(row["generator_id"])
        path = (row.get("root_key"), row.get("audio_relpath"))
        if path in audio_paths:
            raise DataError(f"duplicate canonical audio path at row {row_number}")
        audio_paths.add(path)
        if row.get("input_sha256"):
            input_hashes[row["input_sha256"]] += 1
        expected = EXPECTED_BOUNDARIES.get(role)
        if expected and partition != expected:
            boundary_errors.append({"sample_id": sample_id, "role": role,
                                    "official_partition": partition, "expected": expected})
        if row.get("parent_id"):
            parent_known += 1

    for row in iter_jsonl(canonical):
        parent = row.get("parent_id")
        if parent:
            parent_roles = [role for role, values in role_ids.items() if parent in values]
            if parent_roles and any(role != row.get("split_role") for role in parent_roles):
                cross_parent += 1

    manifest_findings = {}
    manifest_union = set()
    for role, expected_count in metadata.get("role_counts", {}).items():
        manifest = root / "manifests" / f"{role}.jsonl"
        expected_hash = metadata.get("role_manifest_hashes", {}).get(manifest.name)
        if not manifest.is_file() or sha256_file(manifest) != expected_hash:
            raise DataError(f"role manifest missing or changed: {role}")
        manifest_ids = [row["sample_id"] for row in iter_jsonl(manifest)]
        if len(manifest_ids) != len(set(manifest_ids)) or len(manifest_ids) != expected_count:
            raise DataError(f"role manifest count/uniqueness mismatch: {role}")
        if manifest_union.intersection(manifest_ids):
            raise DataError(f"sample crosses role manifests: {role}")
        manifest_union.update(manifest_ids)
        largest = max(groups[role].values()) if groups[role] else 0
        manifest_findings[role] = {
            "sample_count": len(manifest_ids), "group_count": len(groups[role]),
            "speaker_count": len(speakers[role]), "attack_coverage": sorted(attacks[role]),
            "max_group_count": largest,
            "max_group_share": largest / len(manifest_ids) if manifest_ids else None,
            "manifest_sha256": expected_hash,
        }
    if manifest_union != ids:
        raise DataError("role manifests do not cover canonical samples exactly once")

    report = {
        "schema_version": "0.1.0", "status": "PASS" if not boundary_errors else "REVIEW_REQUIRED",
        "read_only": True, "snapshot_id": metadata.get("snapshot_id"),
        "snapshot_ref": str(metadata_path), "snapshot_sha256": sha256_file(metadata_path),
        "canonical_sha256": metadata.get("canonical_sha256"), "record_count": count,
        "official_partition_role_class": [
            {"official_partition": key[0], "role": key[1], "canonical_label": key[2], "count": value}
            for key, value in sorted(matrix.items(), key=lambda item: tuple(str(v) for v in item[0]))
        ],
        "roles": manifest_findings, "unique_sample_ids": len(ids),
        "unique_audio_paths": len(audio_paths),
        "non_null_input_hash_count": sum(input_hashes.values()),
        "duplicate_non_null_input_hashes": sum(value - 1 for value in input_hashes.values() if value > 1),
        "boundary_errors": boundary_errors,
        "parent_relation": {"known_count": parent_known, "cross_role_count": cross_parent,
                            "unknown_count": count - parent_known,
                            "interpretation": "UNKNOWN when parent_id is null"},
        "audit_role_contract": "source-only damage evaluation; forbidden for training loss, rollback, or step selection",
        "audit_access_history": "UNKNOWN_WITHOUT_APPEND_ONLY_LEDGER",
    }
    check(report, "source_audit")
    if output:
        write_json_new(output, report)
    return report
