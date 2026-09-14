"""Disk-indexed immutable reviewed snapshot and role-manifest publication."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

from eptta.config.validate import check_contract, content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.data.roles import SOURCE_TRAIN_ROLES
from eptta.data.splits import _staging_paths
from eptta.errors import ContractError, DataError


def _write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def build_manifests(staging_ref, split_plan_ref, output):
    staging, records_path = _staging_paths(staging_ref)
    split_path = Path(split_plan_ref)
    split = read_json(split_path)
    issues = check_contract(split, "split", "split")
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))
    payload = split["payload"]
    if payload.get("staging_id") != staging["staging_id"] or payload.get("staging_records_sha256") != staging["records_sha256"]:
        raise DataError("split proposal does not belong to this staging snapshot")
    assignments_path = split_path.parent / payload["assignments_ref"]
    if payload.get("assignments_sha256") != sha256_file(assignments_path):
        raise DataError("split assignments changed after review")
    snapshot_basis = {"staging": staging["records_sha256"], "split": split["approval"]["content_sha256"],
                      "assignments": payload["assignments_sha256"]}
    snapshot_id = "snapshot-" + content_hash(snapshot_basis)[:20]
    counts = Counter()
    record_count = 0
    with AtomicDirectory(output) as temporary:
        database_path = temporary / ".assignment-index.sqlite3"
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("CREATE TABLE assignments (sample_id TEXT PRIMARY KEY, group_id TEXT, role TEXT, seen INTEGER DEFAULT 0)")
        database.execute("CREATE TABLE group_roles (group_id TEXT PRIMARY KEY, role TEXT)")
        try:
            for item in iter_jsonl(assignments_path):
                try:
                    database.execute("INSERT INTO assignments(sample_id,group_id,role) VALUES(?,?,?)",
                                     (item["sample_id"], item["source_group_id"], item["split_role"]))
                except sqlite3.IntegrityError as exc:
                    raise DataError(f"duplicate split assignment: {item['sample_id']}") from exc
                previous = database.execute("SELECT role FROM group_roles WHERE group_id=?",
                                            (item["source_group_id"],)).fetchone()
                if previous and previous[0] != item["split_role"]:
                    raise DataError(f"source group crosses roles: {item['source_group_id']}")
                database.execute("INSERT OR IGNORE INTO group_roles(group_id,role) VALUES(?,?)",
                                 (item["source_group_id"], item["split_role"]))
            database.commit()
            role_dir = temporary / "manifests"
            label_dir = temporary / "evaluation_labels"
            role_dir.mkdir()
            label_dir.mkdir()
            canonical_path = temporary / "canonical.jsonl"
            streams = {}
            label_streams = {}
            try:
                with canonical_path.open("x", encoding="utf-8") as canonical_stream:
                    for record in iter_jsonl(records_path):
                        record_count += 1
                        if record["split_role"] == "quarantine":
                            role = "quarantine"
                        else:
                            assignment = database.execute("SELECT role FROM assignments WHERE sample_id=?",
                                                          (record["sample_id"],)).fetchone()
                            if not assignment:
                                raise DataError(f"split coverage missing sample: {record['sample_id']}")
                            role = assignment[0]
                            database.execute("UPDATE assignments SET seen=1 WHERE sample_id=?", (record["sample_id"],))
                        record["snapshot_id"] = snapshot_id
                        record["split_role"] = role
                        counts[role] += 1
                        _write_line(canonical_stream, record)
                        if role == "quarantine":
                            continue
                        if role not in streams:
                            streams[role] = (role_dir / f"{role}.jsonl").open("x", encoding="utf-8")
                            label_streams[role] = (label_dir / f"{role}.jsonl").open("x", encoding="utf-8")
                        item = {"schema_version": "0.1.0", "sample_id": record["sample_id"],
                                "root_key": record["root_key"], "audio_relpath": record["audio_relpath"],
                                "input_sha256": record["input_sha256"], "split_role": role}
                        if role in SOURCE_TRAIN_ROLES:
                            item["canonical_label"] = record["canonical_label"]
                        _write_line(streams[role], item)
                        _write_line(label_streams[role], {"schema_version": "0.1.0", "sample_id": record["sample_id"],
                                                          "canonical_label": record["canonical_label"]})
            finally:
                for stream in [*streams.values(), *label_streams.values()]:
                    stream.close()
            unseen = database.execute("SELECT COUNT(*) FROM assignments WHERE seen=0").fetchone()[0]
            if unseen:
                raise DataError(f"split coverage has {unseen} assignments absent from staging")
        finally:
            database.close()
        database_path.unlink()
        role_hashes = {path.name: sha256_file(path) for path in sorted(role_dir.glob("*.jsonl"))}
        label_hashes = {path.name: sha256_file(path) for path in sorted(label_dir.glob("*.jsonl"))}
        metadata = {"schema_version": "0.1.0", "status": "LOCKED", "snapshot_id": snapshot_id,
                    "parent_snapshot_id": None, "dataset_id": staging["dataset_id"],
                    "record_count": record_count, "role_counts": dict(sorted(counts.items())),
                    "canonical_ref": "canonical.jsonl", "canonical_sha256": sha256_file(canonical_path),
                    "role_manifest_hashes": role_hashes, "evaluation_label_hashes": label_hashes,
                    "split_contract_sha256": split["approval"]["content_sha256"],
                    "staging_records_sha256": staging["records_sha256"], "immutable": True,
                    "indexing": "sqlite_disk_v1", "fixture_only": False}
        write_json_new(temporary / "snapshot.json", metadata)
    return metadata
