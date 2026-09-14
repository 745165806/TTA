"""Reviewed protocol parsing and disk-indexed staging snapshot construction."""
import json
import sqlite3
from collections import Counter
from dataclasses import asdict

from eptta.config.schema import check
from eptta.config.validate import check_contract, content_hash
from eptta.data.adapters import DelimitedAdapter, JsonRecordsAdapter
from eptta.data.adapters.base import select_protocols
from eptta.data.contracts import ContractIssue
from eptta.data.io import AtomicDirectory, sha256_file, write_json_new
from eptta.data.permissions import resolve_under_root
from eptta.data.records import normalize_record
from eptta.errors import ContractError, DataError


def _locked(document, kind):
    check(document, kind)
    issues = check_contract(document, kind, kind)
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))


def _line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def stage_dataset(inventory, raw_contract, label_policy, group_policy, output):
    for kind, document in (("raw", raw_contract), ("label", label_policy), ("group", group_policy)):
        _locked(document, kind)
    if len(inventory["datasets"]) != 1:
        raise ContractError("stage-data currently requires a single-dataset inventory")
    dataset_id, observed = next(iter(inventory["datasets"].items()))
    payload = raw_contract["payload"]
    if any(context["root_key"] != dataset_id for context in payload["protocol_contexts"].values()):
        raise ContractError("protocol context root_key must match the selected dataset root binding")
    selected = select_protocols(observed["protocol_files"], payload.get("protocol_globs"))
    adapter = (JsonRecordsAdapter() if payload["format"] in ("json", "jsonl") or
               (payload["format"] == "sidecar" and payload.get("json_paths")) else DelimitedAdapter())
    staging_id = "staging-" + content_hash({"dataset": dataset_id, "inventory": observed,
        "raw": raw_contract["approval"]["content_sha256"], "label": label_policy["approval"]["content_sha256"],
        "group": group_policy["approval"]["content_sha256"]})[:20]
    issue_count = 0
    strict_blocked = False
    root = observed["root"]
    group_mapping = None
    resolver = group_policy["payload"]["resolver"]
    if resolver.startswith("mapping:"):
        from eptta.config.schema import read_document
        mapping_ref = group_policy["payload"]["source_mapping_ref"]
        if sha256_file(mapping_ref) != group_policy["payload"]["source_mapping_sha256"]:
            raise DataError("source group mapping changed after review")
        group_mapping = read_document(mapping_ref)
        if not isinstance(group_mapping, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                                      for key, value in group_mapping.items()):
            raise ContractError("source group mapping must be a JSON object of string IDs")
    with AtomicDirectory(output) as temporary:
        database_path = temporary / ".staging-index.sqlite3"
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("CREATE TABLE records (seq INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT UNIQUE, "
                         "audio_relpath TEXT, canonical_label INTEGER, payload TEXT, conflict INTEGER DEFAULT 0)")
        parse_path = temporary / "parse_errors.jsonl"
        conflict_path = temporary / "label_conflicts.jsonl"
        try:
            with parse_path.open("x", encoding="utf-8") as parse_stream:
                for protocol in selected:
                    if sha256_file(protocol["path"]) != protocol["sha256"]:
                        raise DataError(f"protocol changed since inventory: {protocol['path']}")
                    for raw_record in adapter.iter_file(protocol["path"], payload):
                        canonical, issues = normalize_record(raw_record, dataset_id, payload["dataset_release"], staging_id,
                                                              raw_contract, label_policy, group_policy, group_mapping)
                        if canonical is not None:
                            try:
                                audio = resolve_under_root(root, canonical["audio_relpath"])
                                if not audio.is_file():
                                    issues.append(ContractIssue("AUDIO_MISSING", "audio_relpath", raw_record.record_ref,
                                                                "resolved audio file does not exist", True))
                            except Exception as exc:
                                issues.append(ContractIssue("AUDIO_PATH_REJECTED", "audio_relpath", raw_record.record_ref,
                                                            str(exc), True))
                            if any(issue.blocking for issue in issues):
                                canonical["split_role"] = "quarantine"
                                canonical["status"] = "quarantined"
                            serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, allow_nan=False)
                            try:
                                database.execute("INSERT INTO records(sample_id,audio_relpath,canonical_label,payload) "
                                                 "VALUES(?,?,?,?)", (canonical["sample_id"], canonical["audio_relpath"],
                                                                     canonical["canonical_label"], serialized))
                            except sqlite3.IntegrityError:
                                previous = database.execute("SELECT payload FROM records WHERE sample_id=?",
                                                            (canonical["sample_id"],)).fetchone()
                                previous_record = json.loads(previous[0])
                                same = all(previous_record.get(key) == canonical.get(key) for key in
                                           ("audio_relpath", "canonical_label", "source_group_id"))
                                duplicate_issue = {"code": "DUPLICATE_RECORD" if same else "SAMPLE_ID_CONFLICT",
                                                   "field": "sample_id", "record_ref": canonical["raw_record_ref"],
                                                   "message": "duplicate raw record ignored" if same else
                                                              "same stable sample ID has conflicting data",
                                                   "blocking": not same}
                                _line(parse_stream, duplicate_issue)
                                issue_count += 1
                                strict_blocked |= duplicate_issue["blocking"] and payload["missing_policy"] == "error"
                                if not same:
                                    database.execute("UPDATE records SET conflict=1 WHERE sample_id=?",
                                                     (canonical["sample_id"],))
                        for issue in issues:
                            _line(parse_stream, asdict(issue))
                            issue_count += 1
                            strict_blocked |= issue.blocking and payload["missing_policy"] == "error"
                        if database.total_changes and database.total_changes % 10000 == 0:
                            database.commit()
            database.commit()
            conflicting_paths = [row[0] for row in database.execute(
                "SELECT audio_relpath FROM records WHERE canonical_label IS NOT NULL "
                "GROUP BY audio_relpath HAVING COUNT(DISTINCT canonical_label)>1")]
            if conflicting_paths:
                database.executemany("UPDATE records SET conflict=1 WHERE audio_relpath=?",
                                     ((path,) for path in conflicting_paths))
                database.commit()
            records_path = temporary / "records.jsonl"
            quarantine_path = temporary / "quarantine.jsonl"
            counts = Counter()
            with records_path.open("x", encoding="utf-8") as records_stream, \
                    quarantine_path.open("x", encoding="utf-8") as quarantine_stream, \
                    conflict_path.open("x", encoding="utf-8") as conflict_stream:
                for serialized, conflict in database.execute("SELECT payload,conflict FROM records ORDER BY seq"):
                    record = json.loads(serialized)
                    if conflict:
                        record["split_role"] = "quarantine"
                        record["status"] = "label_conflict"
                        issue = {"code": "LABEL_CONFLICT", "field": "canonical_label",
                                 "record_ref": record["raw_record_ref"],
                                 "message": "same path or stable ID has conflicting canonical data", "blocking": True}
                        _line(conflict_stream, issue)
                        issue_count += 1
                        strict_blocked |= payload["missing_policy"] == "error"
                    _line(records_stream, record)
                    if record["split_role"] == "quarantine":
                        _line(quarantine_stream, record)
                    counts[record["status"]] += 1
        finally:
            database.close()
        database_path.unlink()
        records_hash = sha256_file(temporary / "records.jsonl")
        manifest = {"schema_version": "0.1.0", "status": "BLOCKED_DATA" if strict_blocked else "STAGED",
                    "snapshot_kind": "staging", "staging_id": staging_id, "dataset_id": dataset_id,
                    "record_count": sum(counts.values()), "status_counts": dict(sorted(counts.items())),
                    "issue_count": issue_count, "records_ref": "records.jsonl", "records_sha256": records_hash,
                    "contract_hashes": {"raw": raw_contract["approval"]["content_sha256"],
                                        "label": label_policy["approval"]["content_sha256"],
                                        "group": group_policy["approval"]["content_sha256"]},
                    "source_inventory_sha256": content_hash(inventory), "immutable": True,
                    "indexing": "sqlite_disk_v1"}
        write_json_new(temporary / "staging.json", manifest)
    return manifest
