"""Build and validate auditable CSV label packs from explicitly described protocols.

The first seven columns intentionally match the lightweight ALLM-DF manifest
reader.  Remaining columns preserve the EP-TTA v0.1.0 canonical-data lineage.
Raw audio and source protocols are never modified.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping

from eptta.data.io import AtomicDirectory, sha256_file, write_json_new
from eptta.data.permissions import safe_relative
from eptta.data.records import opaque_sample_id
from eptta.errors import ContractError, DataError, PermissionDenied


SCHEMA_VERSION = "0.1.0"
COMPATIBILITY_FIELDS = ("utt_id", "path", "label", "attack", "speaker", "split", "source")
LINEAGE_FIELDS = (
    "sample_id",
    "dataset_id",
    "dataset_release",
    "root_key",
    "audio_relpath",
    "original_label",
    "label_mapping_status",
    "source_group_id",
    "group_quality",
    "protocol_file",
    "protocol_sha256",
    "protocol_line",
    "split_role",
    "status",
    "schema_version",
)
FIELDS = COMPATIBILITY_FIELDS + LINEAGE_FIELDS
REQUIRED_SOURCE_FIELDS = {
    "dataset_id",
    "dataset_release",
    "output_name",
    "root",
    "protocol",
    "format",
    "delimiter",
    "header",
    "columns",
    "record_id",
    "audio_path_template",
    "label_field",
    "label_map",
    "official_split",
    "source",
}


def _validate_output_name(value: Any) -> str:
    name = str(value)
    if not name.endswith(".csv") or Path(name).name != name or name.startswith("."):
        raise ContractError(f"output_name must be a plain non-hidden .csv filename: {value!r}")
    return name


def _template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def _validate_source(source: Mapping[str, Any], position: int) -> None:
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(source))
    if missing:
        raise ContractError(f"sources[{position}] missing fields: {missing}")
    if source["format"] != "delimited":
        raise ContractError(f"sources[{position}].format must be 'delimited'")
    delimiter = source["delimiter"]
    if delimiter != "whitespace" and (not isinstance(delimiter, str) or len(delimiter) != 1):
        raise ContractError(f"sources[{position}].delimiter must be one character or 'whitespace'")
    if not isinstance(source["header"], bool):
        raise ContractError(f"sources[{position}].header must be boolean")
    columns = source["columns"]
    if not isinstance(columns, dict) or not columns:
        raise ContractError(f"sources[{position}].columns must be a non-empty object")
    if source["record_id"] not in columns or source["label_field"] not in columns:
        raise ContractError(f"sources[{position}] record_id and label_field must name logical columns")
    expected_type = str if source["header"] else int
    if any(type(value) is not expected_type for value in columns.values()):
        kind = "header names" if source["header"] else "zero-based integer indices"
        raise ContractError(f"sources[{position}].columns must contain {kind}")
    label_map = source["label_map"]
    if not isinstance(label_map, dict) or not label_map:
        raise ContractError(f"sources[{position}].label_map must be explicit and non-empty")
    if any(type(value) is not int or value not in (0, 1) for value in label_map.values()):
        raise ContractError(f"sources[{position}].label_map values must be integer 0 or 1")
    logical = set(columns)
    for key in ("audio_path_template", "source"):
        unknown = _template_fields(str(source[key])) - logical
        if unknown:
            raise ContractError(f"sources[{position}].{key} references unknown columns: {sorted(unknown)}")
    audio_template = str(source["audio_path_template"])
    if _template_fields(str(Path(audio_template).parent)):
        raise ContractError(
            f"sources[{position}].audio_path_template may only substitute the filename; "
            "use an explicit source for each reviewed audio directory"
        )
    transform = source.get("utt_id_transform")
    if transform not in (None, "stem"):
        raise ContractError(f"sources[{position}].utt_id_transform must be null or 'stem'")
    group_field = source.get("source_group_field")
    if group_field is not None and group_field not in logical:
        raise ContractError(f"sources[{position}].source_group_field is not a logical column")
    _validate_output_name(source["output_name"])
    root = Path(source["root"])
    protocol = Path(source["protocol"])
    if not root.is_dir():
        raise DataError(f"dataset root is not a directory: {root}")
    if not protocol.is_file():
        raise DataError(f"protocol is not a file: {protocol}")


def validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("unified label spec schema_version must be 0.1.0")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ContractError("unified label spec requires a non-empty sources list")
    names = []
    for position, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ContractError(f"sources[{position}] must be an object")
        _validate_source(source, position)
        names.append(source["output_name"])
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ContractError(f"duplicate output_name values: {duplicates}")
    blocked = spec.get("blocked_sources", [])
    if not isinstance(blocked, list) or any(
        not isinstance(item, dict) or not all(item.get(key) for key in ("dataset_id", "reason", "evidence"))
        for item in blocked
    ):
        raise ContractError("blocked_sources must contain dataset_id, reason, and evidence")


def _iter_delimited(source: Mapping[str, Any]) -> Iterable[tuple[int, dict[str, str]]]:
    protocol = Path(source["protocol"])
    delimiter = source["delimiter"]
    columns = source["columns"]
    with protocol.open("r", encoding=source.get("encoding", "utf-8"), newline="") as stream:
        if delimiter == "whitespace":
            rows: Iterable[tuple[int, list[str]]] = (
                (line_no, line.split()) for line_no, line in enumerate(stream, 1) if line.strip()
            )
        else:
            rows = enumerate(csv.reader(stream, delimiter=delimiter), 1)
        names = None
        if source["header"]:
            try:
                _, names = next(iter(rows))
            except StopIteration as exc:
                raise DataError(f"empty protocol: {protocol}") from exc
        for line_no, row in rows:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            fields = {}
            try:
                for logical, column in columns.items():
                    if names is None:
                        fields[logical] = row[column].strip()
                    else:
                        if column not in names:
                            raise DataError(f"reviewed header column absent: {column!r} in {protocol}")
                        fields[logical] = row[names.index(column)].strip()
            except IndexError as exc:
                raise DataError(f"short row at {protocol}:{line_no}") from exc
            yield line_no, fields


def _map_label(value: str, label_map: Mapping[str, int]) -> int | None:
    if value in label_map:
        return label_map[value]
    lowered = value.lower()
    matches = {mapped for raw, mapped in label_map.items() if str(raw).lower() == lowered}
    return next(iter(matches)) if len(matches) == 1 else None


def _write_csv_header(stream: Any) -> csv.DictWriter:
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    return writer


def _quarantine_row(source: Mapping[str, Any], fields: Mapping[str, str], line_no: int,
                    protocol_hash: str, reason: str, audio_relpath: str = "") -> dict[str, str]:
    record_id = fields.get(source["record_id"], "")
    original = fields.get(source["label_field"], "")
    return {
        "utt_id": record_id,
        "path": "",
        "label": "",
        "attack": fields.get(source.get("attack_field", ""), "unknown") or "unknown",
        "speaker": fields.get(source.get("speaker_field", ""), "unknown") or "unknown",
        "split": str(source["official_split"]),
        "source": str(source["source"]),
        "sample_id": "",
        "dataset_id": str(source["dataset_id"]),
        "dataset_release": str(source["dataset_release"]),
        "root_key": str(source.get("root_key", source["dataset_id"])),
        "audio_relpath": audio_relpath,
        "original_label": original,
        "label_mapping_status": reason,
        "source_group_id": "",
        "group_quality": str(source.get("group_quality", "unknown")),
        "protocol_file": str(Path(source["protocol"]).resolve()),
        "protocol_sha256": protocol_hash,
        "protocol_line": str(line_no),
        "split_role": "quarantine",
        "status": reason,
        "schema_version": SCHEMA_VERSION,
    }


def _canonical_row(source: Mapping[str, Any], fields: Mapping[str, str], line_no: int,
                   protocol_hash: str, audio_base: Path) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    record_id = fields[source["record_id"]]
    utt_id = Path(record_id).stem if source.get("utt_id_transform") == "stem" else record_id
    original_label = fields[source["label_field"]]
    if not record_id:
        return None, _quarantine_row(source, fields, line_no, protocol_hash, "missing_record_id")
    canonical_label = _map_label(original_label, source["label_map"])
    if canonical_label is None:
        return None, _quarantine_row(source, fields, line_no, protocol_hash, "unmapped_label")
    try:
        audio_relpath = str(source["audio_path_template"]).format_map(fields)
        safe_relative(audio_relpath)
        if Path(audio_relpath).parent != Path(str(source["audio_path_template"])).parent:
            raise PermissionDenied("substitution changed the reviewed audio directory")
        filename = Path(audio_relpath).name
        if filename != audio_relpath.rsplit("/", 1)[-1] or filename in ("", ".", ".."):
            raise PermissionDenied("substituted audio filename is unsafe")
        audio = audio_base / filename
    except (KeyError, ValueError, OSError, PermissionDenied) as exc:
        row = _quarantine_row(source, fields, line_no, protocol_hash, "audio_path_rejected")
        row["label_mapping_status"] = f"audio_path_rejected:{type(exc).__name__}"
        return None, row
    if audio.is_symlink():
        return None, _quarantine_row(source, fields, line_no, protocol_hash, "audio_symlink_rejected", audio_relpath)
    if not audio.is_file():
        return None, _quarantine_row(source, fields, line_no, protocol_hash, "audio_missing", audio_relpath)
    namespace = str(source.get("record_namespace", source["output_name"]))
    stable_record_id = f"{namespace}:{record_id}"
    sample_id = opaque_sample_id(str(source["dataset_id"]), str(source["dataset_release"]), stable_record_id)
    attack = fields.get(source.get("attack_field", ""), "unknown") or "unknown"
    speaker = fields.get(source.get("speaker_field", ""), "unknown") or "unknown"
    group_field = source.get("source_group_field")
    source_group = fields.get(group_field, "") if group_field else ""
    try:
        source_name = str(source["source"]).format_map(fields)
    except KeyError as exc:
        raise ContractError(f"source template field disappeared at {source['protocol']}:{line_no}") from exc
    return {
        "utt_id": utt_id,
        "path": str(audio),
        "label": str(canonical_label),
        "attack": attack,
        "speaker": speaker,
        "split": str(source["official_split"]),
        "source": source_name,
        "sample_id": sample_id,
        "dataset_id": str(source["dataset_id"]),
        "dataset_release": str(source["dataset_release"]),
        "root_key": str(source.get("root_key", source["dataset_id"])),
        "audio_relpath": audio_relpath,
        "original_label": original_label,
        "label_mapping_status": "verified_explicit_mapping",
        "source_group_id": source_group,
        "group_quality": str(source.get("group_quality", "unknown")),
        "protocol_file": str(Path(source["protocol"]).resolve()),
        "protocol_sha256": protocol_hash,
        "protocol_line": str(line_no),
        "split_role": "unassigned",
        "status": "ready",
        "schema_version": SCHEMA_VERSION,
    }, None


def prepare_label_pack(spec: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Create a new immutable, ALLM-compatible label pack.

    Missing audio and unmapped labels are written only to ``quarantine.csv``;
    training/evaluation manifests therefore contain paths verified at build time.
    """
    validate_spec(spec)
    totals = Counter()
    manifests: list[dict[str, Any]] = []
    with AtomicDirectory(output) as temporary:
        manifest_dir = temporary / "manifests"
        manifest_dir.mkdir()
        quarantine_path = temporary / "quarantine.csv"
        database_path = temporary / ".label-index.sqlite3"
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("CREATE TABLE records(sample_id TEXT PRIMARY KEY, path TEXT NOT NULL, label INTEGER NOT NULL)")
        try:
            with quarantine_path.open("x", encoding="utf-8", newline="") as quarantine_stream:
                quarantine_writer = _write_csv_header(quarantine_stream)
                for source in spec["sources"]:
                    protocol_hash = sha256_file(source["protocol"])
                    name = _validate_output_name(source["output_name"])
                    destination = manifest_dir / name
                    source_counts = Counter()
                    root = Path(source["root"]).resolve()
                    template_parent = Path(str(source["audio_path_template"])).parent
                    audio_base = (root / template_parent).resolve()
                    if not audio_base.is_relative_to(root):
                        raise PermissionDenied(f"audio directory escapes dataset root: {audio_base}")
                    if not audio_base.is_dir():
                        raise DataError(f"reviewed audio directory does not exist: {audio_base}")
                    with destination.open("x", encoding="utf-8", newline="") as stream:
                        writer = _write_csv_header(stream)
                        for line_no, fields in _iter_delimited(source):
                            totals["source_rows"] += 1
                            row, rejected = _canonical_row(source, fields, line_no, protocol_hash, audio_base)
                            if rejected is not None:
                                quarantine_writer.writerow(rejected)
                                totals["quarantine"] += 1
                                source_counts[rejected["status"]] += 1
                                continue
                            assert row is not None
                            try:
                                database.execute(
                                    "INSERT INTO records(sample_id,path,label) VALUES(?,?,?)",
                                    (row["sample_id"], row["path"], int(row["label"])),
                                )
                            except sqlite3.IntegrityError as exc:
                                previous = database.execute(
                                    "SELECT path,label FROM records WHERE sample_id=?", (row["sample_id"],)
                                ).fetchone()
                                reason = "duplicate_record" if previous == (row["path"], int(row["label"])) else "record_conflict"
                                rejected = dict(row)
                                rejected.update(path="", label="", split_role="quarantine", status=reason,
                                                label_mapping_status=reason)
                                quarantine_writer.writerow(rejected)
                                totals["quarantine"] += 1
                                source_counts[reason] += 1
                                if reason == "record_conflict":
                                    raise DataError(f"conflicting stable sample ID: {row['sample_id']}") from exc
                                continue
                            writer.writerow(row)
                            totals["ready"] += 1
                            totals[f"label_{row['label']}"] += 1
                            source_counts["ready"] += 1
                            source_counts[f"label_{row['label']}"] += 1
                            if totals["ready"] % 10000 == 0:
                                database.commit()
                    database.commit()
                    manifests.append({
                        "dataset_id": source["dataset_id"],
                        "dataset_release": source["dataset_release"],
                        "official_split": source["official_split"],
                        "ref": f"manifests/{name}",
                        "sha256": sha256_file(destination),
                        "protocol_ref": str(Path(source["protocol"]).resolve()),
                        "protocol_sha256": protocol_hash,
                        "counts": dict(sorted(source_counts.items())),
                    })
        finally:
            database.close()
        database_path.unlink()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "READY_WITH_QUARANTINE" if totals["quarantine"] else "READY",
            "format": "eptta_unified_labels_csv_v1",
            "compatibility_columns": list(COMPATIBILITY_FIELDS),
            "canonical_labels": {"bonafide": 0, "spoof": 1},
            "manifests": manifests,
            "counts": dict(sorted(totals.items())),
            "quarantine_ref": "quarantine.csv",
            "quarantine_sha256": sha256_file(quarantine_path),
            "blocked_sources": spec.get("blocked_sources", []),
            "immutable": True,
        }
        write_json_new(temporary / "index.json", metadata)
    return metadata


def iter_unified_manifest(path: str | Path, *, require_labels: bool = True,
                          check_audio: bool = False) -> Iterable[dict[str, Any]]:
    """Stream one unified CSV without loading a million-row manifest in memory."""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(FIELDS) - set(reader.fieldnames or ()))
        if missing:
            raise DataError(f"unified manifest missing columns {missing}: {source}")
        for line_no, row in enumerate(reader, 2):
            if row["schema_version"] != SCHEMA_VERSION:
                raise DataError(f"unsupported schema version at {source}:{line_no}")
            if require_labels and row["label"] not in ("0", "1"):
                raise DataError(f"invalid canonical label at {source}:{line_no}: {row['label']!r}")
            if row["audio_relpath"]:
                safe_relative(row["audio_relpath"])
            if check_audio and not Path(row["path"]).is_file():
                raise DataError(f"audio missing at {source}:{line_no}: {row['path']}")
            value: dict[str, Any] = dict(row)
            value["canonical_label"] = int(row["label"]) if row["label"] in ("0", "1") else None
            yield value


def validate_label_pack(path: str | Path, *, check_audio: bool = False) -> dict[str, Any]:
    root = Path(path)
    index_path = root / "index.json" if root.is_dir() else root
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != SCHEMA_VERSION or index.get("format") != "eptta_unified_labels_csv_v1":
        raise DataError(f"not an EP-TTA unified label pack: {index_path}")
    pack_root = index_path.parent
    counts = Counter()
    database = sqlite3.connect("")
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("CREATE TABLE sample_ids(sample_id TEXT PRIMARY KEY)")
    try:
        for item in index.get("manifests", []):
            safe_relative(item["ref"])
            manifest_path = pack_root / item["ref"]
            if sha256_file(manifest_path) != item["sha256"]:
                raise DataError(f"manifest changed after publication: {manifest_path}")
            manifest_counts = Counter()
            for row in iter_unified_manifest(manifest_path, check_audio=check_audio):
                try:
                    database.execute("INSERT INTO sample_ids(sample_id) VALUES(?)", (row["sample_id"],))
                except sqlite3.IntegrityError as exc:
                    raise DataError(f"duplicate sample_id across manifests: {row['sample_id']}") from exc
                counts["ready"] += 1
                counts[f"label_{row['label']}"] += 1
                manifest_counts["ready"] += 1
                manifest_counts[f"label_{row['label']}"] += 1
                if counts["ready"] % 10000 == 0:
                    database.commit()
            for key in ("ready", "label_0", "label_1"):
                if manifest_counts[key] != item.get("counts", {}).get(key, 0):
                    raise DataError(
                        f"manifest index count mismatch for {item['ref']} {key}: "
                        f"expected {item.get('counts', {}).get(key, 0)}, got {manifest_counts[key]}"
                    )
        database.commit()
    finally:
        database.close()
    expected = index.get("counts", {})
    for key in ("ready", "label_0", "label_1"):
        if counts[key] != expected.get(key, 0):
            raise DataError(f"index count mismatch for {key}: expected {expected.get(key, 0)}, got {counts[key]}")
    quarantine_ref = index.get("quarantine_ref")
    if not isinstance(quarantine_ref, str):
        raise DataError("label pack index is missing quarantine_ref")
    safe_relative(quarantine_ref)
    quarantine_path = pack_root / quarantine_ref
    if sha256_file(quarantine_path) != index.get("quarantine_sha256"):
        raise DataError(f"quarantine manifest changed after publication: {quarantine_path}")
    quarantine_count = 0
    for row in iter_unified_manifest(quarantine_path, require_labels=False):
        if row["split_role"] != "quarantine" or row["status"] == "ready":
            raise DataError(f"invalid quarantine row: {row['protocol_file']}:{row['protocol_line']}")
        quarantine_count += 1
    if quarantine_count != expected.get("quarantine", 0):
        raise DataError(
            f"index count mismatch for quarantine: expected {expected.get('quarantine', 0)}, got {quarantine_count}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "VALID",
        "label_pack": str(index_path),
        "checked_audio": check_audio,
        "counts": dict(sorted(counts.items())),
        "manifest_count": len(index.get("manifests", [])),
        "quarantine_count": quarantine_count,
    }
