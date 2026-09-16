"""Strict metadata adapters for the four declared target evaluation scopes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from eptta.config.schema import check
from eptta.data.io import iter_jsonl, read_json, sha256_file, write_json_new
from eptta.data.permissions import project_target
from eptta.errors import ContractError, DataError, ResourceError


SUBSETS = frozenset({"eval", "progress", "hidden"})


def _opaque(dataset_id, release, record_id):
    value = f"{dataset_id}\0{release}\0{record_id}".encode("utf-8")
    return "s-" + hashlib.sha256(value).hexdigest()


def _asvspoof_rows(dataset_id, metadata):
    expected = 8 if dataset_id == "asvspoof2021_la" else 13
    with metadata.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            values = line.split()
            if len(values) != expected:
                raise DataError(f"{dataset_id} metadata row {number} has {len(values)} fields; expected {expected}")
            if dataset_id == "asvspoof2021_la":
                speaker, trial, codec, transmission, attack, label, trim, subset = values
                extra = {"transmission": transmission}
            else:
                speaker, trial, codec, source, attack, label, trim, subset, vocoder, task, team, gender, language = values
                extra = {"source": source, "vocoder": vocoder, "task": task, "team": team,
                         "gender": gender, "language": language}
            if subset not in SUBSETS:
                raise DataError(f"{dataset_id} metadata row {number} has invalid subset {subset!r}")
            if label not in ("bonafide", "spoof"):
                raise DataError(f"{dataset_id} metadata row {number} has invalid label {label!r}")
            yield {"record_id": trial, "speaker_id": speaker, "codec_id": codec,
                   "attack_id": attack, "raw_label": label,
                   "canonical_label": 0 if label == "bonafide" else 1,
                   "trim": trim, "subset": subset, **extra}


def _itw_rows(metadata):
    with metadata.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file", "speaker", "label"]:
            raise DataError("In-the-Wild metadata header must be exactly file,speaker,label")
        for number, row in enumerate(reader, 2):
            if row["label"] not in ("bona-fide", "spoof"):
                raise DataError(f"In-the-Wild metadata row {number} has invalid label")
            yield {"record_id": row["file"], "speaker_id": row["speaker"],
                   "raw_label": row["label"],
                   "canonical_label": 0 if row["label"] == "bona-fide" else 1,
                   "subset": "eval"}


def _lineage_summary(selected, source_snapshot):
    target_speakers = {item["sidecar"].get("speaker_id") for item in selected
                       if item["sidecar"].get("speaker_id")}
    target_speaker_samples = Counter(item["sidecar"].get("speaker_id") for item in selected
                                     if item["sidecar"].get("speaker_id"))
    target_ids = {item["sidecar"]["record_id"] for item in selected}
    root = Path(source_snapshot).resolve()
    metadata_path = root / "snapshot.json" if root.is_dir() else root
    metadata = read_json(metadata_path)
    canonical = metadata_path.parent / metadata.get("canonical_ref", "canonical.jsonl")
    if sha256_file(canonical) != metadata.get("canonical_sha256"):
        raise DataError("source snapshot canonical hash mismatch during lineage audit")
    roles = {}
    source_speaker_samples = Counter()
    source_speaker_sets = {}
    source_ids = Counter()
    for row in iter_jsonl(canonical):
        role = row["split_role"]
        source_speaker_sets.setdefault(role, set())
        if row.get("speaker_id"):
            source_speaker_sets[role].add(row["speaker_id"])
        if row.get("speaker_id") in target_speakers:
            source_speaker_samples[role] += 1
        source_record_id = Path(row["audio_relpath"]).stem
        if source_record_id in target_ids:
            source_ids[role] += 1
    for role in metadata.get("role_counts", {}):
        overlap = target_speakers.intersection(source_speaker_sets.get(role, set()))
        roles[role] = {"overlap_speaker_count": len(overlap),
                       "source_sample_count_with_overlap_speaker": source_speaker_samples[role],
                       "target_sample_count_with_overlap_speaker": sum(target_speaker_samples[s] for s in overlap),
                       "exact_record_id_count": source_ids[role]}
    return {"source_snapshot_id": metadata.get("snapshot_id"),
            "source_canonical_sha256": metadata.get("canonical_sha256"),
            "target_speaker_count": len(target_speakers),
            "by_source_role": roles,
            "original_recording_family_relation": "UNKNOWN_NO_REVIEWED_CROSS_DATASET_MAPPING"}


def audit_target_metadata(dataset_id, metadata_ref, audio_dir, release, subset, output=None,
                          source_snapshot=None):
    """Parse authoritative metadata and propose a scope; never publish a snapshot."""
    if dataset_id not in ("asvspoof2021_la", "asvspoof2021_df", "in_the_wild"):
        raise ContractError("unsupported principal target adapter")
    if subset != "eval":
        raise ContractError("principal target scope must be exactly subset=eval")
    metadata = Path(metadata_ref).resolve()
    audio = Path(audio_dir).resolve()
    if not metadata.is_file():
        raise ResourceError(f"target metadata is missing: {metadata}")
    if not audio.is_dir():
        raise ResourceError(f"target audio directory is missing: {audio}")
    rows = _itw_rows(metadata) if dataset_id == "in_the_wild" else _asvspoof_rows(dataset_id, metadata)
    selected = []
    all_subsets = Counter()
    seen = set()
    labels = Counter()
    missing_audio = []
    extension = ".wav" if dataset_id == "in_the_wild" else ".flac"
    for row in rows:
        all_subsets[row["subset"]] += 1
        if row["subset"] != subset:
            continue
        record_id = row["record_id"]
        if record_id in seen:
            raise DataError(f"duplicate target record_id: {record_id}")
        seen.add(record_id)
        labels[row["canonical_label"]] += 1
        filename = record_id if record_id.endswith(extension) else record_id + extension
        path = audio / filename
        if not path.is_file() and len(missing_audio) < 20:
            missing_audio.append(filename)
        trusted = {"schema_version": "0.1.0", "sample_id": _opaque(dataset_id, release, record_id),
                   "root_key": dataset_id, "audio_relpath": filename, "input_sha256": None,
                   "decode_profile_id": "UNBOUND", "probe_profile_id": "UNBOUND",
                   **row}
        public = project_target(trusted)
        selected.append({"sidecar": row, "runtime": public.__dict__ if hasattr(public, "__dict__") else
                         {name: getattr(public, name) for name in public.__slots__}})
    if not selected or set(labels) != {0, 1}:
        raise DataError("selected target scope is empty or lacks both label classes")
    lineage = (_lineage_summary(selected, source_snapshot) if source_snapshot else
               {"status": "UNKNOWN_NO_SOURCE_SNAPSHOT_BOUND"})
    report = {
        "schema_version": "0.1.0", "status": "PROPOSED", "publishable": False,
        "approval_required": True, "dataset_id": dataset_id, "dataset_release": release,
        "metadata_ref": str(metadata), "metadata_sha256": sha256_file(metadata),
        "audio_dir": str(audio), "selection_rule": "subset == 'eval'",
        "selected_subset": subset, "available_subset_counts": dict(sorted(all_subsets.items())),
        "selected_count": len(selected), "label_counts": {str(k): labels[k] for k in sorted(labels)},
        "unique_record_ids": len(seen), "missing_audio_count": sum(1 for item in selected if not (audio / item["runtime"]["audio_relpath"]).is_file()),
        "missing_audio_examples": missing_audio,
        "runtime_field_whitelist": ["schema_version", "sample_id", "root_key", "audio_relpath",
                                    "input_sha256", "decode_profile_id", "probe_profile_id"],
        "sidecar_separation_verified": all(set(item["runtime"]).isdisjoint({"canonical_label", "raw_label", "attack_id", "speaker_id"}) for item in selected),
        "lineage_audit": lineage,
        "lineage_to_exposed_2019": (lineage.get("by_source_role", {}).get("control_test")
                                     if source_snapshot else "UNKNOWN"),
        "lineage_to_source_development": ({role: value for role, value in lineage.get("by_source_role", {}).items()
                                           if role != "control_test"} if source_snapshot else "UNKNOWN"),
        "effect_access_history": "UNKNOWN_WITHOUT_APPEND_ONLY_LEDGER",
        "content_sha256": hashlib.sha256(json.dumps(
            {"dataset_id": dataset_id, "release": release, "metadata_sha256": sha256_file(metadata),
             "subset": subset, "ids": sorted(seen)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    check(report, "target_audit")
    if output:
        write_json_new(output, report)
    return report
