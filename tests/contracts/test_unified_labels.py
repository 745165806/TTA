import csv
import json

import pytest

from eptta.data.unified_labels import FIELDS, iter_unified_manifest, prepare_label_pack, validate_label_pack
from eptta.errors import DataError, EPTTAError


def spec(root, protocol):
    return {
        "schema_version": "0.1.0",
        "sources": [
            {
                "dataset_id": "fixture",
                "dataset_release": "fixture-v1",
                "output_name": "fixture_train.csv",
                "record_namespace": "train",
                "root": str(root),
                "root_key": "fixture",
                "protocol": str(protocol),
                "format": "delimited",
                "encoding": "utf-8",
                "delimiter": "whitespace",
                "header": False,
                "columns": {"speaker_id": 0, "utt_id": 1, "attack_id": 2, "raw_label": 3},
                "record_id": "utt_id",
                "audio_path_template": "audio/{utt_id}.wav",
                "label_field": "raw_label",
                "label_map": {"real": 0, "fake": 1},
                "attack_field": "attack_id",
                "speaker_field": "speaker_id",
                "source_group_field": "speaker_id",
                "group_quality": "fixture_reviewed",
                "official_split": "train",
                "source": "fixture:train:{attack_id}",
            }
        ],
    }


def build_fixture(tmp_path):
    root = tmp_path / "audio-root"
    (root / "audio").mkdir(parents=True)
    (root / "audio/u1.wav").write_bytes(b"RIFF-fixture")
    (root / "audio/u2.wav").write_bytes(b"RIFF-fixture")
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("s1 u1 - real\ns2 u2 A01 fake\ns3 missing A02 fake\ns4 u4 A03 unknown\n")
    return root, protocol


def test_unified_pack_is_allmdf_compatible_and_preserves_lineage(tmp_path):
    root, protocol = build_fixture(tmp_path)
    output = tmp_path / "labels"
    result = prepare_label_pack(spec(root, protocol), output)
    assert result["status"] == "READY_WITH_QUARANTINE"
    assert result["counts"] == {"label_0": 1, "label_1": 1, "quarantine": 2, "ready": 2, "source_rows": 4}
    manifest = output / "manifests/fixture_train.csv"
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == FIELDS
        rows = list(reader)
    assert rows[0]["utt_id"] == "u1" and rows[0]["path"].endswith("audio/u1.wav")
    assert rows[0]["label"] == "0" and rows[1]["label"] == "1"
    assert rows[0]["audio_relpath"] == "audio/u1.wav"
    assert rows[0]["original_label"] == "real"
    assert rows[0]["source_group_id"] == "s1"
    assert len(rows[0]["sample_id"]) == 66 and rows[0]["sample_id"].startswith("s-")
    with (output / "quarantine.csv").open(newline="") as stream:
        rejected = list(csv.DictReader(stream))
    assert {row["status"] for row in rejected} == {"audio_missing", "unmapped_label"}
    assert validate_label_pack(output, check_audio=True)["counts"] == {"label_0": 1, "label_1": 1, "ready": 2}


def test_unified_reader_streams_numeric_canonical_label(tmp_path):
    root, protocol = build_fixture(tmp_path)
    output = tmp_path / "labels"
    prepare_label_pack(spec(root, protocol), output)
    rows = list(iter_unified_manifest(output / "manifests/fixture_train.csv"))
    assert [row["canonical_label"] for row in rows] == [0, 1]


def test_label_pack_is_immutable_and_hash_bound(tmp_path):
    root, protocol = build_fixture(tmp_path)
    output = tmp_path / "labels"
    prepare_label_pack(spec(root, protocol), output)
    before = (output / "index.json").read_bytes()
    with pytest.raises(EPTTAError, match="overwrite"):
        prepare_label_pack(spec(root, protocol), output)
    assert (output / "index.json").read_bytes() == before
    manifest = output / "manifests/fixture_train.csv"
    manifest.write_text(manifest.read_text() + "tampered\n")
    with pytest.raises(DataError, match="changed after publication"):
        validate_label_pack(output)


def test_unified_reader_rejects_invalid_label(tmp_path):
    path = tmp_path / "bad.csv"
    row = {field: "" for field in FIELDS}
    row.update(schema_version="0.1.0", label="real", audio_relpath="audio/u.wav")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(DataError, match="invalid canonical label"):
        list(iter_unified_manifest(path))
