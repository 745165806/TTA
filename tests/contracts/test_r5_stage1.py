import json

import pytest

from eptta.execution.r5_proposal import _select_role, _unlabeled


def _fixture(tmp_path, missing_attack=False):
    rows, metadata = [], {}
    entries = []
    for index in range(64):
        entries.append((0, "-", "speaker-%02d" % (index % 8)))
    for attack in range(1, 7):
        for index in range(11):
            entries.append((1, None if missing_attack and attack == 6 else "A%02d" % attack,
                            "speaker-%02d" % (index % 5)))
    for index, (label, attack, speaker) in enumerate(entries):
        sample_id = "sample-%03d" % index
        path = tmp_path / (sample_id + ".wav")
        path.write_bytes(b"x" * (index + 1))
        rows.append({"schema_version": "0.1.0", "sample_id": sample_id, "root_key": "root",
                     "audio_relpath": path.name, "input_sha256": None, "split_role": "cal0"})
        metadata[sample_id] = {"sample_id": sample_id, "split_role": "cal0",
                               "canonical_label": label, "generator_id": attack,
                               "speaker_id": speaker}
    return rows, metadata


def test_stage1_selection_is_balanced_six_attack_and_unlabeled(tmp_path):
    rows, metadata = _fixture(tmp_path)
    selected = _select_role("cal0", rows, metadata, {"root": str(tmp_path)})
    assert len(selected) == 128
    assert sum(row["canonical_label"] == 0 for row in selected) == 64
    assert sum(row["canonical_label"] == 1 for row in selected) == 64
    assert sorted({row["attack_id"] for row in selected if row["canonical_label"] == 1}) == [
        "A01", "A02", "A03", "A04", "A05", "A06"]
    stripped = _unlabeled(selected[0])
    assert set(stripped) == {"schema_version", "sample_id", "root_key", "audio_relpath",
                             "input_sha256", "split_role"}
    assert "canonical_label" not in json.dumps(stripped)


def test_stage1_selection_rejects_missing_real_attack_field(tmp_path):
    rows, metadata = _fixture(tmp_path, missing_attack=True)
    with pytest.raises(Exception, match="metadata is missing"):
        _select_role("cal0", rows, metadata, {"root": str(tmp_path)})
