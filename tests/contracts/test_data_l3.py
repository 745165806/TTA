import json
from pathlib import Path

import pytest

from eptta.config.schema import check
from eptta.config.validate import content_hash
from eptta.data.adapters import DelimitedAdapter, JsonRecordsAdapter
from eptta.data.approval import approve
from eptta.data.incremental import build_delta, commit_delta, refresh_plan
from eptta.data.inventory import inspect_selected
from eptta.data.io import iter_jsonl, read_json
from eptta.data.manifests import build_manifests
from eptta.data.preprocess import cache_identity, require_cache_preprocess
from eptta.data.splits import propose_splits
from eptta.data.staging import stage_dataset
from eptta.errors import ContractError, DataError, EPTTAError


def review():
    return {"accepted": True, "reviewer": "fixture-reviewer", "approved_at": "2026-09-14T00:00:00Z",
            "report_ref": "fixture-review.md", "sample_evidence_ref": "fixture-samples.json"}


def lock(kind, payload):
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval": None, "payload": payload}
    return approve(proposal, review(), kind)


def raw_payload(pattern="protocol.txt", missing_policy="error"):
    return {"format": "delimited", "dataset_release": "fixture-release", "encoding": "utf-8", "delimiter": "whitespace", "header": False,
            "columns": {"speaker_id": 0, "record_id": 1, "raw_label": 2}, "json_paths": None,
            "record_id": "record_id", "audio_path_rule": "protocol_context_template",
            "label_field": "raw_label", "allowed_values": ["real", "fake", "unknown"],
            "missing_policy": missing_policy, "protocol_globs": [pattern],
            "protocol_contexts": {pattern: {"root_key": "fixture", "official_split": "train",
                                                    "audio_path_template": "audio/{record_id}.wav"}}}


def label_payload(unknown="quarantine"):
    return {"policy_id": "fixture", "raw_to_canonical": {"real": 0, "fake": 1}, "unknown_policy": unknown}


def group_payload():
    return {"resolver": "field:speaker_id", "source_mapping_ref": "fixture-reviewed-field",
            "source_mapping_sha256": None,
            "group_quality": "fixture_reviewed_group"}


def fixture_inventory(tmp_path, lines):
    root = tmp_path / "root"
    protocol_dir = tmp_path / "protocols"
    (root / "audio").mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(exist_ok=True)
    protocol = protocol_dir / "protocol.txt"
    protocol.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        record_id = line.split()[1]
        (root / "audio" / f"{record_id}.wav").write_bytes(b"RIFF-fixture")
    inventory = inspect_selected(["fixture"], {"fixture": str(root)}, {"fixture": str(protocol_dir)})
    return inventory


def stage_fixture(tmp_path, lines, name="stage", missing_policy="error"):
    inventory = fixture_inventory(tmp_path, lines)
    output = tmp_path / name
    result = stage_dataset(inventory, lock("raw", raw_payload(missing_policy=missing_policy)),
                           lock("label", label_payload()), lock("group", group_payload()), output)
    return output, result


def test_d02_delimited_csv_tsv_whitespace_and_sidecar(tmp_path):
    cases = [("comma.csv", ",", True, "speaker,utt,label\nspk,u1,real\n", {"speaker_id": "speaker", "record_id": "utt", "raw_label": "label"}),
             ("tab.tsv", "\t", False, "spk\tu1\treal\n", {"speaker_id": 0, "record_id": 1, "raw_label": 2}),
             ("space.txt", "whitespace", False, "spk u1 real\n", {"speaker_id": 0, "record_id": 1, "raw_label": 2})]
    for name, delimiter, header, text, columns in cases:
        path = tmp_path / name
        path.write_text(text)
        payload = raw_payload(name)
        payload.update(delimiter=delimiter, header=header, columns=columns)
        rows = list(DelimitedAdapter().iter_file(path, payload))
        assert rows[0].fields == {"speaker_id": "spk", "record_id": "u1", "raw_label": "real"}
    sidecar = raw_payload("space.txt")
    sidecar["format"] = "sidecar"
    check(lock("raw", sidecar), "raw")


def test_d02_json_and_jsonl_explicit_paths(tmp_path):
    for fmt, name, text in (("json", "records.json", '[{"meta":{"speaker":"s"},"id":"u","label":"fake"}]'),
                            ("jsonl", "records.jsonl", '{"meta":{"speaker":"s"},"id":"u","label":"fake"}\n')):
        path = tmp_path / name
        path.write_text(text)
        payload = raw_payload(name)
        payload.update(format=fmt, delimiter=None, header=None, columns=None,
                       json_paths={"speaker_id": "meta.speaker", "record_id": "id", "raw_label": "label"})
        assert list(JsonRecordsAdapter().iter_file(path, payload))[0].fields["raw_label"] == "fake"


def test_d03_unknown_and_conflicting_labels_are_quarantined(tmp_path):
    stage, result = stage_fixture(tmp_path, ["s u1 real", "s u2 unknown"], missing_policy="quarantine")
    assert result["status"] == "STAGED"
    rows = list(iter_jsonl(stage / "records.jsonl"))
    assert rows[0]["canonical_label"] == 0 and rows[0]["split_role"] == "unassigned"
    assert rows[1]["canonical_label"] is None and rows[1]["split_role"] == "quarantine"
    assert list(iter_jsonl(stage / "parse_errors.jsonl"))[0]["code"] == "UNMAPPED_LABEL"


def test_reviewed_cross_dataset_group_mapping_is_hash_bound(tmp_path):
    inventory = fixture_inventory(tmp_path, ["local-speaker u1 real"])
    mapping = tmp_path / "groups.json"
    mapping.write_text(json.dumps({"local-speaker": "global-source-001"}))
    from eptta.data.io import sha256_file
    group = group_payload()
    group.update(resolver="mapping:speaker_id", source_mapping_ref=str(mapping),
                 source_mapping_sha256=sha256_file(mapping), group_quality="reviewed_cross_dataset_mapping")
    output = tmp_path / "mapped-stage"
    stage_dataset(inventory, lock("raw", raw_payload()), lock("label", label_payload()),
                  lock("group", group), output)
    assert next(iter_jsonl(output / "records.jsonl"))["source_group_id"] == "global-source-001"
    mapping.write_text(json.dumps({"local-speaker": "changed-after-review"}))
    with pytest.raises(DataError, match="changed after review"):
        stage_dataset(inventory, lock("raw", raw_payload()), lock("label", label_payload()),
                      lock("group", group), tmp_path / "rejected-stage")


def publish_snapshot(tmp_path, lines=None):
    lines = lines or ["s1 u1 real", "s2 u2 fake", "s3 u3 real", "s4 u4 fake"]
    stage, _ = stage_fixture(tmp_path, lines)
    proposal_path = tmp_path / "split.proposal.json"
    proposal = propose_splits(stage, {"payload": {"ratios": {"fit": 0.5, "source_val": 0.5}, "seed": 13}}, proposal_path)
    split_lock = approve(proposal, review(), "split")
    split_path = tmp_path / "split.lock.json"
    split_path.write_text(json.dumps(split_lock))
    snapshot = tmp_path / "snapshot"
    result = build_manifests(stage, split_path, snapshot)
    return snapshot, result


def test_split_review_and_immutable_manifests(tmp_path):
    stage, _ = stage_fixture(tmp_path, ["s1 u1 real", "s1 u2 fake", "s2 u3 real"])
    proposal_path = tmp_path / "split.json"
    proposal = propose_splits(stage, {"payload": {"ratios": {"fit": 0.5, "source_val": 0.5}, "seed": 7}}, proposal_path)
    with pytest.raises(ContractError):
        build_manifests(stage, proposal_path, tmp_path / "unreviewed")
    locked = approve(proposal, review(), "split")
    lock_path = tmp_path / "split.lock.json"
    lock_path.write_text(json.dumps(locked))
    result = build_manifests(stage, lock_path, tmp_path / "published")
    assert result["status"] == "LOCKED"
    records = list(iter_jsonl(tmp_path / "published/canonical.jsonl"))
    assert len({r["split_role"] for r in records if r["source_group_id"] == "s1"}) == 1
    for path in (tmp_path / "published/manifests").glob("*.jsonl"):
        for row in iter_jsonl(path):
            if row["split_role"] not in ("fit", "source_val"):
                assert "canonical_label" not in row


def test_d05_incremental_idempotency_and_d07_correction(tmp_path):
    parent, _ = publish_snapshot(tmp_path / "base")
    same_stage, _ = stage_fixture(tmp_path / "same", ["s1 u1 real", "s2 u2 fake", "s3 u3 real", "s4 u4 fake"])
    same_delta = tmp_path / "same-delta"
    result = build_delta(parent, same_stage, same_delta)
    assert result["counts"] == {"unchanged": 4}
    changed_stage, _ = stage_fixture(tmp_path / "changed", ["s1 u1 fake", "s5 u5 real"])
    changed_delta = tmp_path / "changed-delta"
    result = build_delta(parent, changed_stage, changed_delta)
    assert result["counts"] == {"added": 1, "relabeled": 1}
    child = tmp_path / "child"
    child_result = commit_delta(changed_delta, review(), child)
    parent_u1 = next(r for r in iter_jsonl(parent / "canonical.jsonl") if r["audio_relpath"].endswith("u1.wav"))
    child_u1 = next(r for r in iter_jsonl(child / "canonical.jsonl") if r["audio_relpath"].endswith("u1.wav"))
    assert parent_u1["canonical_label"] == child_u1["canonical_label"] == 0
    assert child_result["parent_snapshot_id"] == read_json(parent / "snapshot.json")["snapshot_id"]


def test_d06_new_lineage_connecting_cross_role_groups_is_quarantined(tmp_path):
    parent, _ = publish_snapshot(tmp_path / "base")
    parent_records = list(iter_jsonl(parent / "canonical.jsonl"))
    first = parent_records[0]
    other = next(record for record in parent_records if record["split_role"] != first["split_role"])
    candidate, _ = stage_fixture(tmp_path / "candidate", ["new u9 real"])
    candidate_records_path = candidate / "records.jsonl"
    candidate_record = next(iter_jsonl(candidate_records_path))
    candidate_record["source_group_id"] = first["source_group_id"]
    candidate_record["parent_id"] = other["sample_id"]
    candidate_records_path.write_text(json.dumps(candidate_record) + "\n")
    metadata = read_json(candidate / "staging.json")
    from eptta.data.io import sha256_file
    metadata["records_sha256"] = sha256_file(candidate_records_path)
    (candidate / "staging.json").write_text(json.dumps(metadata))
    result = build_delta(parent, candidate, tmp_path / "delta")
    assert result["counts"] == {"lineage_conflict": 1}
    row = next(iter_jsonl(tmp_path / "delta/delta.jsonl"))
    assert row["candidate"]["split_role"] == "quarantine"
    assert set(row["related_roles"].values()) == {first["split_role"], other["split_role"]}


def test_d08_output_is_atomic_and_never_overwritten(tmp_path):
    stage, _ = stage_fixture(tmp_path, ["s u real"])
    before = (stage / "staging.json").read_bytes()
    with pytest.raises(EPTTAError, match="overwrite"):
        stage_fixture(tmp_path, ["s u real"])
    assert (stage / "staging.json").read_bytes() == before


def test_d09_refresh_plan_distinguishes_append_from_source_dependencies():
    delta = {"counts": {"added": 2}}
    artifacts = {"artifacts": [{"artifact_id": "model", "kind": "frozen_model", "depends_on_fit": True},
                                {"artifact_id": "target", "kind": "target_features", "depends_on_fit": False}]}
    result = refresh_plan(delta, artifacts)
    assert result["artifacts"] == [{"artifact_id": "model", "action": "reuse"},
                                    {"artifact_id": "target", "action": "append_new_inputs"}]
    changed = refresh_plan({"counts": {}, "checkpoint_changed": True},
                           {"artifacts": [{"artifact_id": "features", "kind": "target_features"},
                                          {"artifact_id": "audio", "kind": "raw_audio"}]})
    assert changed["artifacts"] == [{"artifact_id": "features", "action": "rebuild_new_checkpoint_identity"},
                                     {"artifact_id": "audio", "action": "reuse"}]
    by_hash = refresh_plan({"counts": {}}, {"current_dependency_hashes": {"checkpoint": "b" * 64},
        "artifacts": [{"artifact_id": "features", "kind": "target_features",
                       "dependency_hashes": {"checkpoint": "a" * 64}}]})
    assert by_hash["artifacts"] == [{"artifact_id": "features", "action": "rebuild_dependency_hash_changed",
                                     "changed_dependencies": ["checkpoint"]}]


def preprocess(value):
    payload = {"decode": f"decode-{value}", "train_unit": f"train-{value}", "eval_unit": f"eval-{value}",
               "source_probe": f"source-{value}", "target_probe": f"target-{value}",
               "quality_policy": "quarantine_corrupt"}
    return lock("preprocess", payload)


def test_d10_preprocess_is_part_of_cache_identity_and_old_cache_is_rejected():
    first, second = preprocess("a"), preprocess("b")
    first_id = cache_identity("a" * 64, "b" * 64, first, "probe", "float32", "deterministic")
    second_id = cache_identity("a" * 64, "b" * 64, second, "probe", "float32", "deterministic")
    assert first_id != second_id
    with pytest.raises(DataError, match="mismatch"):
        require_cache_preprocess({"preprocess_sha256": first["approval"]["content_sha256"]}, second)
