import json
from pathlib import Path

import pytest

from eptta import __version__, SCHEMA_VERSION
from eptta.cli import main
from eptta.config.schema import read_document
from eptta.errors import EPTTAError
from eptta.research import COMMANDS, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_versions():
    assert __version__ == SCHEMA_VERSION == "0.1.0"
    assert 'version = "0.1.0"' in (ROOT / "pyproject.toml").read_text()


@pytest.mark.parametrize("cmd", [None, *COMMANDS])
def test_six_daily_entries_have_help(cmd):
    with pytest.raises(SystemExit) as exc:
        main(([cmd] if cmd else []) + ["--help"])
    assert exc.value.code == 0


def test_old_approval_commands_are_not_exposed():
    choices = main.__globals__["parser"]()._subparsers._group_actions[0].choices
    assert set(choices) == set(COMMANDS)
    assert not {"approve-contract", "seal-scores", "lock-r5-stage1-proposal"} & set(choices)


def test_prepare_data_reuses_assignments_and_writes_label_free_target(tmp_path, capsys):
    manifest = tmp_path / "input.csv"
    manifest.write_text(
        "id,path,label,group,role\n"
        "a,a.wav,bonafide,s1,fit\n"
        "b,b.wav,spoof,s2,target_test\n",
        encoding="utf-8")
    config = tmp_path / "prepare.yaml"
    output = tmp_path / "prepared"
    config.write_text(json.dumps({
        "schema_version": "0.1.0", "command": "prepare-data",
        "dataset": {"dataset_id": "fixture", "release": "r1", "subset": "eval",
                    "manifest": str(manifest)},
        "columns": {"sample_id": "id", "audio_relpath": "path", "label": "label",
                    "group_id": "group", "split_role": "role"},
        "label_map": {"bonafide": 0, "spoof": 1},
        "split": {"reuse_existing": True, "seed": 13}, "output": str(output)}))
    assert main(["prepare-data", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "READY"
    target = (output / "manifests" / "target_test.jsonl").read_text()
    assert "canonical_label" not in target and "spoof" not in target


def test_prepare_data_rejects_unknown_fields(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text(json.dumps({"schema_version": "0.1.0", "command": "prepare-data",
                                  "mystery": True}))
    assert main(["prepare-data", "--config", str(config)]) == 2
    assert "unknown top-level" in json.loads(capsys.readouterr().out)["message"]


def test_config_reader_rejects_duplicate_keys_and_nonfinite_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"command":"prepare-data","command":"run-tta"}')
    with pytest.raises(EPTTAError, match="duplicate key"):
        read_document(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"threshold":NaN}')
    with pytest.raises(EPTTAError, match="non-finite"):
        read_document(nonfinite)


def test_prepare_data_rejects_unknown_nested_field(tmp_path, capsys):
    config = tmp_path / "nested.yaml"
    config.write_text(json.dumps({
        "schema_version": "0.1.0", "command": "prepare-data",
        "dataset": {"dataset_id": "fixture", "release": "r1", "subset": "train",
                    "manifest": "/missing", "guessed_label": True},
        "columns": {"sample_id": "id", "audio_relpath": "path", "label": "label",
                    "group_id": "group"},
        "label_map": {"bonafide": 0, "spoof": 1},
        "split": {"reuse_existing": True, "seed": 13}, "output": str(tmp_path / "out")}))
    assert main(["prepare-data", "--config", str(config)]) == 2
    assert "unknown=['guessed_label']" in json.loads(capsys.readouterr().out)["message"]


def test_prepare_data_rejects_unfilled_semantic_placeholder(tmp_path, capsys):
    config = tmp_path / "placeholder.yaml"
    config.write_text(json.dumps({
        "schema_version": "0.1.0", "command": "prepare-data",
        "dataset": {"dataset_id": "fixture", "release": "r1",
                    "subset": "REPLACE_WITH_EXPLICIT_SCOPE", "manifest": "/missing"},
        "columns": {"sample_id": "id", "audio_relpath": "path", "label": "label",
                    "group_id": "group"},
        "label_map": {"bonafide": 0, "spoof": 1},
        "split": {"reuse_existing": True, "seed": 13}, "output": str(tmp_path / "out")}))
    assert main(["prepare-data", "--config", str(config)]) == 2
    assert "replace its REPLACE_WITH_ placeholder" in json.loads(capsys.readouterr().out)["message"]


def test_local_path_map_is_bound_but_not_persisted(tmp_path):
    config = tmp_path / "config.json"
    paths = tmp_path / "paths.json"
    config.write_text(json.dumps({
        "schema_version": "0.1.0", "command": "train-source",
        "recipe": "@path:recipe", "phase": "smoke", "output": "out"}))
    paths.write_text(json.dumps({"schema_version": "0.1.0", "paths": {
        "recipe": "/private/recipe.json", "unrelated_secret_path": "/private/secret"}}))
    resolved = load_config(config, "train-source", paths)
    assert resolved["recipe"] == "/private/recipe.json"
    assert "paths" not in resolved and "unrelated_secret_path" not in str(resolved)
