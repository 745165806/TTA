import json
from pathlib import Path

import pytest

from eptta import __version__, SCHEMA_VERSION
from eptta.cli import STUBS, main
from eptta.registry import get_spec
from eptta.errors import EPTTAError

ROOT = Path(__file__).resolve().parents[2]


def test_versions():
    assert __version__ == SCHEMA_VERSION == "0.1.0"
    assert 'version = "0.1.0"' in (ROOT / "pyproject.toml").read_text()


@pytest.mark.parametrize("cmd", [None, *STUBS, "validate", "plan"])
def test_all_help(cmd):
    with pytest.raises(SystemExit) as exc:
        main(([cmd] if cmd else []) + ["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("cmd", list(STUBS))
def test_real_commands_never_succeed(cmd, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main([cmd]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["code"] == "NOT_IMPLEMENTED_STAGE"
    assert result["execution_ready"] is False


def test_plan_is_preview_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    dest = tmp_path / "plan.json"
    args = ["plan", "--experiment", "configs/experiments/source_pilot.yaml", "--task", "source_prepare", "--out", str(dest), "--dry-run"]
    assert main(args) == 0
    original = dest.read_bytes()
    data = json.loads(original)
    assert data["status"] == "PREVIEW_ONLY" and not data["execution_ready"]
    assert all(n["status"] == "NOT_RUN" for n in data["nodes"])
    assert main(args) == 2
    assert dest.read_bytes() == original


def test_unknown_method_and_registered_todo():
    with pytest.raises(EPTTAError):
        get_spec("methods", "made_up")
    assert get_spec("methods", "tent_audio_ep")["implementation_status"] == "TODO"
    assert get_spec("models", "ssl_aasist_source")["source_training_required"] is True
