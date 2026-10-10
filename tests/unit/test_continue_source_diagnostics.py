"""One-shot diagnostic runner command and terminal-event checks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/meta_audio_tta"))
from continue_source_diagnostics import (_stage2_stopped, diagnostic_command,
                                         validate_prior_ce)


def test_command_uses_current_python_and_fixed_source_only_config(tmp_path):
    command = diagnostic_command("joint", tmp_path / "joint", Path("stage1"),
                                 Path("stage2"), Path("stage1/gate.json"), "cuda:2")
    assert command[0] == sys.executable
    assert command[1].endswith("source_gpu_diagnostics.py")
    assert "--variant" in command and command[command.index("--variant") + 1] == "joint"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert not any("target" in part or "label" in part for part in command)


def test_existing_stage2_stop_is_detected_without_restart(tmp_path):
    events = tmp_path / "launch.jsonl"
    events.write_text(json.dumps({"event": "WAIT_STAGE1"}) + "\n")
    assert not _stage2_stopped(events)
    with events.open("a") as stream:
        stream.write(json.dumps({"event": "EXIT", "exit_code": 2}) + "\n")
    assert _stage2_stopped(events)


def test_resuming_after_ce_requires_matching_completed_frozen_result(tmp_path):
    gate = {"ce": {"checkpoint": "selected-ce.pt"}}
    path = tmp_path / "summary.json"
    prior = {"status": "PASS", "variant": "ce", "checkpoint": "selected-ce.pt",
             "source_val": {"count": 64, "k1": "NOT_RUN: no trained BYOL head"},
             "checkpoint_state_unchanged": True}
    path.write_text(json.dumps(prior))
    validate_prior_ce(path, gate)
    prior["checkpoint"] = "other.pt"
    path.write_text(json.dumps(prior))
    try:
        validate_prior_ce(path, gate)
    except ValueError as error:
        assert "mismatched" in str(error)
    else:
        raise AssertionError("mismatched CE checkpoint was accepted")
