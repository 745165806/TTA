"""One-shot source-only continuation after the two fixed stage-1 jobs.

Run inside the tta conda environment. This script never accesses target
assignments or labels and never starts stage-3 scoring.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _event(path, kind, **details):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                 "event": kind, **details}, allow_nan=False) + "\n")


def _run_one(command, log_path, events, name):
    _event(events, "START", name=name, command=command, log=str(log_path))
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                check=False)
    _event(events, "EXIT", name=name, exit_code=result.returncode)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=10.0)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "tta":
        raise RuntimeError("source continuation must use conda environment tta")
    if args.stage2_root.exists():
        raise FileExistsError("stage-2 run root already exists")
    args.stage2_root.mkdir(parents=True, exist_ok=False)
    events = args.stage2_root / "launch.jsonl"
    events.touch(exist_ok=False)
    selections = [args.stage1_root / mode / "selection.json" for mode in ("ce", "joint")]
    _event(events, "WAIT_STAGE1", selections=[str(path) for path in selections],
           deadline_hours=args.wait_hours)
    deadline = time.monotonic() + args.wait_hours * 3600
    while not all(path.is_file() for path in selections):
        if time.monotonic() >= deadline:
            _event(events, "STOP", reason="stage-1 selections incomplete at deadline")
            raise TimeoutError("stage-1 fixed-budget jobs did not complete")
        time.sleep(30)
    _event(events, "STAGE1_SELECTIONS_PRESENT")
    base = Path("experiments/meta_audio_tta")
    gate = args.stage1_root / "gate.json"
    gate_command = [sys.executable, str(base / "gate_stage1.py"),
                    "--config", str(base / "config_stage1_20261008.json"),
                    "--root", str(args.stage1_root), "--output", str(gate),
                    "--device", "cuda:0"]
    _run_one(gate_command, args.stage2_root / "gate_stage1.log", events, "stage1_real_model_gate")
    gate_record = json.loads(gate.read_text(encoding="utf-8"))
    if gate_record.get("status") != "PASS":
        _event(events, "STOP", reason="stage-1 real-model gate did not pass")
        raise ValueError("stage-1 real-model gate did not pass")
    _event(events, "STAGE1_GATE_PASS", ce=gate_record["ce"],
           joint=gate_record["joint"], backbone_gradient_l1=gate_record["backbone_gradient_l1"],
           score_change=gate_record["score_change"])
    runs = {}
    for variant, device in (("cross", "cuda:0"), ("same", "cuda:1")):
        command = [sys.executable, str(base / "stage2_meta.py"),
                   "--config", str(base / "config_stage2_20261008.json"),
                   "--source-config", str(base / "config_stage1_20261008.json"),
                   "--source-selection", str(args.stage1_root / "joint" / "selection.json"),
                   "--stage1-gate", str(gate), "--variant", variant,
                   "--output", str(args.stage2_root / variant), "--device", device]
        log_path = args.stage2_root / f"{variant}.log"
        stream = log_path.open("x", encoding="utf-8")
        _event(events, "START", name=variant, command=command, log=str(log_path))
        runs[variant] = (subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT),
                         stream)
    results = {}
    for variant, (process, stream) in runs.items():
        results[variant] = process.wait()
        stream.close()
        _event(events, "EXIT", name=variant, exit_code=results[variant])
    if any(results.values()):
        _event(events, "STOP", reason="stage-2 variant failed", exit_codes=results)
        raise RuntimeError("stage-2 source-only meta run failed")
    for variant in ("cross", "same"):
        selection = args.stage2_root / variant / "selection.json"
        if not selection.is_file():
            _event(events, "STOP", reason=variant + " selection missing")
            raise FileNotFoundError(selection)
    _event(events, "STAGE2_COMPLETE", variants=list(results))


if __name__ == "__main__":
    main()
