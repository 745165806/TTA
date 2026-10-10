"""One-shot, bounded source-only diagnostics after the existing stage gates."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("experiments/meta_audio_tta")


def _event(path, name, **details):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                 "event": name, **details}, allow_nan=False) + "\n")


def diagnostic_command(variant, output, stage1_root, stage2_root, gate, device):
    return [sys.executable, str(BASE / "source_gpu_diagnostics.py"),
            "--config", str(BASE / "config_source_gpu_diagnostics_20261008.json"),
            "--source-config", str(BASE / "config_stage1_20261008.json"),
            "--stage1-root", str(stage1_root), "--stage2-root", str(stage2_root),
            "--stage1-gate", str(gate), "--variant", variant, "--device", device,
            "--output", str(output)]


def validate_prior_ce(summary_path, gate):
    if summary_path is None:
        raise ValueError("resuming after CE requires its completed summary")
    prior = json.loads(summary_path.read_text(encoding="utf-8"))
    if (prior.get("status") != "PASS" or prior.get("variant") != "ce" or
            prior.get("checkpoint") != gate["ce"]["checkpoint"] or
            prior.get("source_val", {}).get("count") != 64 or
            not str(prior["source_val"].get("k1", "")).startswith("NOT_RUN") or
            prior.get("checkpoint_state_unchanged") is not True):
        raise ValueError("prior CE Frozen diagnostic is incomplete or mismatched")


def _stage2_stopped(stage2_events):
    if not stage2_events.exists():
        return False
    with stage2_events.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return any(row["event"] == "STOP" or
               (row["event"] == "EXIT" and row.get("exit_code", 0) != 0)
               for row in rows)


def _wait_for(paths, deadline, events, stage2_events=None):
    while not all(path.is_file() for path in paths):
        if stage2_events is not None and _stage2_stopped(stage2_events):
            _event(events, "BLOCKED", reason="existing stage-2 runner stopped")
            return False
        if time.monotonic() >= deadline:
            _event(events, "WAITING_TIMEOUT", missing=[str(path) for path in paths],
                   meaning="operational wait only; no scientific conclusion")
            return False
        time.sleep(30)
    return True


def _run_variant(variant, args, events):
    output = args.output_root / variant
    command = diagnostic_command(variant, output, args.stage1_root, args.stage2_root,
                                 args.stage1_gate, args.device)
    log_path = args.output_root / f"{variant}.log"
    _event(events, "START", variant=variant, command=command, log=str(log_path))
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                check=False)
    _event(events, "EXIT", variant=variant, exit_code=result.returncode,
           summary_exists=(output / "summary.json").is_file())
    if result.returncode or not (output / "summary.json").is_file():
        _event(events, "BLOCKED", reason=variant + " diagnostic failed or resource gate refused")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--stage1-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), required=True)
    parser.add_argument("--wait-hours", type=float, default=8.0)
    parser.add_argument("--resume-after-ce", action="store_true")
    parser.add_argument("--prior-ce-summary", type=Path)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "tta":
        raise RuntimeError("diagnostic continuation requires conda environment tta")
    if args.output_root.exists():
        raise FileExistsError("refusing to overwrite diagnostic run root")
    args.output_root.mkdir(parents=True, exist_ok=False)
    events = args.output_root / "launch.jsonl"
    events.touch(exist_ok=False)
    deadline = time.monotonic() + args.wait_hours * 3600
    _event(events, "WAIT_STAGE1_GATE", gate=str(args.stage1_gate),
           stage2_root=str(args.stage2_root), device=args.device,
           wait_hours=args.wait_hours)
    if not _wait_for([args.stage1_gate], deadline, events,
                     args.stage2_root / "launch.jsonl"):
        return
    gate = json.loads(args.stage1_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        _event(events, "BLOCKED", reason="stage-1 real-model gate did not pass")
        return
    if args.resume_after_ce:
        validate_prior_ce(args.prior_ce_summary, gate)
        _event(events, "REUSE_COMPLETED_CE", summary=str(args.prior_ce_summary))
    elif args.prior_ce_summary is not None:
        raise ValueError("prior CE summary requires --resume-after-ce")
    for variant in (("joint",) if args.resume_after_ce else ("ce", "joint")):
        if not _run_variant(variant, args, events):
            return
    _event(events, "WAIT_STAGE2_SELECTIONS")
    selections = [args.stage2_root / variant / "selection.json"
                  for variant in ("cross", "same")]
    if not _wait_for(selections, deadline, events, args.stage2_root / "launch.jsonl"):
        return
    for variant in ("cross", "same"):
        if not _run_variant(variant, args, events):
            return
    _event(events, "COMPLETE", variants=["ce", "joint", "cross", "same"])


if __name__ == "__main__":
    main()
