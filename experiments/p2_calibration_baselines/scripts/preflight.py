#!/usr/bin/env python
"""P2 preflight (also powers --dry-run). Checks only; never adapts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args()

    checks = {}
    import torch
    checks["runtime"] = {
        "calibration_path": "cpu",
        "waveform_ports_path": "cuda",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.worker_count:
        raise SystemExit("need %d CUDA GPUs for waveform ports, found %d" %
                         (args.worker_count, torch.cuda.device_count()))
    if args.worker_count != 4:
        raise SystemExit("P2 protocol requires exactly 4 GPUs")

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    checks["git_head"] = git.stdout.strip()
    checks["branch"] = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()

    files = {
        "frozen_bundle": "outputs_v2/ssl_aasist/frozen/bundle.json",
        "resources": "outputs_v2/ssl_aasist/resources/resources.json",
        "checkpoint": "outputs/source/ssl-aasist-full-asvspoof2019train-random/checkpoints/epoch_0007.pt",
        "xlsr_init": "/media/dell/data/fakeAudioDection/pretrained-model/xlsr2_300m.pt",
        "author_repo": "/media/dell/data/fakeAudioDection/SSL_Anti-spoofing/model.py",
        "cal0_cache": "outputs_v2/ssl_aasist/cache-cal0/index.json",
        "target_cache": "outputs_v2/ssl_aasist/cache-target-in_the_wild/index.json",
        "cal0_labels": "data/manifests_v2/asv2019_la/labels/cal0.jsonl",
        "target10_select": "experiments/target10_selection/manifests/inwild_target10_select.json",
        "target10_labels": "experiments/target10_selection/manifests/inwild_target10.json",
    }
    checks["files"] = {k: (ROOT / v).is_file() for k, v in files.items()}

    audits = ["tent_audit.json", "sar_audit.json", "memo_audit.json"]
    checks["audits"] = {a: (EXP_DIR / "audits" / a).is_file() for a in audits}

    run_dir = Path(args.run_dir)
    checks["output_dir_exists"] = run_dir.exists()
    if run_dir.exists():
        raise SystemExit("run directory already exists: %s" % run_dir)

    # Historical P0/P1 results must be unchanged.
    hist = ROOT / "experiments/p0_p1_taskaware/results/run_20260922_133805"
    status = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(hist.relative_to(ROOT))],
                            cwd=ROOT, capture_output=True)
    checks["historical_p0_p1_unchanged"] = status.returncode == 0

    report = {"schema_version": "0.1.0", "status": "PASS", "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
