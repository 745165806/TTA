"""Frozen waveform parity integration test.

The full SSL-AASIST model (300M XLS-R) is heavy; this test SKIPs by default and
is executed for real only when the model resources are present (server P2
preflight).  A skipped run is NOT_RUN, never a fabricated parity PASS.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
CHECKPOINT = ROOT / "outputs/source/ssl-aasist-full-asvspoof2019train-random/checkpoints/epoch_0007.pt"
XLSR = Path("/media/dell/data/fakeAudioDection/pretrained-model/xlsr2_300m.pt")


def _resources_present():
    import torch
    return (BUNDLE.is_file() and CHECKPOINT.is_file() and XLSR.is_file()
            and torch.cuda.is_available())


@pytest.mark.skipif(not _resources_present(),
                    reason="NOT_RUN_RESOURCE: SSL-AASIST model/GPU resources are unavailable")
def test_frozen_waveform_parity_real(tmp_path):
    import json
    import subprocess
    import sys
    script = ROOT / "experiments/p2_calibration_baselines/baselines/score_frozen.py"
    output = tmp_path / "parity"
    completed = subprocess.run([sys.executable, str(script), "--split", "target10",
                                "--output", str(output), "--n", "8"],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "waveform_frozen_parity.json").read_text())
    assert report["all_within_project_tolerance"] is True
    for key in ("max_abs_diff_logits_vs_exported", "max_abs_diff_exported_vs_cache",
                "max_abs_diff_logits_vs_cache"):
        assert report[key] <= report["tolerance"]
