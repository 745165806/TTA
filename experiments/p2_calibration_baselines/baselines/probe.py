#!/usr/bin/env python
"""Shared production 3-view probe (original / noise / FIR) for waveform scoring."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workers" / "compat"))

import torch

from author_training import independent_seed

PROBE = {"num_views": 3, "seed": 13, "noise_snr_db": 30.0, "fir_side_gain": 0.05}


def three_view_probe(waveform, sample_index, probe=None):
    probe = probe or PROBE
    generator = torch.Generator(device="cpu").manual_seed(
        independent_seed(int(probe["seed"]), int(sample_index), view_index=1, namespace=2))
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    signal_rms = waveform.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noise_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noisy = waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-float(probe["noise_snr_db"]) / 20.0))
    gain = float(probe["fir_side_gain"])
    padded = torch.nn.functional.pad(waveform[None, None], (1, 1), mode="reflect")
    kernel = torch.tensor([gain, 1.0, -gain], dtype=waveform.dtype).view(1, 1, 3)
    filtered = torch.nn.functional.conv1d(padded, kernel).view(-1)
    return torch.stack([waveform, noisy, filtered])
