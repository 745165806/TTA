#!/usr/bin/env python
"""Frozen waveform scoring + parity vs the existing feature cache (P2 hard gate).

Loads the SSL-AASIST model from the frozen bundle, forwards N target10 waveforms
and compares each frozen score with the cached original-view score (z0 @ w + b).
Label-free: reads only the select manifest + waveform + checkpoint.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))

import torch

from author_training import build_author_model, independent_seed, load_audio
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
RESOURCES = ROOT / "outputs_v2/ssl_aasist/resources"
CACHE_TARGET = ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"
SELECT_MANIFEST = ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"

PROBE = {"num_views": 3, "seed": 13, "noise_snr_db": 30.0, "fir_side_gain": 0.05}


def _views(waveform, sample_index, probe):
    """Reproduce the production 3-view probe policy (original / noise / FIR)."""
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


def load_model(device):
    bundle, _m, _p, _s = verify_frozen_export(FROZEN_BUNDLE)
    root = FROZEN_BUNDLE.parent
    state = torch.load(root / bundle["detector_state_ref"], map_location="cpu")
    construction = {"source_job": {"model_id": bundle["model_id"],
                                    "initialization": bundle.get("initialization")},
                    "execution": {"architecture": state["architecture"]}}
    adapter, _patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    adapter.model.eval()
    return bundle, adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    parity_path = out_dir / "waveform_frozen_parity.json"
    scores_path = out_dir / "frozen_waveform_scores.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match the frozen cache extraction numerical mode (TF32 disabled) and pin
    # cuDNN to a deterministic path for cross-forward parity.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    bundle, adapter = load_model(device)
    resources, _extras, _meta = load_frozen_resources(RESOURCES, bundle)
    mapping = bundle["class_index_map"]

    select_doc = json.loads(SELECT_MANIFEST.read_text(encoding="utf-8"))
    sample_ids = [r["sample_id"] for r in select_doc["records"]]
    rng = random.Random(args.seed)
    chosen = rng.sample(sorted(sample_ids), min(args.n, len(sample_ids)))

    # Resolve audio paths from the select manifest's split metadata (root_key + relpath).
    row_by_id = {r["sample_id"]: r for r in select_doc["records"]}
    data_root = "/media/dell/data/fakedata/release_in_the_wild"

    cache = FeatureCache(CACHE_TARGET)
    features = cache.load_by_id()

    w_dev = resources.w.to(device)
    diffs = []
    with scores_path.open("w", encoding="utf-8") as stream:
        for sid in chosen:
            row = row_by_id[sid]
            audio_path = os.path.join(data_root, row["audio_relpath"])
            waveform = torch.from_numpy(load_audio(audio_path))
            views = _views(waveform, row.get("sample_index", 0), PROBE).to(device)
            with torch.no_grad():
                embedding, _logits = adapter.forward(views, freq_aug=False)
            score_waveform = float(embedding[0] @ w_dev + resources.b)
            z0 = torch.from_numpy(features[sid][0]).to(resources.w.dtype)
            score_cache = float(z0 @ resources.w + resources.b)
            diff = abs(score_waveform - score_cache)
            diffs.append(diff)
            stream.write(json.dumps({"sample_id": sid, "score_waveform": score_waveform,
                                     "score_cache": score_cache, "abs_diff": diff}) + "\n")

    tol = 1e-5
    result = {
        "schema_version": "0.1.0",
        "seed": args.seed,
        "sample_count": len(chosen),
        "mean_abs_diff": float(sum(diffs) / len(diffs)),
        "max_abs_diff": float(max(diffs)),
        "all_within_project_tolerance": bool(max(diffs) <= tol),
        "tolerance": tol,
        "tolerance_source": "frozen parity contract (parity.json atol/rtol=1e-5)",
    }
    parity_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_within_project_tolerance"]:
        print("P2_BASELINE_BLOCKED_PARITY", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
