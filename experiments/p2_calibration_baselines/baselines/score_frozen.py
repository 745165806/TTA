#!/usr/bin/env python
"""Frozen waveform scoring + 3-way parity (logits / exported head / cache).

Hard gate: max_abs_diff(logits vs exported), (exported vs cache) and
(logits vs cache) must all be <= 1e-5 for the published ports to proceed.
Label-free: reads only the manifest + waveform + checkpoint.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/p2_calibration_baselines"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))
sys.path.insert(0, str(EXP_DIR))

import torch

from author_training import build_author_model, load_audio
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources
from baselines.probe import three_view_probe

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
RESOURCES = ROOT / "outputs_v2/ssl_aasist/resources"
CACHE_TARGET = ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"
SELECT_MANIFEST = ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"


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
    row_by_id = {r["sample_id"]: r for r in select_doc["records"]}
    data_root = "/media/dell/data/fakedata/release_in_the_wild"

    cache = FeatureCache(CACHE_TARGET)
    features = cache.load_by_id()
    w_dev = resources.w.to(device)

    diffs = {"logits_vs_exported": [], "exported_vs_cache": [], "logits_vs_cache": []}
    with scores_path.open("w", encoding="utf-8") as stream:
        for sid in chosen:
            row = row_by_id[sid]
            waveform = torch.from_numpy(load_audio(data_root + "/" + row["audio_relpath"]))
            views = three_view_probe(waveform, row.get("sample_index", 0)).to(device)
            with torch.no_grad():
                embedding, logits = adapter.forward(views, freq_aug=False)
            score_logits = float(logits[0, mapping["spoof"]] - logits[0, mapping["bonafide"]])
            score_exported = float(embedding[0] @ w_dev + resources.b)
            z0 = torch.from_numpy(features[sid][0]).to(resources.w.dtype)
            score_cache = float(z0 @ resources.w + resources.b)

            d1 = abs(score_logits - score_exported)
            d2 = abs(score_exported - score_cache)
            d3 = abs(score_logits - score_cache)
            diffs["logits_vs_exported"].append(d1)
            diffs["exported_vs_cache"].append(d2)
            diffs["logits_vs_cache"].append(d3)
            stream.write(json.dumps({"sample_id": sid, "score_logits": score_logits,
                                     "score_exported": score_exported, "score_cache": score_cache}) + "\n")

    tol = 1e-5
    result = {
        "schema_version": "0.1.0",
        "seed": args.seed,
        "sample_count": len(chosen),
        "tolerance": tol,
        "tolerance_source": "frozen parity contract (parity.json atol/rtol=1e-5)",
        "max_abs_diff_logits_vs_exported": float(max(diffs["logits_vs_exported"])),
        "max_abs_diff_exported_vs_cache": float(max(diffs["exported_vs_cache"])),
        "max_abs_diff_logits_vs_cache": float(max(diffs["logits_vs_cache"])),
    }
    result["all_within_project_tolerance"] = all(
        result[k] <= tol for k in ("max_abs_diff_logits_vs_exported",
                                   "max_abs_diff_exported_vs_cache",
                                   "max_abs_diff_logits_vs_cache"))
    parity_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_within_project_tolerance"]:
        print("P2_BASELINE_BLOCKED_PARITY", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
