#!/usr/bin/env python
"""P2.1 published-baseline worker (NormOnly / TENT / SAR / MEMO).

Label-free: reads only the TargetWaveformDataset (label-free manifest) + cache
frozen reference + checkpoint.  Labels are read ONLY by the aggregator.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/p2_calibration_baselines"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))
sys.path.insert(0, str(EXP_DIR))

import torch

from author_training import build_author_model, load_audio
from eptta.baselines.ports.common import (configure_full_model,
                                          configure_normalization_adaptation,
                                          frozen_parameters, restore_parameters)
from eptta.baselines.ports.tent_audio import score_current, tent_adapt
from eptta.baselines.ports.sar_audio import sar_adapt
from eptta.baselines.ports.memo_audio import memo_adapt
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources
from baselines.probe import three_view_probe
from baselines.splits import get_split, load_split_sample_ids
from baselines.target_waveform import TargetWaveformDataset

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
RESOURCES = ROOT / "outputs_v2/ssl_aasist/resources"

METHODS = {
    "norm_only_audio": "norm",
    "tent_audio_ep": "tent",
    "sar_audio_ep": "sar",
    "memo_audio_ep_full": "memo",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=sorted(METHODS))
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    split = get_split(args.split)  # raises on unknown split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    bundle, _m, _p, _s = verify_frozen_export(FROZEN_BUNDLE)
    root = FROZEN_BUNDLE.parent
    state = torch.load(root / bundle["detector_state_ref"], map_location="cpu")
    construction = {"source_job": {"model_id": bundle["model_id"],
                                    "initialization": bundle.get("initialization")},
                    "execution": {"architecture": state["architecture"]}}
    adapter, _patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    adapter.model.eval()
    mapping = bundle["class_index_map"]

    resources, _extras, _meta = load_frozen_resources(RESOURCES, bundle)

    cache = FeatureCache(ROOT / split["feature_cache_ref"])
    features = cache.load_by_id()

    kind = METHODS[args.method]
    if kind == "memo":
        configure_full_model(adapter.model)
    else:
        configure_normalization_adaptation(adapter.model)
    frozen_params = frozen_parameters(adapter.model)

    dataset = TargetWaveformDataset(split["manifest_ref"], split["data_roots"], role=split["role"],
                                    format=split.get("manifest_format", "jsonl"))
    # Apply the split-level exclusion (e.g. itw_target90 removes target10 select).
    valid_ids, _excluded = load_split_sample_ids(args.split)
    valid_set = set(valid_ids)
    rows = [r for r in dataset.rows if r["sample_id"] in valid_set]
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.jsonl"
    if out_path.exists():
        raise SystemExit("refusing to overwrite: %s" % out_path)

    with out_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            sid = row["sample_id"]
            restore_parameters(adapter.model, frozen_params)
            waveform_cpu = torch.from_numpy(load_audio(row["audio_path"]))

            score_frozen_reference = float(torch.from_numpy(features[sid][0]) @ resources.w
                                           + resources.b)
            started = time.perf_counter()

            if kind == "memo":
                views = three_view_probe(waveform_cpu, row["sample_index"]).to(device)
                score_norm_only = score_current(adapter.model, adapter, views[:1], mapping)
                score_after = memo_adapt(adapter.model, adapter, views, mapping)
                applied, reason = True, None
                extra = {}
            else:
                wf = waveform_cpu.unsqueeze(0).to(device)
                score_norm_only = score_current(adapter.model, adapter, wf, mapping)
                if kind == "norm":
                    score_after, applied, reason = score_norm_only, False, "norm_only_control"
                    extra = {}
                elif kind == "tent":
                    score_after = tent_adapt(adapter.model, adapter, wf, mapping)
                    applied, reason, extra = True, None, {}
                else:  # sar
                    score_after, applied, reason, extra = sar_adapt(
                        adapter.model, adapter, wf, mapping)

            elapsed = time.perf_counter() - started
            record = {
                "sample_id": sid, "method": args.method,
                "score_frozen_reference": score_frozen_reference,
                "score_norm_only": score_norm_only,
                "score_after": score_after,
                "delta_total": score_after - score_frozen_reference,
                "delta_norm": score_norm_only - score_frozen_reference,
                "delta_update": score_after - score_norm_only,
                "adaptation_applied": applied,
                "abstain_reason": reason,
                "runtime": elapsed,
                "numeric_failure": False,
                "resource_failure": False,
            }
            record.update(extra)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
