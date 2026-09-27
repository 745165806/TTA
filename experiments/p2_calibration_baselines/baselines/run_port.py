#!/usr/bin/env python
"""P2 published-baseline pilot worker (TENT/SAR episodic, per-sample reset).

Label-free: reads only the inference manifest + waveform + checkpoint.  Target
labels are read ONLY by the aggregator.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))

import torch

from author_training import build_author_model, load_audio
from eptta.baselines.ports.common import (configure_normalization_adaptation,
                                          frozen_parameters, restore_parameters)
from eptta.baselines.ports.tent_audio import tent_episodic
from eptta.baselines.ports.sar_audio import sar_episodic
from eptta.models.frozen import verify_frozen_export

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
SELECT_MANIFEST = ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["tent_audio_ep", "sar_audio_ep"])
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

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
    configure_normalization_adaptation(adapter.model)
    frozen_params = frozen_parameters(adapter.model)
    mapping = bundle["class_index_map"]

    select_doc = json.loads(SELECT_MANIFEST.read_text(encoding="utf-8"))
    row_by_id = {r["sample_id"]: r for r in select_doc["records"]}
    sample_ids = [r["sample_id"] for r in select_doc["records"]]
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    data_root = "/media/dell/data/fakedata/release_in_the_wild"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.jsonl"
    if out_path.exists():
        raise SystemExit("refusing to overwrite: %s" % out_path)

    with out_path.open("w", encoding="utf-8") as stream:
        for sid in sample_ids:
            restore_parameters(adapter.model, frozen_params)
            audio_path = os.path.join(data_root, row_by_id[sid]["audio_relpath"])
            waveform = torch.from_numpy(load_audio(audio_path)).unsqueeze(0).to(device)
            started = time.perf_counter()
            if args.method == "tent_audio_ep":
                sb, sa = tent_episodic(adapter.model, adapter, waveform, mapping)
                applied, reason = True, None
            else:
                sb, sa, applied, reason = sar_episodic(adapter.model, adapter, waveform, mapping)
            elapsed = time.perf_counter() - started
            stream.write(json.dumps({
                "sample_id": sid, "method": args.method,
                "score_before": sb, "score_after": sa, "delta_score": sa - sb,
                "adaptation_applied": applied, "abstain_reason": reason,
                "runtime": elapsed,
            }, sort_keys=True, allow_nan=False) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
