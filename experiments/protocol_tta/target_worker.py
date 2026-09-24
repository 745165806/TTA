#!/usr/bin/env python
"""CUDA-only, label-free TENT protocol sequence; no resume or failure fallback."""
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from protocol import ROOT, MANIFEST, PROTOCOLS, load_config, manifest_ids, reset_info

P2 = ROOT / "experiments/p2_calibration_baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))
sys.path.insert(0, str(P2))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args):
    # Heavy imports only in actual workstation execution. No CPU/model mock path.
    import numpy as np
    import torch
    from experiments.audio_native_tta.target_worker import load_runtime
    from author_training import load_audio
    from baselines.splits import get_split
    from baselines.target_waveform import TargetWaveformDataset
    from eptta.baselines.ports.audio_native import (
        configure_audio_native, snapshot_episode_state, reset_episode_state,
        assert_bn_buffers_unchanged, selected_parameter_names)
    from eptta.baselines.ports.common import prediction_entropy
    from eptta.baselines.ports.tent_audio import logits_to_score, score_current, tent_adapt

    cfg = load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("protocol TTA requires workstation CUDA; no CPU fallback")
    device = torch.device("cuda:0")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    split = get_split("target10")
    dataset = TargetWaveformDataset(MANIFEST, split["data_roots"], role="select", format="json_records")
    rows = dataset.rows
    expected = manifest_ids()
    if [row["sample_id"] for row in rows] != expected:
        raise ValueError("dataset changed manifest order")
    if args.limit is not None:
        if args.limit != 32 or len(rows) < 32:
            raise ValueError("only a 32-sample smoke limit is supported")
        rows = rows[:32]
    adapter, bundle, resources = load_runtime(args.asset_root, device)
    model = configure_audio_native(adapter.model, cfg["parameter_scope"])
    names = selected_parameter_names(model)
    params = dict(model.named_parameters())
    selected = [params[name] for name in names]
    source_state = snapshot_episode_state(model)
    source_selected = [source_state[name] for name in names]
    bundle_path = args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    write_json(args.output / "run_config.json", {
        "schema_version": "0.1.0", "comparison_track": "protocol_tta",
        "protocol": args.protocol, "split": "target10", "parameters": cfg,
        "sample_order_policy": "manifest_order", "sample_count": len(rows),
        "manifest_path": str(MANIFEST), "asset_root": str(args.asset_root.resolve()),
        "bundle_path": str(bundle_path.resolve()), "checkpoint_ref": bundle["checkpoint_ref"],
        "detector_state_ref": str((bundle_path.parent / bundle["detector_state_ref"]).resolve()),
        "baseline_id": bundle["baseline_id"], "source_run_id": bundle["source_run_id"],
        "updated_parameter_names": names, "tau0": float(resources.tau0),
        "source_reference_policy": "separate_source_waveform_pass_before_sequence",
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })

    def distance(reference):
        with torch.no_grad():
            value = torch.stack([(p.detach().double() - ref.double()).square().sum()
                                 for p, ref in zip(selected, reference)]).sum().sqrt()
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("nonfinite selected-parameter distance")
            return float(value.item())

    def predict(waveform):
        # Match existing tent_audio.score_current's grad-enabled waveform path;
        # no backward/optimizer operation occurs in diagnostic/reference scoring.
        _, logits = adapter.forward(waveform)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("nonfinite logits")
        return (float(logits_to_score(logits, bundle["class_index_map"])[0].item()),
                float(prediction_entropy(logits).mean().item()))

    def waveform_for(row):
        return torch.from_numpy(load_audio(row["audio_path"])).unsqueeze(0).to(device)

    # One frozen pass with the original model, before any adaptation. This costs
    # no second model and never swaps/reset parameters inside the continual loop.
    references = []
    for index, row in enumerate(rows):
        torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = waveform_for(row)
        score, _entropy = predict(waveform)
        torch.cuda.synchronize()
        references.append((score, time.perf_counter() - started))
        del waveform
        if (index + 1) % 128 == 0:
            print("source_reference %d/%d" % (index + 1, len(rows)), flush=True)
    assert_bn_buffers_unchanged(model, source_state)
    optimizer = None
    with (args.output / "scores.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            reset, episode, since = reset_info(args.protocol, index)
            record = {"sample_id": row["sample_id"], "sample_index": index,
                      "manifest_sample_index": row["sample_index"], "protocol": args.protocol,
                      "reset_applied": reset, "episode_index": episode, "samples_since_reset": since,
                      "numeric_failure": False, "resource_failure": False}
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                if reset:
                    reset_episode_state(model, source_state)
                    optimizer = torch.optim.Adam(selected, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                # configure/load_state_dict/optimizer construction never occur elsewhere in this loop.
                before_params = [p.detach().clone() for p in selected]
                distance_before = distance(source_selected)
                waveform = waveform_for(row)
                before, entropy_before = predict(waveform)
                for _step in range(cfg["steps"]):
                    _, logits = adapter.forward(waveform)
                    loss = prediction_entropy(logits).mean()
                    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite TENT objective")
                    optimizer.zero_grad()
                    loss.backward()
                    gradients = [p.grad for p in selected if p.grad is not None]
                    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
                        raise FloatingPointError("missing/nonfinite TENT gradients")
                    grad_norm = float(torch.stack([g.detach().double().square().sum()
                                                  for g in gradients]).sum().sqrt().item())
                    optimizer.step()
                after, entropy_after = predict(waveform)
                delta = distance(before_params)
                drift = distance(source_selected)
                assert_bn_buffers_unchanged(model, source_state)
                torch.cuda.synchronize()
                source_score, source_runtime = references[index]
                record.update({
                    "source_frozen_score": source_score, "current_pre_adapt_score": before,
                    "score_before_update": before, "score_after": after,
                    "entropy_before": entropy_before, "entropy_after": entropy_after,
                    "grad_norm": grad_norm, "parameter_delta_norm": delta,
                    "parameter_distance_before": distance_before,
                    "parameter_distance_from_source": drift, "adaptation_applied": delta > 0,
                    "bn_running_stats_unchanged": True,
                    "source_reference_runtime": source_runtime,
                    "runtime": time.perf_counter() - started,
                })
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                del waveform, logits, loss, before_params
            except Exception as exc:
                record.update({"numeric_failure": isinstance(exc, FloatingPointError),
                               "resource_failure": isinstance(exc, (torch.cuda.OutOfMemoryError, OSError)),
                               "error": str(exc), "runtime": time.perf_counter() - started})
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                raise  # No implicit reset, continuation or invented fallback score.
            if (index + 1) % 32 == 0:
                print("%s %d/%d" % (args.protocol, index + 1, len(rows)), flush=True)
    if args.protocol == "episodic" and args.limit == 32:
        # Smoke only: replay the same real waveforms through the unchanged old
        # production TENT port, on the same GPU/model/source snapshot. No cache
        # scores, toy model, second model load, or full legacy target10 run.
        with (args.output / "legacy_tent_scores.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                reset_episode_state(model, source_state)
                waveform = waveform_for(row)
                before = score_current(model, adapter, waveform, bundle["class_index_map"])
                after = tent_adapt(model, adapter, waveform, bundle["class_index_map"],
                                   lr=cfg["lr"], steps=cfg["steps"], weight_decay=cfg["weight_decay"])
                assert_bn_buffers_unchanged(model, source_state)
                stream.write(json.dumps({"sample_id": row["sample_id"],
                                         "method": "tent_audio_native_v1",
                                         "score_before_update": before, "score_after": after},
                                        allow_nan=False) + "\n")
                stream.flush()
                del waveform
        print("legacy TENT real-waveform parity scores written (32 samples)", flush=True)
    print("complete", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--split", choices=["target10"], default="target10")
    parser.add_argument("--asset-root", type=Path, default=Path(os.environ.get("TTA_ASSET_ROOT", ROOT)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output / "failure.json", {"status": "FAIL", "error": str(exc),
                                                  "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
