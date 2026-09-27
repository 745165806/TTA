#!/usr/bin/env python
"""Strictly label-free episodic worker for audio-native standard TTA."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "experiments/p2_calibration_baselines"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))
sys.path.insert(0, str(P2))

import torch

from author_training import build_author_model, load_audio
from baselines.probe import three_view_probe
from baselines.splits import get_split, load_split_sample_ids
from baselines.target_waveform import TargetWaveformDataset
from eptta.baselines.ports.audio_native import (
    SCOPE_A, assert_bn_buffers_unchanged, configure_audio_native, configure_full_safeaug,
    reset_episode_state, selected_parameter_names, snapshot_episode_state)
from eptta.baselines.ports.memo_audio import memo_adapt
from eptta.baselines.ports.sar_audio import sar_adapt
from eptta.baselines.ports.tent_audio import score_current, tent_adapt
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

METHODS = {"tent_audio_native_v1", "sar_audio_native_v1", "memo_audio_native_v1",
           "memo_audio_full_safeaug_v1",
           "tent_audio_native_scope_b_v1"}
VIEW_INDEX = {"noise_snr30": 1, "fir_side_gain0.05": 2}


def default_asset_root():
    return Path(os.environ.get("TTA_ASSET_ROOT", ROOT.parent / "TTA")).resolve()


def load_runtime(asset_root, device):
    bundle_path = asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, _m, _p, _s = verify_frozen_export(bundle_path)
    state = torch.load(bundle_path.parent / bundle["detector_state_ref"], map_location="cpu")
    construction = {"source_job": {"model_id": bundle["model_id"],
                                    "initialization": bundle.get("initialization")},
                    "execution": {"architecture": state["architecture"]}}
    adapter, _patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    resources, _extras, _meta = load_frozen_resources(
        asset_root / "outputs_v2/ssl_aasist/resources", bundle)
    return adapter, bundle, resources


def accepted_view_indices(audit_path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("source_only") or audit.get("target_labels_read") is not False:
        raise ValueError("MEMO requires a valid source-only augmentation audit")
    return [0] + [VIEW_INDEX[name] for name in audit["accepted_augmentations"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--split", default="target10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--config", type=Path,
                        default=ROOT / "experiments/audio_native_tta/config.json")
    parser.add_argument("--augmentation-audit", type=Path,
                        default=ROOT / "experiments/audio_native_tta/augmentation_audit.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    method_config = config["methods"][args.method]
    out_path = args.output / "scores.jsonl"
    if args.output.exists():
        raise SystemExit("refusing to overwrite output directory: %s" % args.output)
    args.output.mkdir(parents=True)
    (args.output / "run_config.json").write_text(json.dumps({
        "schema_version": "0.1.0", "method": args.method, "split": args.split,
        "comparison_track": config["comparison_track"], "parameters": method_config,
        "normalization_policy": config["normalization_policy"],
        "asset_root_read_only": str(args.asset_root),
    }, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter, bundle, resources = load_runtime(args.asset_root, device)
    if args.method == "memo_audio_full_safeaug_v1":
        configure_full_safeaug(adapter.model)
    else:
        configure_audio_native(adapter.model, method_config["parameter_scope"])
    episode_state = snapshot_episode_state(adapter.model)
    parameter_names = selected_parameter_names(adapter.model)

    split = get_split(args.split)
    cache_rel = Path(split["feature_cache_ref"]).relative_to(ROOT)
    features = FeatureCache(args.asset_root / cache_rel).load_by_id()
    dataset = TargetWaveformDataset(split["manifest_ref"], split["data_roots"],
                                    role=split["role"], format=split.get("manifest_format", "jsonl"))
    valid_ids, _excluded = load_split_sample_ids(args.split)
    valid = set(valid_ids)
    rows = [row for row in dataset.rows if row["sample_id"] in valid]
    if args.limit is not None:
        rows = rows[:args.limit]
    view_indices = accepted_view_indices(args.augmentation_audit) \
        if args.method in ("memo_audio_native_v1", "memo_audio_full_safeaug_v1") else None

    with out_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            reset_episode_state(adapter.model, episode_state)
            waveform_cpu = torch.from_numpy(load_audio(row["audio_path"]))
            waveform = waveform_cpu.unsqueeze(0).to(device)
            frozen_reference = float(torch.from_numpy(features[row["sample_id"]][0]) @ resources.w
                                     + resources.b)
            started = time.perf_counter()
            before = score_current(adapter.model, adapter, waveform, bundle["class_index_map"])
            applied, reason, extra = True, None, {}
            if args.method.startswith("tent_audio_native"):
                after = tent_adapt(adapter.model, adapter, waveform, bundle["class_index_map"],
                                   lr=method_config["lr"], steps=method_config["steps"],
                                   weight_decay=method_config["weight_decay"])
            elif args.method == "sar_audio_native_v1":
                after, applied, reason, extra = sar_adapt(
                    adapter.model, adapter, waveform, bundle["class_index_map"],
                    lr=method_config["lr"], momentum=method_config["momentum"],
                    rho=method_config["rho"], steps=method_config["steps"],
                    entropy_margin=method_config["entropy_margin"])
            else:
                if len(view_indices) < 2:
                    after, applied, reason = before, False, "no_safe_augmentation"
                else:
                    all_views = three_view_probe(waveform_cpu, row["sample_index"])
                    views = all_views[view_indices].to(device)
                    after = memo_adapt(adapter.model, adapter, views, bundle["class_index_map"],
                                       lr=method_config["lr"], steps=method_config["steps"],
                                       weight_decay=method_config["weight_decay"],
                                       respect_configured_scope=True)
                    extra["view_indices"] = view_indices
            assert_bn_buffers_unchanged(adapter.model, episode_state)
            elapsed = time.perf_counter() - started
            if not all(torch.isfinite(torch.tensor(value)) for value in (before, after)):
                raise FloatingPointError("non-finite audio-native score")
            record = {
                "sample_id": row["sample_id"], "method": args.method,
                "score_frozen_reference": frozen_reference,
                "score_before_update": before, "score_after": after,
                "delta_total": after - frozen_reference,
                "delta_update": after - before,
                "adaptation_applied": applied, "abstain_reason": reason,
                "runtime": elapsed, "updated_parameter_names": parameter_names,
                "normalization_policy": method_config.get(
                    "normalization_policy", config["normalization_policy"]),
                "parameter_scope_id": method_config["parameter_scope"],
                "numeric_failure": False, "resource_failure": False,
                "bn_running_stats_unchanged": True,
            }
            record.update(extra)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
