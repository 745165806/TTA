#!/usr/bin/env python
"""Resource, contract, scope, and 32-sample before-path preflight."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/audio_native_tta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))
sys.path.insert(0, str(EXP))

import torch

from aggregate import BEFORE_TOLERANCE, METHODS, before_path_diagnostics, read_scores
from audit_model import load_model
from eptta.baselines.ports.audio_native import SCOPE_A, SCOPE_B, preregistered_parameter_names


def default_asset_root():
    return Path(os.environ.get("TTA_ASSET_ROOT", ROOT.parent / "TTA")).resolve()


def parse_gpus(value):
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("GPU_LIST must contain integer device IDs") from exc
    if not result or len(result) != len(set(result)) or any(index < 0 for index in result):
        raise ValueError("GPU_LIST must contain unique non-negative device IDs")
    return result


def validate_static(args):
    gpus = parse_gpus(args.gpu_list)
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or any(index >= count for index in gpus):
        raise RuntimeError("requested GPUs %s unavailable (torch sees %d CUDA devices)" % (gpus, count))
    required = {
        "frozen_bundle": args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json",
        "target_manifest": ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json",
        "target_cache": args.asset_root / "outputs_v2/ssl_aasist/cache-target-in_the_wild/index.json",
        "augmentation_audit": EXP / "augmentation_audit.json",
    }
    missing = {name: str(path) for name, path in required.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError("missing preflight resources: %s" % missing)
    audit = json.loads(required["augmentation_audit"].read_text(encoding="utf-8"))
    status = {row["candidate"]: bool(row["accepted"]) for row in audit["candidates"]}
    if (not audit.get("source_only") or audit.get("target_labels_read") is not False or
            status.get("fir_side_gain0.05") is not True or status.get("noise_snr30") is not False or
            audit.get("accepted_augmentations") != ["fir_side_gain0.05"]):
        raise ValueError("source augmentation audit is not the frozen noise-reject/FIR-accept contract")
    if args.output_root.exists():
        raise FileExistsError("output root already exists: %s" % args.output_root)
    for method_id in METHODS:
        if (args.output_root / method_id).exists() or (args.output_root / "smoke" / method_id).exists():
            raise FileExistsError("method output already exists: %s" % method_id)

    model, _bundle, _patch, _path = load_model(args.asset_root, torch.device("cpu"))
    scopes = {SCOPE_A: preregistered_parameter_names(model, SCOPE_A),
              SCOPE_B: preregistered_parameter_names(model, SCOPE_B)}
    for scope_id, names in scopes.items():
        if (not names or any(name.startswith("ssl_model.") or name.startswith("out_layer.")
                             for name in names)):
            raise ValueError("scope escaped frontend/head exclusion: %s" % scope_id)
    return {"CUDA": True, "GPU_count": count, "requested_GPUs": gpus,
            "resources": {name: str(path) for name, path in required.items()},
            "augmentation_views": ["original", "fir_side_gain0.05"],
            "scope_tensor_counts": {scope: len(names) for scope, names in scopes.items()},
            "output_root_new": True}


def validate_smoke(smoke_root):
    records = {}
    for method_id, (label, _gain, _neutral) in METHODS.items():
        path = smoke_root / method_id / "scores.jsonl"
        if not path.is_file():
            raise FileNotFoundError("missing smoke scores: %s" % path)
        records[label] = read_scores(path)
        if len(records[label]) != 32:
            raise ValueError("smoke method does not contain exactly 32 samples: %s" % label)
        for row in records[label].values():
            if row.get("numeric_failure") or row.get("resource_failure"):
                raise ValueError("smoke recorded numeric/resource failure: %s" % label)
            if row.get("bn_running_stats_unchanged") is not True:
                raise ValueError("smoke did not verify BN buffers: %s" % label)
        if label in ("MEMO-FullSafeAug", "MEMO-Audio-Limited"):
            if any(row.get("view_indices") != [0, 2] for row in records[label].values()):
                raise ValueError("MEMO smoke views differ from original+FIR: %s" % label)
    parity = before_path_diagnostics(records, BEFORE_TOLERANCE)
    if parity["AUDIO_NATIVE_BEFORE_PATH_PARITY_FAIL"]:
        raise ValueError("before-path smoke parity failed: %s" % parity)
    return parity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-list", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--check-smoke", type=Path)
    args = parser.parse_args()
    if args.check_smoke is not None:
        print(json.dumps({"smoke": validate_smoke(args.check_smoke)}, indent=2))
        return
    if args.output_root is None:
        parser.error("--output-root is required for static preflight")
    print(json.dumps(validate_static(args), indent=2))


if __name__ == "__main__":
    main()
