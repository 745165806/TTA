#!/usr/bin/env python
"""Static CUDA/resource checks or dependency-free 32-sample smoke validation."""
import argparse
import json
import os
from pathlib import Path

from protocol import ROOT, load_config, manifest_ids, validate_run


def parse_gpus(value):
    fields = value.split(",")
    if len(fields) != 4 or any(not x.isdigit() for x in fields):
        raise ValueError("GPU_LIST requires exactly four nonnegative integer device IDs")
    gpus = [int(x) for x in fields]
    if len(set(gpus)) != 4:
        raise ValueError("GPU_LIST must contain four distinct GPUs")
    return gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-list", default=os.environ.get("GPU_LIST", "0,1,2,3"))
    parser.add_argument("--asset-root", type=Path, default=Path(os.environ.get("TTA_ASSET_ROOT", ROOT)))
    parser.add_argument("--check-smoke", type=Path)
    args = parser.parse_args()
    load_config()
    ids = manifest_ids()
    if len(ids) < 32:
        raise ValueError("target10 has fewer than 32 samples")
    if args.check_smoke:
        validate_run(args.check_smoke, ids[:32])
        print(json.dumps({"status": "PASS", "sample_count_per_protocol": 32,
                          "checks": "order, reset schedule, continuity, finite diagnostics, adaptation, BN, reference parity"}))
        return
    gpus = parse_gpus(args.gpu_list)
    import torch
    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise RuntimeError("four requested CUDA GPUs must be available")
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from eptta.models.frozen import verify_frozen_export
    from eptta.offline.artifacts import load_frozen_resources
    bundle_path = args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, *_ = verify_frozen_export(bundle_path)
    load_frozen_resources(args.asset_root / "outputs_v2/ssl_aasist/resources", bundle)
    # Scope/model/waveform checks occur in the mandatory actual 32-sample smoke.
    print(json.dumps({"status": "PASS", "GPUs": gpus, "target10_sample_count": len(ids),
                      "bundle": str(bundle_path), "next": "mandatory CUDA smoke"}))


if __name__ == "__main__":
    main()
