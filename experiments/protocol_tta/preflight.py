#!/usr/bin/env python
"""Static CUDA/resource checks or dependency-free 32-sample smoke validation."""
import argparse
import json
import math
import os
from pathlib import Path

from protocol import ROOT, MANIFEST, load_config, manifest_ids, validate_run


def validate_target10_manifest():
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (doc.get("role") != "select" or type(doc.get("count")) is not int or
            doc["count"] != 3178 or not isinstance(doc.get("records"), list) or
            len(doc["records"]) != 3178):
        raise ValueError("target10 requires role=select, count=3178, len(records)=3178")
    # Existing strict row whitelist, select roles and unique IDs remain mandatory.
    return manifest_ids()


def validate_legacy_parity(smoke_root, episodic_rows, expected_ids):
    path = smoke_root / "episodic/legacy_tent_scores.jsonl"
    legacy = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(expected_ids) != 32 or [r["sample_id"] for r in legacy] != expected_ids:
        raise ValueError("legacy TENT parity requires exactly the same 32 IDs in manifest order")
    if [r["sample_id"] for r in episodic_rows] != expected_ids:
        raise ValueError("episodic parity sample order mismatch")
    maxima = {"source_before": 0.0, "current_before": 0.0, "post_update": 0.0}
    for old, new in zip(legacy, episodic_rows):
        if old["method"] != "tent_audio_native_v1":
            raise ValueError("wrong legacy parity method")
        for name, new_key, old_key in (
                ("source_before", "source_frozen_score", "score_before_update"),
                ("current_before", "score_before_update", "score_before_update"),
                ("post_update", "score_after", "score_after")):
            a, b = new[new_key], old[old_key]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not math.isfinite(v) for v in (a, b)):
                raise ValueError("nonfinite legacy/episodic parity score")
            maxima[name] = max(maxima[name], abs(a - b))
    if any(value > 1e-5 for value in maxima.values()):
        raise ValueError("SMOKE FAIL: episodic/legacy TENT parity mismatch: %s" % maxima)
    return {"status": "PASS", "sample_count": 32, "absolute_tolerance": 1e-5,
            "max_abs_difference": maxima, "legacy_scores": str(path)}


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
    ids = validate_target10_manifest()
    if args.check_smoke:
        records, _config = validate_run(args.check_smoke, ids[:32])
        parity = validate_legacy_parity(args.check_smoke, records["episodic"], ids[:32])
        print(json.dumps({"status": "PASS", "sample_count_per_protocol": 32,
                          "legacy_tent_parity": parity,
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
