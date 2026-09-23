#!/usr/bin/env python
"""Run one label-free P3 development variant on evalU."""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "scripts"))

import torch

from eptta.baselines.dispatch import run_method
from p3_common import (ep_config, frozen_score, load_context, load_ids,
                       make_target, p1_params, p3_params)

torch.set_num_threads(1)

VARIANTS = ("Frozen-CalOnly", "SourceTauTeacher", "CalibratedTeacher-NoSelect",
            "CalibratedSelective")


def record_result(variant, result):
    return {
        "variant": variant, "method_id": result.get("method_id"),
        "sample_id": result["sample_id"], "score_before": result["score_before"],
        "score_after": result["score"], "delta_score": result["score"] - result["score_before"],
        "adaptation_applied": bool(result.get("adaptation_applied", False)),
        "abstain_reason": result.get("abstain_reason"),
        "teacher_label": result.get("teacher_label"),
        "teacher_p_spoof": result.get("teacher_p_spoof"),
        "gate_confidence": result.get("gate_confidence"),
        "gate_agreement": result.get("gate_agreement"),
        "safety_rejected": bool(result.get("safety_rejected", False)),
        "source_anchor_flip_count": result.get("source_anchor_flip_count"),
        "final_source_anchor_flip_count": result.get("final_source_anchor_flip_count"),
        "numeric_fallback": result.get("status") == "fallback_numeric",
    }


def frozen_record(variant, sample_id, score, reason=None):
    return {
        "variant": variant, "method_id": "frozen", "sample_id": sample_id,
        "score_before": score, "score_after": score, "delta_score": 0.0,
        "adaptation_applied": False, "abstain_reason": reason,
        "teacher_label": None, "teacher_p_spoof": None,
        "gate_confidence": None, "gate_agreement": None,
        "safety_rejected": False, "source_anchor_flip_count": 0,
        "final_source_anchor_flip_count": 0, "numeric_fallback": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("refusing to overwrite: %s" % out)
    config, resources, _meta, cache, features = load_context()
    sample_ids = load_ids("evalU", args.manifest_dir)
    if args.limit is not None:
        sample_ids = sample_ids[:args.limit]
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration_ok = calibration.get("status") == "OK"
    cfg = ep_config(config)
    with out.open("w", encoding="utf-8") as stream:
        for sample_id in sample_ids:
            if args.variant == "Frozen-CalOnly":
                record = frozen_record(args.variant, sample_id,
                                       frozen_score(sample_id, features, resources))
            elif not calibration_ok:
                record = frozen_record(args.variant, sample_id,
                                       frozen_score(sample_id, features, resources),
                                       "calibration_fit_fail")
            else:
                target = make_target(sample_id, features, cache.cache_id)
                if args.variant == "SourceTauTeacher":
                    result = run_method("ep_tta_taskaware_v1", target, resources, cfg,
                                        p1_params(config))
                else:
                    params = p3_params(config, calibration,
                                       args.variant == "CalibratedSelective")
                    result = run_method("ep_tta_calibrated_selective_v1", target,
                                        resources, cfg, params)
                record = record_result(args.variant, result)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print("%s: wrote %d records to %s" % (args.variant, len(sample_ids), out))


if __name__ == "__main__":
    main()
