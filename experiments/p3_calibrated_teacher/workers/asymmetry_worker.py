#!/usr/bin/env python
"""P3.2 class-asymmetric mechanism worker (CPU/cache only)."""
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

from eptta.adaptation.calibrated_selective import posterior_spoof
from eptta.baselines.dispatch import run_method
from p3_common import (ep_config, load_context, load_ids, make_target, p3_params,
                       resolve_input)
from workers.oracle_teacher import run_one as run_oracle
from workers.run_variant import record_result

torch.set_num_threads(1)

VARIANTS = (
    "Oracle-BonaOnly", "Oracle-SpoofOnly",
    "Calibrated-BonaOnly", "Calibrated-SpoofOnly",
)


def class_mode(variant):
    return "bona_only" if variant.endswith("BonaOnly") else "spoof_only"


def class_is_held(teacher, mode):
    return (mode == "bona_only" and teacher == 1) or (mode == "spoof_only" and teacher == 0)


def held_record(variant, target, resources, params, teacher, oracle=False):
    before = float(target.features[0] @ resources.w + resources.b)
    posterior = posterior_spoof(before, params)
    calibrated_teacher = int(posterior >= 0.5)
    view_labels = target.features @ resources.w + resources.b >= params["tau_hat"]
    agreement = float((view_labels == bool(calibrated_teacher)).to(target.features.dtype).mean())
    reason = "class_asymmetric_spoof_hold" if teacher == 1 else "class_asymmetric_bonafide_hold"
    result = {
        "variant": variant,
        "method_id": "POST_HOC_ORACLE_ONLY" if oracle else "ep_tta_calibrated_selective_v1",
        "sample_id": target.sample_id, "score_before": before, "score_after": before,
        "delta_score": 0.0, "adaptation_applied": False,
        "abstain_reason": reason, "teacher_label": teacher,
        "teacher_p_spoof": posterior, "gate_confidence": max(posterior, 1.0 - posterior),
        "gate_agreement": agreement, "safety_rejected": False,
        "source_anchor_flip_count": 0, "final_source_anchor_flip_count": 0,
        "numeric_fallback": False,
    }
    if oracle:
        result.update(POST_HOC_ORACLE_ONLY=True, NOT_DEPLOYABLE=True)
    return result


def load_posthoc_labels(config):
    doc = json.loads(resolve_input(config["inputs"]["target10_label_manifest"]).read_text(
        encoding="utf-8"))
    return {row["sample_id"]: int(row["label"]) for row in doc["records"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite: %s" % output)
    config, resources, _meta, cache, features = load_context()
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("status") != "OK":
        raise SystemExit("CALIBRATION_FIT_FAIL: asymmetric adaptation disabled")
    params = p3_params(config, calibration, True)
    cfg = ep_config(config)
    ids = load_ids("evalU")
    oracle = args.variant.startswith("Oracle-")
    labels = load_posthoc_labels(config) if oracle else None
    mode = class_mode(args.variant)
    with output.open("w", encoding="utf-8") as stream:
        for sid in ids:
            target = make_target(sid, features, cache.cache_id)
            before = float(target.features[0] @ resources.w + resources.b)
            calibrated_teacher = int(posterior_spoof(before, params) >= 0.5)
            teacher = labels[sid] if oracle else calibrated_teacher
            if class_is_held(teacher, mode):
                record = held_record(args.variant, target, resources, params, teacher, oracle)
            elif oracle:
                record = run_oracle(target, resources, cfg, params, teacher)
                record["variant"] = args.variant
            else:
                result = run_method("ep_tta_calibrated_selective_v1", target,
                                    resources, cfg, params)
                record = record_result(args.variant, result)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print("%s: wrote %d records to %s" % (args.variant, len(ids), output))


if __name__ == "__main__":
    main()
