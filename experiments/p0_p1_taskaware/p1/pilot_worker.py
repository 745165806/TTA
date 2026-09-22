#!/usr/bin/env python
"""P1 pilot worker: run one ablation variant on the label-free target10 select set.

Variant -> method mapping:
  taskaware_full           ep_tta_taskaware_v1 (default params)
  taskaware_no_gate        ep_tta_taskaware_v1 (gate disabled)
  taskaware_no_source_keep ep_tta_taskaware_v1 (lambda_source = 0, safety keep)
  taskaware_control        ep_tta_guarded K=3/lr=0.03 (matched control)

This worker MUST NOT read target labels.
"""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

import torch

from eptta.adaptation.types import EPConfig
from eptta.baselines.dispatch import run_method
from scripts.common import load_context, load_select_sample_ids, make_target

torch.set_num_threads(1)

PILOT_CFG = dict(steps=3, lr=0.03, rho=0.05, gamma=0.1, lambda_keep=1.0)

VARIANTS = {
    "taskaware_full": ("ep_tta_taskaware_v1", {
        "lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 1.0,
        "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.5,
        "min_agreement": 1.0,
    }),
    "taskaware_no_gate": ("ep_tta_taskaware_v1", {
        "lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 1.0,
        "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.0,
        "min_agreement": 0.0,
    }),
    "taskaware_no_source_keep": ("ep_tta_taskaware_v1", {
        "lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 0.0,
        "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.5,
        "min_agreement": 1.0,
    }),
    "taskaware_control": ("ep_tta_guarded", {}),
}


def record_sample(variant, sample_id, features, resources, cache_id):
    method_id, params = VARIANTS[variant]
    target = make_target(sample_id, features, cache_id)
    cfg = EPConfig(**PILOT_CFG)
    result = run_method(method_id, target, resources, cfg, params)

    status = result.get("status")
    numeric_fallback = status != "ok"
    score_before = float(result["score_before"])
    score_after = float(result["score"])

    loss_fields = [
        "pseudo_loss_before", "task_consistency_loss_before", "source_logit_loss_before",
        "parameter_l2_before", "total_objective_before",
        "pseudo_loss_final", "task_consistency_loss_final", "source_logit_loss_final",
        "parameter_l2_final", "total_objective_final",
        "attempted_pseudo_loss", "attempted_task_consistency_loss",
        "attempted_source_logit_loss", "attempted_parameter_l2", "attempted_total_objective",
    ]
    if method_id == "ep_tta_taskaware_v1":
        record = {
            "variant": variant, "method_id": method_id, "sample_id": sample_id,
            "score_before": score_before, "score_after": score_after,
            "delta_score": score_after - score_before,
            "adaptation_applied": bool(result.get("adaptation_applied", False)),
            "abstain_reason": result.get("abstain_reason"),
            "teacher_label": result.get("teacher_label"),
            "gate_agreement": result.get("gate_agreement"),
            "gate_confidence": result.get("gate_confidence"),
            "source_anchor_flip_count": result.get("source_anchor_flip_count"),
            "attempted_source_anchor_flip_count": result.get("attempted_source_anchor_flip_count"),
            "final_source_anchor_flip_count": result.get("final_source_anchor_flip_count"),
            "safety_rejected": bool(result.get("safety_rejected", False)),
            "final_R_norm": float(result.get("final_R_norm", 0.0)),
            "attempted_R_norm": float(result.get("attempted_R_norm", 0.0) or 0.0),
            "steps_completed": int(result.get("steps_completed", 0)),
            "numeric_fallback": numeric_fallback,
        }
        for name in loss_fields:
            record[name] = result.get(name)
        return record
    # matched control (ep_tta_guarded) has no task-aware diagnostics
    applied = (status == "ok" and int(result.get("steps_completed", 0)) > 0 and
               abs(score_after - score_before) > 0)
    R = result.get("R")
    r_norm = float(torch.linalg.vector_norm(R)) if R is not None else 0.0
    record = {
        "variant": variant, "method_id": method_id, "sample_id": sample_id,
        "score_before": score_before, "score_after": score_after,
        "delta_score": score_after - score_before,
        "adaptation_applied": applied,
        "abstain_reason": None,
        "teacher_label": None, "gate_agreement": None, "gate_confidence": None,
        "source_anchor_flip_count": None, "attempted_source_anchor_flip_count": None,
        "final_source_anchor_flip_count": None, "safety_rejected": False,
        "final_R_norm": r_norm, "attempted_R_norm": r_norm,
        "steps_completed": int(result.get("steps_completed", 0)),
        "numeric_fallback": numeric_fallback,
    }
    for name in loss_fields:
        record[name] = None
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=sorted(VARIANTS))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        raise SystemExit("unknown variant: %s" % args.variant)

    sample_ids = load_select_sample_ids()
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    _bundle, resources, _meta, cache, features, _threshold = load_context()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("p1_%s.jsonl" % args.variant)
    if out_path.exists():
        raise SystemExit("refusing to overwrite: %s" % out_path)

    written = 0
    with out_path.open("w", encoding="utf-8") as stream:
        for sample_id in sample_ids:
            record = record_sample(args.variant, sample_id, features, resources, cache.cache_id)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            written += 1
    print("pilot %s: wrote %s (%d records)" % (args.variant, out_path, written))


if __name__ == "__main__":
    main()
