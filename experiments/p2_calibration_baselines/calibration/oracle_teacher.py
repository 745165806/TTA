#!/usr/bin/env python
"""P2-B oracle-teacher counterfactual.

POST_HOC_ORACLE_ONLY
TARGET_LABELS_REQUIRED
NOT_A_DEPLOYABLE_METHOD

Same R/U/objective/gate/safety/steps/lr/rho as P1 ``taskaware_full``; the ONLY
change is the pseudo-BCE teacher: ground-truth label instead of the frozen tau0
pseudo label.  The selective gate still uses the label-free frozen agreement /
confidence (ground truth never changes whether a sample is adapted).
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from eptta.adaptation.adapter import (_taskaware_gate_state, _taskaware_loss_components,
                                      _taskaware_total)
from eptta.adaptation.math import apply_adapter, project_frobenius_
from eptta.adaptation.types import EPConfig, TargetViews
from eptta.adaptation.validation import validate_inputs
from eptta.cache.reader import FeatureCache
from calibration.common import (CACHE_TARGET, POST_HOC_MARKER, load_resources,
                                load_target10)

torch.set_num_threads(1)

P1_CFG = dict(steps=3, lr=0.03, rho=0.05, gamma=0.1, lambda_keep=1.0)
P1_PARAMS = {"lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 1.0,
             "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.5,
             "min_agreement": 1.0}
P1_FULL_JSONL = (ROOT / "experiments/p0_p1_taskaware/results/run_20260922_133805"
                 / "p1/p1_taskaware_full.jsonl")


def run_oracle_teacher(sample_id, features, resources, cache_id, y, cfg, params):
    target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache_id)
    before = validate_inputs(target, resources, cfg)
    rank = resources.U.shape[1]
    zero_R = torch.zeros((rank, rank), dtype=target.features.dtype, device=target.features.device)
    _frozen_teacher, agreement, confidence = _taskaware_gate_state(target, resources)

    if agreement < params["min_agreement"] or confidence < params["confidence_margin"]:
        return {"sample_id": sample_id, "score_before": before, "score_after": before,
                "delta_score": 0.0, "adaptation_applied": False,
                "abstain_reason": "low_confidence_or_view_disagreement",
                "safety_rejected": False}

    parameter = torch.zeros((rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    try:
        with torch.enable_grad():
            for _step in range(cfg.steps):
                components = _taskaware_loss_components(parameter, target, resources, y, params)
                loss = _taskaware_total(components, params)
                gradient, = torch.autograd.grad(loss, parameter)
                with torch.no_grad():
                    parameter.add_(gradient, alpha=-cfg.lr)
                    project_frobenius_(parameter, cfg.rho)
        R = parameter.detach().clone()
        calibrated = apply_adapter(resources.anchors_z, resources.U, R) @ resources.w + resources.b - resources.tau0
        flips = int(((calibrated >= 0).to(torch.long) != resources.anchors_y.to(torch.long)).sum().item())
        if flips > 0:
            return {"sample_id": sample_id, "score_before": before, "score_after": before,
                    "delta_score": 0.0, "adaptation_applied": False, "abstain_reason": None,
                    "safety_rejected": True}
        score = float((apply_adapter(target.features[:1], resources.U, R)
                       @ resources.w + resources.b)[0])
        return {"sample_id": sample_id, "score_before": before, "score_after": score,
                "delta_score": score - before, "adaptation_applied": True,
                "abstain_reason": None, "safety_rejected": False}
    except FloatingPointError:
        return {"sample_id": sample_id, "score_before": before, "score_after": before,
                "delta_score": 0.0, "adaptation_applied": False, "abstain_reason": None,
                "safety_rejected": False, "numeric_fallback": True}


def compute_metrics(records, labels, threshold):
    from eptta.evaluation.metrics import binary_metrics
    after = {r["sample_id"]: r["score_after"] for r in records}
    before = {r["sample_id"]: r["score_before"] for r in records}
    order = sorted(after)
    m = binary_metrics([after[k] for k in order], [labels[k] for k in order],
                       threshold, frozen_scores=[before[k] for k in order])
    signed = [(2 * labels[r["sample_id"]] - 1) * r["delta_score"] for r in records]
    b = [sd for sd, r in zip(signed, records) if labels[r["sample_id"]] == 0]
    s = [sd for sd, r in zip(signed, records) if labels[r["sample_id"]] == 1]
    n = len(records)
    return {
        "EER": m["eer"], "AUC": m["auroc"],
        "mean_signed_task_delta": statistics.fmean(signed),
        "bonafide_mean_signed_delta": statistics.fmean(b) if b else None,
        "spoof_mean_signed_delta": statistics.fmean(s) if s else None,
        "helpful_flips": m["helpful_flips"], "harmful_flips": m["harmful_flips"],
        "adaptation_coverage": sum(1 for r in records if r.get("adaptation_applied")) / n,
        "safety_reject_rate": sum(1 for r in records if r.get("safety_rejected")) / n,
        "mean_R_norm": None, "mean_abs_delta_score": statistics.fmean(abs(r["delta_score"]) for r in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.run_dir) / "oracle_teacher"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, ys, sample_ids, resources = load_target10()
    labels = {sid: y for sid, y in zip(sample_ids, ys)}
    bundle, resources, _meta = load_resources()
    cache = FeatureCache(CACHE_TARGET)
    features = cache.load_by_id()
    cfg = EPConfig(**P1_CFG)

    oracle_records = []
    for sid in sample_ids:
        r = run_oracle_teacher(sid, features, resources, cache.cache_id, labels[sid], cfg, P1_PARAMS)
        oracle_records.append(r)
    with (out_dir / "oracle_teacher_records.jsonl").open("w", encoding="utf-8") as stream:
        for r in oracle_records:
            stream.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")

    oracle_metrics = compute_metrics(oracle_records, labels, resources.tau0)

    # Pseudo-teacher reference from the historical P1 taskaware_full run.
    pseudo_records = []
    with P1_FULL_JSONL.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                pseudo_records.append(json.loads(line))
    pseudo_metrics = compute_metrics(pseudo_records, labels, resources.tau0)

    pseudo_signed = pseudo_metrics["mean_signed_task_delta"]
    oracle_signed = oracle_metrics["mean_signed_task_delta"]
    teacher_mismatch_supported = (pseudo_signed <= 0 and oracle_signed > 0)
    teacher_mismatch_insufficient = (oracle_signed <= 0)

    summary = {
        "POST_HOC_ORACLE_ONLY": True,
        "TARGET_LABELS_REQUIRED": True,
        "NOT_A_DEPLOYABLE_METHOD": True,
        "pseudo_teacher": pseudo_metrics,
        "oracle_teacher": oracle_metrics,
        "delta_signed_direction": oracle_signed - pseudo_signed,
        "TEACHER_MISMATCH_SUPPORTED": bool(teacher_mismatch_supported),
        "TEACHER_MISMATCH_INSUFFICIENT": bool(teacher_mismatch_insufficient),
    }
    (out_dir / "oracle_teacher_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["method", "EER", "AUC", "mean_signed_task_delta",
              "bonafide_mean_signed_delta", "spoof_mean_signed_delta",
              "helpful_flips", "harmful_flips", "adaptation_coverage",
              "safety_reject_rate", "mean_abs_delta_score"]
    with (out_dir / "oracle_teacher_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "pseudo_teacher", **{k: pseudo_metrics[k] for k in fields[1:]}})
        writer.writerow({"method": "oracle_teacher", **{k: oracle_metrics[k] for k in fields[1:]}})

    (out_dir / "oracle_teacher_summary.md").write_text(
        "# P2-B Oracle-teacher counterfactual\n\n"
        "**POST_HOC_ORACLE_ONLY / TARGET_LABELS_REQUIRED / NOT_A_DEPLOYABLE_METHOD**\n\n"
        "pseudo signed delta = %.6f, oracle signed delta = %.6f\n\n"
        "TEACHER_MISMATCH_SUPPORTED = %s\nTEACHER_MISMATCH_INSUFFICIENT = %s\n" %
        (pseudo_signed, oracle_signed, teacher_mismatch_supported, teacher_mismatch_insufficient),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
