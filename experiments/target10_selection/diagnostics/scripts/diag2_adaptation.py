#!/usr/bin/env python
"""Diagnostic 2: confirm whether EP actually performs adaptation.

Uses a deterministic 500-sample subset of the target10 select manifest and runs
an instrumented (faithful) replica of the production ``ep_tta`` loop for five
configs, recording gradient norms, parameter-update norms, score/logit deltas,
flip rate, entropy and objective before/after.  Verifies the instrumented loop
against ``run_method("ep_tta", ...)`` on a handful of samples.
"""
import argparse
import csv
import json
import statistics

import numpy as np
import torch

from _diag_common import (DIAG_RESULTS, DIAG_SEED, DIAG_SAMPLE_COUNT, DIAG_CONFIG_SPECS,
                          TargetViews, run_method, load_context, load_select_sample_ids,
                          make_config, run_ep_diagnostic, write_json_atomic, ensure_dirs)


def aggregate(values):
    if not values:
        return {"mean": None, "median": None, "max": None, "min": None, "count": 0}
    return {"mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
            "min": float(min(values)),
            "count": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    _bundle, resources, _meta, cache, features, _threshold = load_context()
    all_ids = load_select_sample_ids()
    rng = np.random.default_rng(DIAG_SEED)
    idx = rng.permutation(len(all_ids))[:DIAG_SAMPLE_COUNT]
    sample_ids = [all_ids[i] for i in idx.tolist()]
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    write_json_atomic(DIAG_RESULTS / "diagnostic_sample_ids.json",
                      {"schema_version": "0.1.0", "seed": DIAG_SEED,
                       "sample_count": len(sample_ids), "source": "target10_select",
                       "sample_ids": sample_ids})

    # --- verify instrumented loop against production run_method on first 8 samples ---
    verifications = []
    for sid in sample_ids[:8]:
        z = torch.from_numpy(features[sid])
        target = TargetViews(sid, z, cache.cache_id)
        for cfg_spec in DIAG_CONFIG_SPECS:
            cfg = make_config(cfg_spec)
            prod = float(run_method("ep_tta", target, resources, cfg, {})["score"])
            diag = run_ep_diagnostic(sid, z, resources, cfg)
            verifications.append({"sample_id": sid, "key": cfg_spec["key"],
                                  "prod_score": prod, "diag_score": diag["score_after"],
                                  "abs_diff": abs(prod - diag["score_after"])})
    max_verify_diff = max(v["abs_diff"] for v in verifications)

    # --- run instrumented diagnostics for every sample x config ---
    csv_rows = []
    per_config = {}
    representative_R = {}
    for cfg_spec in DIAG_CONFIG_SPECS:
        cfg = make_config(cfg_spec)
        sample_records = []
        for sid in sample_ids:
            z = torch.from_numpy(features[sid])
            rec = run_ep_diagnostic(sid, z, resources, cfg)
            sample_records.append(rec)
            csv_rows.append({
                "key": cfg_spec["key"], "K": cfg_spec["K"], "lr": cfg_spec["lr"],
                "sample_id": sid, "score_before": rec["score_before"],
                "score_after": rec["score_after"], "score_delta": rec["score_delta"],
                "abs_score_delta": rec["abs_score_delta"], "abs_prob_delta": rec["abs_prob_delta"],
                "flip": rec["flip"], "entropy_before": rec["entropy_before"],
                "entropy_after": rec["entropy_after"], "entropy_delta": rec["entropy_delta"],
                "objective_before": rec["objective_before"], "objective_after": rec["objective_after"],
                "keep_before": rec["keep_before"], "keep_after": rec["keep_after"],
                "grad_norm_mean": rec["grad_norm_mean"], "grad_norm_max": rec["grad_norm_max"],
                "reg_grad_norm_mean": rec["reg_grad_norm_mean"], "reg_grad_norm_max": rec["reg_grad_norm_max"],
                "update_norm_total": rec["update_norm_total"],
                "update_norm_relative_rho": rec["update_norm_relative_rho"],
                "R_norm_final": rec["R_norm_final"],
            })
        per_config[cfg_spec["key"]] = {
            "K": cfg_spec["K"], "lr": cfg_spec["lr"], "steps": cfg_spec["steps"],
            "sample_count": len(sample_records),
            "grad_norm_summary": {
                "mean_grad_norm": aggregate([r["grad_norm_mean"] for r in sample_records]),
                "max_grad_norm": aggregate([r["grad_norm_max"] for r in sample_records]),
            },
            "update_norm_summary": {
                "total_update_norm": aggregate([r["update_norm_total"] for r in sample_records]),
                "relative_rho_update_norm": aggregate([r["update_norm_relative_rho"] for r in sample_records]),
            },
            "score_delta_summary": {
                "mean_abs_score_delta": aggregate([r["abs_score_delta"] for r in sample_records]),
                "median_abs_score_delta": aggregate([r["abs_score_delta"] for r in sample_records]),
                "max_abs_score_delta": aggregate([r["abs_score_delta"] for r in sample_records]),
            },
            "logit_delta_summary": {
                "mean_abs_logit_delta": aggregate([r["abs_score_delta"] for r in sample_records]),
                "max_abs_logit_delta": aggregate([r["abs_score_delta"] for r in sample_records]),
                "note": "single-logit head: score == logit",
            },
            "prob_delta_summary": {"mean_abs_prob_delta": aggregate([r["abs_prob_delta"] for r in sample_records])},
            "flip_rate_summary": {
                "flip_rate": float(statistics.mean(r["flip"] for r in sample_records)),
                "flip_count": int(sum(r["flip"] for r in sample_records)),
            },
            "entropy_summary": {
                "mean_entropy_before": float(statistics.mean(r["entropy_before"] for r in sample_records)),
                "mean_entropy_after": float(statistics.mean(r["entropy_after"] for r in sample_records)),
                "mean_entropy_delta": float(statistics.mean(r["entropy_delta"] for r in sample_records)),
            },
            "objective_summary": {
                "mean_objective_before": float(statistics.mean(r["objective_before"] for r in sample_records)),
                "mean_objective_after": float(statistics.mean(r["objective_after"] for r in sample_records)),
                "mean_keep_before": float(statistics.mean(r["keep_before"] for r in sample_records)),
                "mean_keep_after": float(statistics.mean(r["keep_after"] for r in sample_records)),
            },
        }
        representative_R[cfg_spec["key"]] = sample_records[0]["step_trace"][-1] if sample_records[0]["step_trace"] else None

    requires_grad_report = {
        "adapted_parameter": {
            "name": "R (subspace adapter matrix)",
            "shape": [resources.U.shape[1], resources.U.shape[1]],
            "requires_grad": True,
            "trainable_element_count": int(resources.U.shape[1] * resources.U.shape[1]),
            "note": "per-sample fresh zero tensor; the ONLY requires_grad tensor in the EP loop",
        },
        "frozen_resources": {
            "U_requires_grad": bool(resources.U.requires_grad),
            "w_requires_grad": bool(resources.w.requires_grad),
            "anchors_z_requires_grad": bool(resources.anchors_z.requires_grad),
            "anchors_y_requires_grad": bool(resources.anchors_y.requires_grad),
            "note": "frozen detector / subspace / anchors are detached and never updated",
        },
        "optimizer": {
            "kind": "manual SGD, no torch.optim.Optimizer",
            "update_rule": "parameter.add_(gradient, alpha=-lr) then project_frobenius_(parameter, rho)",
            "lr": "per-candidate cfg.lr",
            "momentum": 0.0, "weight_decay": 0.0,
            "parameters_in_optimizer": "implicit: only the per-sample R tensor",
        },
        "lr_effective_note": "lr is applied directly as the SGD step size; no scheduler, no scaling.",
    }

    doc = {
        "schema_version": "0.1.0",
        "diagnostic": "ep_adaptation_effectiveness",
        "seed": DIAG_SEED,
        "sample_count": len(sample_ids),
        "instrumented_loop_max_abs_diff_vs_production": max_verify_diff,
        "instrumented_loop_verification_note":
            "max |instrumented score - run_method score| over first 8 samples x 5 configs",
        "requires_grad_report": requires_grad_report,
        "configs": per_config,
        "representative_step_trace_last": representative_R,
    }

    csv_path = DIAG_RESULTS / "adaptation_diagnostics.csv"
    fieldnames = ["key", "K", "lr", "sample_id", "score_before", "score_after", "score_delta",
                  "abs_score_delta", "abs_prob_delta", "flip", "entropy_before", "entropy_after",
                  "entropy_delta", "objective_before", "objective_after", "keep_before", "keep_after",
                  "grad_norm_mean", "grad_norm_max", "reg_grad_norm_mean", "reg_grad_norm_max",
                  "update_norm_total", "update_norm_relative_rho", "R_norm_final"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    write_json_atomic(DIAG_RESULTS / "adaptation_diagnostics.json", doc)
    print(json.dumps({"instrumented_loop_max_abs_diff_vs_production": max_verify_diff,
                      "sample_count": len(sample_ids),
                      "configs": {k: {"grad_norm_mean": v["grad_norm_summary"]["mean_grad_norm"]["mean"],
                                     "update_norm_mean": v["update_norm_summary"]["total_update_norm"]["mean"],
                                     "mean_abs_score_delta": v["score_delta_summary"]["mean_abs_score_delta"]["mean"],
                                     "flip_rate": v["flip_rate_summary"]["flip_rate"]}
                                  for k, v in per_config.items()}},
                     ensure_ascii=False, indent=2))
    print("wrote", csv_path)
    print("wrote", DIAG_RESULTS / "adaptation_diagnostics.json")
    print("wrote", DIAG_RESULTS / "diagnostic_sample_ids.json")


if __name__ == "__main__":
    main()
