#!/usr/bin/env python
"""Diagnostic 4 (worker): per-sample scoring of all 13 candidates on target10.

POST-HOC label analysis support.  The original selection was strictly label-free;
this script scores every candidate on the full 3178-sample target10 select set so
a separate aggregator can compute *true* supervised metrics (EER/AUC/accuracy)
using the raw audit labels.  Labels are used ONLY to judge the metric afterwards.

Run with --group N --num-groups M for parallel CPU processes (each writes its own
posthoc_group_N.json, so no two processes touch the same output file).
"""
import argparse
import json

import torch

from _diag_common import (DIAG_RESULTS, RHO, GAMMA, LAMBDA_KEEP, candidates, sigmoid,
                          binary_entropy, apply_adapter, load_context, load_select_sample_ids,
                          make_config, selection_score, run_method, TargetViews,
                          write_json_atomic, ensure_dirs)


def evaluate_candidate_with_scores(cand, sample_ids, features, resources, cache_id):
    cfg = make_config(cand)
    entropy_sum = consistency_sum = stability_sum = 0.0
    n = 0
    scores = {}
    for sample_id in sample_ids:
        z = torch.from_numpy(features[sample_id])
        target = TargetViews(sample_id, z, cache_id)
        result = run_method("ep_tta", target, resources, cfg, {})

        p_before = sigmoid(result["score_before"])
        p_after = sigmoid(result["score"])
        entropy_sum += binary_entropy(p_before) - binary_entropy(p_after)

        with torch.no_grad():
            R = result["R"].to(z.dtype).to(z.device)
            view_scores = apply_adapter(z, resources.U, R) @ resources.w + resources.b
        view_probs = [sigmoid(s) for s in view_scores.detach().cpu().tolist()]
        votes = [1 if p > 0.5 else 0 for p in view_probs]
        majority = max(votes.count(0), votes.count(1))
        consistency_sum += majority / 3.0
        confidences = [abs(p - 0.5) for p in view_probs]
        mean_c = sum(confidences) / 3.0
        std_c = (sum((c - mean_c) ** 2 for c in confidences) / 3.0) ** 0.5
        stability_sum += max(0.0, min(1.0, 1.0 - std_c / 0.5))
        scores[sample_id] = float(result["score"])
        n += 1

    return {
        "K": cand["K"], "lr": cand["lr"], "steps": cand["steps"],
        "entropy": entropy_sum / n,
        "consistency": consistency_sum / n,
        "stability": stability_sum / n,
        "selection_score": selection_score({"entropy": entropy_sum / n,
                                            "consistency": consistency_sum / n,
                                            "stability": stability_sum / n}),
        "per_sample_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=None)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    _bundle, resources, _meta, cache, features, _threshold = load_context()
    sample_ids = load_select_sample_ids()
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    cands = list(candidates())
    if args.num_groups > 1:
        cands = [c for i, c in enumerate(cands) if i % args.num_groups == args.group]

    rows = []
    for cand in cands:
        row = evaluate_candidate_with_scores(cand, sample_ids, features, resources, cache.cache_id)
        rows.append(row)
        print(json.dumps({"K": row["K"], "lr": row["lr"], "selection_score": row["selection_score"],
                          "entropy": row["entropy"], "consistency": row["consistency"],
                          "stability": row["stability"], "n_scores": len(row["per_sample_scores"])},
                         ensure_ascii=False), flush=True)

    gid = args.group if args.group is not None else 0
    doc = {
        "schema_version": "0.1.0",
        "diagnostic": "posthoc_per_sample_scores",
        "group": gid,
        "num_groups": args.num_groups,
        "labels_read_note": "labels NOT read by this worker; per-sample scores only",
        "sample_count": len(sample_ids),
        "candidates": rows,
    }
    out = DIAG_RESULTS / f"posthoc_group_{gid}.json"
    write_json_atomic(out, doc)
    print("wrote", out)


if __name__ == "__main__":
    main()
