#!/usr/bin/env python
"""Diagnostic 3: explain why the selection 'consistency' metric is constant.

Consistency (production definition, from ``_common.evaluate_candidate``):
    view_scores = apply_adapter(z, U, R) @ w + b      # 3 adapted probe views
    view_probs  = sigmoid(view_scores)
    votes       = [1 if p > 0.5 else 0]
    majority    = max(#votes==0, #votes==1)
    consistency_sample = majority / 3                 # in {2/3, 1}
    consistency        = mean over samples            # hard-label agreement

It is a HARD-label majority-vote agreement, so it only changes when a view's
hard prediction flips across the 0.5 boundary.  This script additionally
computes diagnostic-only SOFT consistency (probability deltas, KL/JS, view
cosine similarity) and the frozen hard consistency over the full 3178-sample
select set to confirm the constant 0.9544787... value's provenance.
"""
import argparse
import json
import math
import statistics

import numpy as np
import torch

from _diag_common import (DIAG_RESULTS, DIAG_SEED, DIAG_SAMPLE_COUNT, DIAG_CONFIG_SPECS,
                          sigmoid, binary_entropy, load_context, load_select_sample_ids,
                          make_config, run_ep_final, write_json_atomic, ensure_dirs)


def _clamp(p):
    return min(max(p, 1e-12), 1.0 - 1e-12)


def kl_bernoulli(p, q):
    p, q = _clamp(p), _clamp(q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def js_bernoulli(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_bernoulli(p, m) + 0.5 * kl_bernoulli(q, m)


def soft_consistency(probs, adapted_views):
    pairs = [(0, 1), (0, 2), (1, 2)]
    abs_diffs = [abs(probs[i] - probs[j]) for i, j in pairs]
    kls = [kl_bernoulli(probs[i], probs[j]) for i, j in pairs] + \
          [kl_bernoulli(probs[j], probs[i]) for i, j in pairs]
    jss = [js_bernoulli(probs[i], probs[j]) for i, j in pairs]
    cosines = []
    for i, j in pairs:
        a = adapted_views[i] - adapted_views[i].mean()
        b = adapted_views[j] - adapted_views[j].mean()
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        if na.item() < 1e-12 or nb.item() < 1e-12:
            cosines.append(1.0)
        else:
            cosines.append(float((a @ b) / (na * nb)))
    return {
        "mean_abs_prob_diff": float(statistics.mean(abs_diffs)),
        "std_prob": float(statistics.pstdev(probs)),
        "mean_pairwise_kl": float(statistics.mean(kls)),
        "mean_pairwise_js": float(statistics.mean(jss)),
        "mean_pairwise_cosine": float(statistics.mean(cosines)),
    }


def hard_consistency(probs):
    votes = [1 if p > 0.5 else 0 for p in probs]
    return max(votes.count(0), votes.count(1)) / 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    _bundle, resources, _meta, cache, features, _threshold = load_context()

    # --- frozen hard consistency over the FULL 3178 select set (provenance check) ---
    all_ids = load_select_sample_ids()
    frozen_consistency_full = 0.0
    for sid in all_ids:
        z = torch.from_numpy(features[sid])
        view_logits = (z @ resources.w + resources.b).detach().cpu().tolist()
        probs = [sigmoid(v) for v in view_logits]
        frozen_consistency_full += hard_consistency(probs) / len(all_ids)

    # --- 500-sample subset (same deterministic draw as Diag 2) ---
    rng = np.random.default_rng(DIAG_SEED)
    idx = rng.permutation(len(all_ids))[:DIAG_SAMPLE_COUNT]
    sample_ids = [all_ids[i] for i in idx.tolist()]
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    config_results = {}
    for cfg_spec in DIAG_CONFIG_SPECS:
        cfg = make_config(cfg_spec)
        hard_vals, soft_vals, flip_vals = [], [], []
        for sid in sample_ids:
            z = torch.from_numpy(features[sid])
            R, adapted_views, view_logits = run_ep_final(z, resources, cfg)
            probs = [sigmoid(v) for v in view_logits]
            hard_vals.append(hard_consistency(probs))
            soft_vals.append(soft_consistency(probs, adapted_views))
            frozen_probs = [sigmoid(float(v)) for v in (z @ resources.w + resources.b).detach().cpu().tolist()]
            flip_vals.append(int(any((p > 0.5) != (fp > 0.5) for p, fp in zip(probs, frozen_probs))))

        config_results[cfg_spec["key"]] = {
            "K": cfg_spec["K"], "lr": cfg_spec["lr"], "steps": cfg_spec["steps"],
            "hard_consistency_mean": float(statistics.mean(hard_vals)),
            "hard_consistency_std": float(statistics.pstdev(hard_vals)),
            "flip_rate_view_vs_frozen": float(statistics.mean(flip_vals)),
            "flip_count_view_vs_frozen": int(sum(flip_vals)),
            "soft_mean_abs_prob_diff": float(statistics.mean(v["mean_abs_prob_diff"] for v in soft_vals)),
            "soft_mean_std_prob": float(statistics.mean(v["std_prob"] for v in soft_vals)),
            "soft_mean_pairwise_kl": float(statistics.mean(v["mean_pairwise_kl"] for v in soft_vals)),
            "soft_mean_pairwise_js": float(statistics.mean(v["mean_pairwise_js"] for v in soft_vals)),
            "soft_mean_pairwise_cosine": float(statistics.mean(v["mean_pairwise_cosine"] for v in soft_vals)),
        }

    doc = {
        "schema_version": "0.1.0",
        "diagnostic": "consistency_metric_diagnosis",
        "consistency_definition": (
            "hard-label majority-vote agreement of 3 adapted probe views: "
            "mean over samples of max(#votes==0, #votes==1)/3, so per sample it is 2/3 or 1"
        ),
        "is_hard_label_agreement": True,
        "why_constant": (
            "hard agreement changes only when a view's sigmoid crosses 0.5; "
            "adaptation score/logit changes are ~1e-6..1e-5, so flip rate is ~0 and "
            "the majority-vote pattern is identical to frozen for every candidate"
        ),
        "frozen_consistency_full_3178": frozen_consistency_full,
        "reference_constant_from_param_search": 0.9544787077826683,
        "seed": DIAG_SEED,
        "sample_count": len(sample_ids),
        "configs": config_results,
    }

    write_json_atomic(DIAG_RESULTS / "consistency_diagnostics.json", doc)
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    print("wrote", DIAG_RESULTS / "consistency_diagnostics.json")


if __name__ == "__main__":
    main()
