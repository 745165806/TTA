#!/usr/bin/env python
"""Step 2: label-free parameter selection on the 10% target subset.

Supports multi-GPU splitting via ``--group N --num-groups M``: the flat candidate
list is distributed round-robin across groups, and each group writes only its own
partial results to ``results/search_group_N.json`` (no best selection here --
``select_best_param.py`` aggregates).  Without ``--group`` the process runs the
full grid in one process and writes ``results/param_search.json`` plus the fixed
config (useful for a single-device run).

Selection metrics (label-free only):
    view_reduction          = relative reduction of the EP view objective
    probability_consistency = 1 - std(sigmoid(view_scores)) / 0.5
    source_margin_retention = mean clipped source-margin retention ratio

``source_safety`` remains a hard feasibility diagnostic, not a score term.

No target label is read from the manifest.
"""
import argparse
import json
import sys

import yaml

from _common import (CONFIG_DIR, FROZEN_BUNDLE, RESOURCES, CACHE, TARGET10_SELECT, RESULTS_DIR,
                     RHO, GAMMA, LAMBDA_KEEP, METHOD_ID, PROTOCOL_ID, candidates,
                     evaluate_candidate, load_context, load_select_sample_ids,
                     selection_score, best_key, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=None,
                        help="0-based group id; when set, only this group's candidates run")
    parser.add_argument("--num-groups", type=int, default=4,
                        help="number of groups to split candidates across")
    parser.add_argument("--limit", type=int, default=None,
                        help="debug only: evaluate on the first N samples")
    args = parser.parse_args()

    if args.num_groups < 1:
        print("ERROR: --num-groups must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.group is not None and not (0 <= args.group < args.num_groups):
        print(f"ERROR: --group must be in [0, {args.num_groups})", file=sys.stderr)
        sys.exit(2)

    _bundle, resources, _meta, cache, features, _threshold = load_context()
    # Only the label-free select manifest may feed the search stage.
    sample_ids = load_select_sample_ids()
    if args.limit:
        sample_ids = sample_ids[: args.limit]

    cands = list(candidates())
    if args.group is not None:
        cands = [c for i, c in enumerate(cands) if i % args.num_groups == args.group]

    rows = []
    for cand in cands:
        row = evaluate_candidate(cand, sample_ids, features, resources, cache.cache_id)
        row["selection_score"] = selection_score(row)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.group is not None:
        doc = {
            "schema_version": "0.1.0",
            "protocol_id": PROTOCOL_ID,
            "group": args.group,
            "num_groups": args.num_groups,
            "method_id": METHOD_ID,
            "selection_data": "target10",
            "labels_read": False,
            "target_labels_read": False,
            "source_labels_used": True,
            "numeric_fallback_policy": "fail",
            "candidates": rows,
        }
        out = RESULTS_DIR / f"search_group_{args.group}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(rows)} candidates)")
        return

    best = max(rows, key=best_key)
    search_doc = {
        "schema_version": "0.1.0",
        "protocol_id": PROTOCOL_ID,
        "experiment": "target10-unsupervised-select",
        "method_id": METHOD_ID,
        "selection_data": "target10",
        "seed": 2026,
        "target10_manifest": str(TARGET10_SELECT.relative_to(ROOT)),
        "frozen_bundle": str(FROZEN_BUNDLE.relative_to(ROOT)),
        "resources": str(RESOURCES.relative_to(ROOT)),
        "feature_cache": str(CACHE.relative_to(ROOT)),
        "selection_metrics": ["view_reduction", "probability_consistency",
                              "source_margin_retention"],
        "selection_rule": "argmax mean(view_reduction, probability_consistency, "
                          "source_margin_retention); "
                          "tie-break smaller K then smaller lr",
        "labels_read": False,
        "target_labels_read": False,
        "source_labels_used": True,
        "numeric_fallback_policy": "fail",
        "candidates": rows,
        "selected": best,
    }
    (RESULTS_DIR / "param_search.json").write_text(
        json.dumps(search_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_config(best)
    print("wrote", RESULTS_DIR / "param_search.json")
    print("wrote", CONFIG_DIR / "target10_selection_eval.yaml")


def _write_config(best):
    config = {
        "schema_version": "0.1.0",
        "protocol_id": PROTOCOL_ID,
        "experiment": "target10-unsupervised-select",
        "method_id": METHOD_ID,
        "selection_data": "target10",
        "selection_complete": True,
        "search_forbidden": True,
        "labels_read": False,
        "target_labels_read": False,
        "source_labels_used": True,
        "numeric_fallback_policy": "fail",
        "selected_K": best["K"],
        "selected_lr": best["lr"],
        "selected_steps": best["steps"],
        "selection_metrics": {
            "view_reduction": best["view_reduction"],
            "probability_consistency": best["probability_consistency"],
            "source_margin_retention": best["source_margin_retention"],
        },
        "fixed_method": {
            "method_id": METHOD_ID,
            "config": {
                "steps": best["steps"],
                "lr": best["lr"] if best["lr"] is not None else 0.003,
                "rho": RHO,
                "gamma": GAMMA,
                "lambda_keep": LAMBDA_KEEP,
            },
        },
        "target90_manifest": "experiments/target10_selection/manifests/inwild_target90.json",
        "frozen_bundle": str(FROZEN_BUNDLE.relative_to(ROOT)),
        "resources": str(RESOURCES.relative_to(ROOT)),
        "feature_cache": str(CACHE.relative_to(ROOT)),
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "target10_selection_eval.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
