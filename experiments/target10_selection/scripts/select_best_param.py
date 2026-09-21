#!/usr/bin/env python
"""Step 3: aggregate per-group search results and pick the best candidate.

Reads every ``results/search_group_*.json`` written by the Step-2 group workers,
recomputes the selection score, picks the argmax (ties break toward smaller K
then smaller lr), and writes:

    results/best_param.json                  # canonical selected parameters
    results/param_search.json                # full aggregated candidate list
    configs/target10_selection_eval.yaml     # fixed params, search forbidden
"""
import json
import sys
from pathlib import Path

import yaml

from _common import (CONFIG_DIR, FROZEN_BUNDLE, RESOURCES, CACHE, RESULTS_DIR, ROOT,
                     RHO, GAMMA, LAMBDA_KEEP, candidates, best_key, selection_score)


def main():
    group_files = sorted(RESULTS_DIR.glob("search_group_*.json"))
    if not group_files:
        print("ERROR: no search_group_*.json files found under results/", file=sys.stderr)
        sys.exit(1)

    rows = []
    seen_groups = []
    for path in group_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("labels_read") is not False:
            print(f"ERROR: {path.name} declares labels_read != false", file=sys.stderr)
            sys.exit(1)
        seen_groups.append(doc.get("group"))
        rows.extend(doc.get("candidates", []))

    expected = list(candidates())
    expected_keys = sorted((c["K"], c["lr"], c["steps"]) for c in expected)
    got_keys = sorted((r["K"], r["lr"], r["steps"]) for r in rows)
    if len(rows) != len(expected) or got_keys != expected_keys:
        print(f"ERROR: aggregated {len(rows)} candidates, expected {len(expected)}", file=sys.stderr)
        print("expected:", expected_keys, file=sys.stderr)
        print("got:     ", got_keys, file=sys.stderr)
        sys.exit(1)

    for r in rows:
        r["selection_score"] = selection_score(r)

    best = max(rows, key=best_key)
    print("groups aggregated:", seen_groups)
    print("BEST:", json.dumps(best, ensure_ascii=False), flush=True)

    best_doc = {
        "schema_version": "0.1.0",
        "experiment": "target10-unsupervised-select",
        "selection_data": "target10",
        "selection_rule": "argmax mean(normalized_entropy_reduction, consistency, stability); "
                          "entropy normalized by ln2; tie-break smaller K then smaller lr",
        "labels_read": False,
        "selected": best,
        "candidate_count": len(rows),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "best_param.json").write_text(
        json.dumps(best_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    full_doc = {
        "schema_version": "0.1.0",
        "experiment": "target10-unsupervised-select",
        "selection_data": "target10",
        "seed": 2026,
        "selection_metrics": ["entropy_reduction", "prediction_consistency", "confidence_stability"],
        "selection_rule": best_doc["selection_rule"],
        "labels_read": False,
        "candidates": rows,
        "selected": best,
    }
    (RESULTS_DIR / "param_search.json").write_text(
        json.dumps(full_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config = {
        "schema_version": "0.1.0",
        "experiment": "target10-unsupervised-select",
        "selection_data": "target10",
        "selection_complete": True,
        "search_forbidden": True,
        "selected_K": best["K"],
        "selected_lr": best["lr"],
        "selected_steps": best["steps"],
        "selection_metrics": {
            "entropy_reduction": best["entropy"],
            "prediction_consistency": best["consistency"],
            "confidence_stability": best["stability"],
        },
        "fixed_method": {
            "method_id": "ep_tta",
            "config": {
                "steps": best["steps"],
                "lr": best["lr"] if best["lr"] is not None else 0.01,
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

    print("wrote", RESULTS_DIR / "best_param.json")
    print("wrote", RESULTS_DIR / "param_search.json")
    print("wrote", CONFIG_DIR / "target10_selection_eval.yaml")


if __name__ == "__main__":
    main()
