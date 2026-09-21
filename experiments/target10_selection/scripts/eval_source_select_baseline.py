#!/usr/bin/env python
"""Step 5: evaluate the existing source-select EP baseline on the 90% target subset.

Uses the source-select EP choice recorded in outputs_v2/ssl_aasist/selection.json
(steps=1, lr=0.003, rho=0.2) and writes ``results/baseline_source_select.json``.
This stage is allowed to read target90 labels.
"""
import argparse
import json

from _common import (BASELINE_METHOD, BASELINE_STEPS, BASELINE_LR, RESULTS_DIR,
                     RHO, GAMMA, LAMBDA_KEEP, EPConfig, compute_metrics, load_context,
                     read_target90_records, score_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="debug only: evaluate on the first N samples")
    args = parser.parse_args()

    _bundle, resources, _meta, cache, features, threshold = load_context()
    sample_ids, labels = read_target90_records(args.limit)

    cfg = EPConfig(steps=BASELINE_STEPS, lr=BASELINE_LR, rho=RHO, gamma=GAMMA,
                   lambda_keep=LAMBDA_KEEP)
    scores = score_dataset(
        sample_ids, features, resources, cache.cache_id, cfg, method_id="ep_tta")

    result = {
        "schema_version": "0.1.0",
        "method": BASELINE_METHOD,
        "method_id": "ep_tta",
        "selection_data": "source-select",
        "K": BASELINE_STEPS,
        "lr": BASELINE_LR,
        "steps": BASELINE_STEPS,
        "test_data": "target90",
        "test_count": len(sample_ids),
        "labels_read": True,
        "target_labels_read": True,
        "source_labels_used": True,
        "metrics": compute_metrics(scores, labels, threshold),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline_source_select.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("wrote", RESULTS_DIR / "baseline_source_select.json")


if __name__ == "__main__":
    main()
