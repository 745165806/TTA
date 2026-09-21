#!/usr/bin/env python
"""Step 4: evaluate the fixed selected parameters on the 90% target subset.

Reads ``results/best_param.json`` (written by ``select_best_param.py``; search is
forbidden), scores target90 with that fixed method, and writes
``results/target90_result.json``.  This stage is allowed to read target90 labels.

Metrics (threshold = frozen source cal0 tau0):
    EER, accuracy, AUC (AUROC) computed exactly;
    minDCF = null (requires ASV scores, unavailable in this pipeline).
"""
import argparse
import json

from _common import (BEST_PARAM, RESULTS_DIR, RHO, GAMMA, LAMBDA_KEEP, METHOD_ID, PROTOCOL_ID,
                     EPConfig, compute_metrics, load_context, read_target90_records,
                     score_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="debug only: evaluate on the first N samples")
    args = parser.parse_args()

    best_doc = json.loads(BEST_PARAM.read_text(encoding="utf-8"))
    if best_doc.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("best_param.json is stale or from another selection protocol")
    best = best_doc["selected"]
    _bundle, resources, _meta, cache, features, threshold = load_context()
    sample_ids, labels = read_target90_records(args.limit)

    cfg = EPConfig(steps=best["steps"], lr=best["lr"] if best["lr"] is not None else 0.003,
                   rho=RHO, gamma=GAMMA, lambda_keep=LAMBDA_KEEP)
    scores = score_dataset(
        sample_ids, features, resources, cache.cache_id, cfg, method_id=METHOD_ID)

    result = {
        "schema_version": "0.1.0",
        "protocol_id": PROTOCOL_ID,
        "method": "target10-unsupervised-select guarded EP",
        "method_id": METHOD_ID,
        "selection_data": "target10",
        "K": best["K"],
        "lr": best["lr"],
        "steps": best["steps"],
        "test_data": "target90",
        "test_count": len(sample_ids),
        "labels_read": True,
        "metrics": compute_metrics(scores, labels, threshold),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "target90_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("wrote", RESULTS_DIR / "target90_result.json")


if __name__ == "__main__":
    main()
