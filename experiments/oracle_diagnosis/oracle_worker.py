"""Label-free production guarded EP worker; target labels are never loaded here."""
import argparse
import json
import os
import traceback
from pathlib import Path

from common import candidates, partition, runtime, sample_ids, write_json


def run(output, asset_root, group, groups, limit=None):
    import torch
    from experiments.target10_selection.scripts._common import evaluate_candidate, selection_score
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(1)
    if limit not in (None, 32):
        raise ValueError("only limit=32 smoke is supported")
    assigned = partition(candidates(), group, groups)
    ids = sample_ids()
    if limit:
        ids = ids[:limit]
    resources, features, cache_id, provenance = runtime(asset_root, ids)
    rows = []
    for candidate in assigned:
        with (output / "scores" / (candidate["candidate_id"] + ".jsonl")).open("x", encoding="utf-8") as stream:
            def sink(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            row = evaluate_candidate(candidate, ids, features, resources, cache_id,
                                     rho=candidate["rho"] or .2, score_sink=sink)
        row.update(candidate)
        row.update(selection_score=selection_score(row), numeric_failures=0)
        rows.append(row)
        print(candidate["candidate_id"], "complete", len(ids), flush=True)
    write_json(output / f"group_{group}.json", {
        "group": group, "groups": groups, "sample_count": len(ids),
        "target_labels_read": False, "provenance": provenance, "candidates": rows})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--num-groups", type=int, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    try:
        run(args.output, args.asset_root, args.group, args.num_groups, args.limit)
    except Exception:
        write_json(args.output / f"failure_{args.group}.json", {
            "status": "FAIL", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
