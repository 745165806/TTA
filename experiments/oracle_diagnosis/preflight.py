"""Read-only input checks. Raw target10 labels are deferred until aggregation."""
import argparse
from pathlib import Path

from common import RAW, candidates, config, partition, runtime, sample_ids, validate_output


def preflight(output, asset_root, groups):
    config()
    validate_output(output)
    rows = candidates()
    partitions = [partition(rows, i, groups) for i in range(groups)]
    if sum(map(len, partitions)) != 113:
        raise ValueError("candidate partition incomplete")
    if not RAW.is_file():
        raise FileNotFoundError(RAW)
    ids = sample_ids()  # strict label-free manifest, exactly 3178 unique IDs
    _, _, _, provenance = runtime(asset_root, ids)
    return {"status": "PASS", "candidate_count": len(rows), "target10_count": len(ids),
            "raw_labelled_manifest_exists": True, "raw_count_validation": "deferred_to_posthoc",
            "target10_labels_read": False, "target90_labels_read": False,
            "groups": groups, "group_sizes": list(map(len, partitions)), "provenance": provenance}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--num-groups", type=int, default=4)
    args = p.parse_args()
    print(preflight(args.output, args.asset_root, args.num_groups))
