#!/usr/bin/env python
"""Create the fixed label-free calU/evalU split from sample_id only."""
import argparse
import hashlib
import json
import math
from pathlib import Path


FORBIDDEN = frozenset({"label", "canonical_label", "original_label", "target", "class", "y"})


def split_ids(sample_ids, salt, fraction=0.10):
    ids = list(sample_ids)
    if not ids or len(ids) != len(set(ids)) or not all(type(x) is str and x for x in ids):
        raise ValueError("sample IDs must be nonempty and unique")
    if type(salt) is not str or not salt or not 0 < fraction < 1:
        raise ValueError("invalid split rule")
    ordered = sorted(ids, key=lambda sid: (hashlib.sha256(
        (salt + "\0" + sid).encode("utf-8")).hexdigest(), sid))
    n_cal = max(1, min(len(ids) - 1, int(math.floor(len(ids) * fraction + 0.5))))
    return ordered[:n_cal], ordered[n_cal:]


def read_label_free_ids(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != doc.get("count"):
        raise ValueError("input manifest count mismatch")
    ids = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("manifest record must be an object")
        for key in record:
            if key.lower() in FORBIDDEN or "label" in key.lower():
                raise ValueError("label field forbidden in split input: %s" % key)
        ids.append(record["sample_id"])
    return doc.get("dataset_id"), ids


def manifest(dataset_id, role, ids, salt, fraction):
    return {
        "schema_version": "0.1.0", "dataset_id": dataset_id,
        "role": role, "protocol": "unlabeled_domain_calibration_then_episodic_tta",
        "split_rule": {"algorithm": "sha256_order_of_salt_nul_sample_id",
                       "salt": salt, "calibration_fraction": fraction},
        "count": len(ids),
        "records": [{"sample_id": sid, "split_role": role} for sid in ids],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    args = parser.parse_args()
    dataset_id, ids = read_label_free_ids(args.input)
    cal_ids, eval_ids = split_ids(ids, args.salt, args.fraction)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        out / "target10_calU.json": manifest(dataset_id, "calU", cal_ids, args.salt, args.fraction),
        out / "target10_evalU.json": manifest(dataset_id, "evalU", eval_ids, args.salt, args.fraction),
    }
    for path, doc in targets.items():
        text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise SystemExit("refusing to replace a different fixed split: %s" % path)
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"calU": len(cal_ids), "evalU": len(eval_ids), "salt": args.salt}))


if __name__ == "__main__":
    main()
