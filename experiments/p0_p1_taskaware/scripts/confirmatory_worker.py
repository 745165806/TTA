#!/usr/bin/env python
"""P1 confirmatory worker: score one frozen dataset with taskaware_full (locked params).

Label-free: reads only inference manifests (sample IDs) and the feature cache.
Frozen and adapted scores are both emitted from the single adaptation pass
(``score_before`` is the frozen original-view score).
"""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

import torch

from eptta.adaptation.types import EPConfig
from eptta.baselines.dispatch import run_method
from eptta.cache.reader import FeatureCache
from eptta.data.io import iter_jsonl
from eptta.offline.artifacts import load_frozen_resources
from eptta.models.frozen import verify_frozen_export
from scripts.common import FROZEN_BUNDLE, RESOURCES

torch.set_num_threads(1)

LOCKED_CONFIG = dict(steps=3, lr=0.03, rho=0.05, gamma=0.1, lambda_keep=1.0)
LOCKED_PARAMS = {"lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 1.0,
                 "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.5,
                 "min_agreement": 1.0}

DATASETS = {
    "itw_target90": {
        "cache": "outputs_v2/ssl_aasist/cache-target-in_the_wild",
        "inference_manifest": "data/manifests_v2/in_the_wild/inference/target_test.jsonl",
        "labels_manifest": "data/manifests_v2/in_the_wild/labels/target_test.jsonl",
        "exclude_select": "experiments/target10_selection/manifests/inwild_target10_select.json",
    },
    "asv2021_la": {
        "cache": "outputs_v2/ssl_aasist/cache-target-asv2021_la_eval",
        "inference_manifest": "data/manifests_v2/asv2021_la_eval/inference/target_test.jsonl",
        "labels_manifest": "data/manifests_v2/asv2021_la_eval/labels/target_test.jsonl",
    },
    "asv2021_df": {
        "cache": "outputs_v2/ssl_aasist/cache-target-asv2021_df_eval",
        "inference_manifest": "data/manifests_v2/asv2021_df_eval/inference/target_test.jsonl",
        "labels_manifest": "data/manifests_v2/asv2021_df_eval/labels/target_test.jsonl",
    },
    "control_test": {
        "cache": "outputs_v2/ssl_aasist/cache-control_test",
        "inference_manifest": "data/manifests_v2/asv2019_la/inference/control_test.jsonl",
        "labels_manifest": "data/manifests_v2/asv2019_la/labels/control_test.jsonl",
    },
}


def load_sample_ids(dataset_spec):
    ids = [row["sample_id"] for row in iter_jsonl(ROOT / dataset_spec["inference_manifest"])]
    if dataset_spec.get("exclude_select"):
        exclude = {row["sample_id"] for row in iter_jsonl(ROOT / dataset_spec["exclude_select"])}
        ids = [sid for sid in ids if sid not in exclude]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate sample IDs in inference manifest")
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    sample_ids = load_sample_ids(spec)

    bundle, _m, _p, _s = verify_frozen_export(FROZEN_BUNDLE)
    resources, _extras, _meta = load_frozen_resources(RESOURCES, bundle)
    cache = FeatureCache(ROOT / spec["cache"])
    features = cache.load_by_id()

    from eptta.adaptation.types import TargetViews
    cfg = EPConfig(**LOCKED_CONFIG)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("confirm_%s.jsonl" % args.dataset)
    if out_path.exists():
        raise SystemExit("refusing to overwrite: %s" % out_path)

    written = 0
    with out_path.open("w", encoding="utf-8") as stream:
        for sample_id in sample_ids:
            target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache.cache_id)
            result = run_method("ep_tta_taskaware_v1", target, resources, cfg, LOCKED_PARAMS)
            record = {
                "dataset": args.dataset, "sample_id": sample_id,
                "score_before": float(result["score_before"]),
                "score_after": float(result["score"]),
                "delta_score": float(result["score"] - result["score_before"]),
                "adaptation_applied": bool(result.get("adaptation_applied", False)),
                "abstain_reason": result.get("abstain_reason"),
                "source_anchor_flip_count": result.get("source_anchor_flip_count"),
                "safety_rejected": bool(result.get("safety_rejected", False)),
                "final_R_norm": float(result.get("final_R_norm", 0.0)),
                "steps_completed": int(result.get("steps_completed", 0)),
                "numeric_fallback": result.get("status") != "ok",
            }
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            written += 1
    print("confirm %s: wrote %s (%d records)" % (args.dataset, out_path, written))


if __name__ == "__main__":
    main()
