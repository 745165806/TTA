#!/usr/bin/env python
"""Fixed source-select safety audit for candidate MEMO audio views."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

VIEW_INDEX = {"noise_snr30": 1, "fir_side_gain0.05": 2}


def default_asset_root():
    return Path(os.environ.get("TTA_ASSET_ROOT", ROOT.parent / "TTA")).resolve()


def read_source_labels(path):
    labels = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"schema_version", "sample_id", "canonical_label"}:
                raise ValueError("source-select label sidecar has unsafe fields")
            if row["canonical_label"] not in (0, 1) or row["sample_id"] in labels:
                raise ValueError("invalid/duplicate source-select label")
            labels[row["sample_id"]] = row["canonical_label"]
    return labels


def rankdata(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return np.asarray(ranks, dtype=np.float64)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def run_audit(asset_root, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["augmentation_audit"]
    if spec["source_role"] != "select":
        raise ValueError("augmentation audit is restricted to source select")
    bundle_path = asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, _m, _p, _s = verify_frozen_export(bundle_path)
    resources, _extras, _meta = load_frozen_resources(
        asset_root / "outputs_v2/ssl_aasist/resources", bundle)
    cache = FeatureCache(asset_root / "outputs_v2/ssl_aasist/cache-select").load_by_id()
    labels = read_source_labels(
        asset_root / "data/manifests_v2/asv2019_la/labels/select.jsonl")
    if set(cache) != set(labels):
        raise ValueError("source-select cache/label coverage mismatch")
    ids = sorted(cache)
    y = [labels[sid] for sid in ids]
    w = resources.w.detach().cpu().numpy()
    b = float(resources.b)
    base = [float(cache[sid][0] @ w + b) for sid in ids]
    tau0 = float(resources.tau0)
    baseline = binary_metrics(base, y, tau0)
    limits = spec["acceptance"]
    rows = []
    for candidate in spec["candidates"]:
        if candidate not in VIEW_INDEX:
            raise ValueError("candidate has no frozen source-select view: %s" % candidate)
        scores = [float(cache[sid][VIEW_INDEX[candidate]] @ w + b) for sid in ids]
        metrics = binary_metrics(scores, y, tau0)
        flip_rate = sum((a > tau0) != (z > tau0) for a, z in zip(base, scores)) / len(ids)
        row = {
            "candidate": candidate,
            "source_role": "select",
            "count": len(ids),
            "EER": metrics["eer"],
            "AUC": metrics["auroc"],
            "prediction_flip_rate": flip_rate,
            "score_rank_correlation": spearman(base, scores),
            "mean_abs_score_change": float(np.mean(np.abs(np.asarray(scores) - np.asarray(base)))),
        }
        checks = {
            "eer": row["EER"] - baseline["eer"] <= limits["max_eer_increase"],
            "auc": baseline["auroc"] - row["AUC"] <= limits["max_auc_decrease"],
            "flip": row["prediction_flip_rate"] <= limits["max_prediction_flip_rate_at_tau0"],
            "rank": row["score_rank_correlation"] >= limits["min_score_rank_correlation"],
            "score_change": row["mean_abs_score_change"] <= limits["max_mean_abs_score_change"],
        }
        row["checks"] = checks
        row["accepted"] = all(checks.values())
        rows.append(row)
    return {
        "schema_version": "0.1.0",
        "source_only": True,
        "source_role": "select",
        "target_labels_read": False,
        "baseline": {"EER": baseline["eer"], "AUC": baseline["auroc"], "tau0": tau0},
        "acceptance": limits,
        "candidates": rows,
        "accepted_augmentations": [row["candidate"] for row in rows if row["accepted"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--config", type=Path,
                        default=ROOT / "experiments/audio_native_tta/config.json")
    parser.add_argument("--output-json", type=Path,
                        default=ROOT / "experiments/audio_native_tta/augmentation_audit.json")
    parser.add_argument("--output-csv", type=Path,
                        default=ROOT / "experiments/audio_native_tta/augmentation_audit.csv")
    args = parser.parse_args()
    result = run_audit(args.asset_root, args.config)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = ["candidate", "source_role", "count", "EER", "AUC", "prediction_flip_rate",
              "score_rank_correlation", "mean_abs_score_change", "accepted"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["candidates"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
