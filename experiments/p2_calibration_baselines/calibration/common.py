"""Shared helpers for P2-A/P2-B calibration and oracle diagnostics.

Path A (calibration) runs entirely on CPU over the existing frozen feature
caches; it never loads the SSL-AASIST model.  Target labels are read ONLY by
aggregate/oracle scripts, and every supervised output is marked
POST_HOC_DEVELOPMENT_ONLY.
"""
import json
from pathlib import Path

import torch

from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/p2_calibration_baselines"
TARGET10_EXP = ROOT / "experiments/target10_selection"

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
RESOURCES = ROOT / "outputs_v2/ssl_aasist/resources"
CACHE_CAL0 = ROOT / "outputs_v2/ssl_aasist/cache-cal0"
CACHE_TARGET = ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"

CAL0_LABELS = ROOT / "data/manifests_v2/asv2019_la/labels/cal0.jsonl"
CAL0_MANIFEST = ROOT / "data/manifests_v2/asv2019_la/inference/cal0.jsonl"

TARGET10_SELECT = TARGET10_EXP / "manifests/inwild_target10_select.json"
TARGET10_LABELS = TARGET10_EXP / "manifests/inwild_target10.json"

POST_HOC_MARKER = "POST_HOC_DEVELOPMENT_ONLY"


def load_resources():
    """Load frozen w/b/tau0 plus U/anchor tensors (CPU)."""
    bundle, _m, _p, _s = verify_frozen_export(FROZEN_BUNDLE)
    resources, _extras, resource_meta = load_frozen_resources(RESOURCES, bundle)
    return bundle, resources, resource_meta


def _load_scores_labels(cache_ref, label_ref, manifest_ref=None):
    """Frozen original-view score (z0 @ w + b) aligned with canonical labels."""
    bundle, resources, resource_meta = load_resources()
    cache = FeatureCache(cache_ref)
    features = cache.load_by_id()
    labels = {}
    with open(label_ref, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[row["sample_id"]] = int(row["canonical_label"])
    ids = sorted(labels)
    scores, ys = [], []
    for sid in ids:
        z0 = torch.from_numpy(features[sid][0])
        score = float(z0 @ resources.w + resources.b)
        scores.append(score)
        ys.append(labels[sid])
    return scores, ys, ids, resources


def load_cal0():
    """Source cal0 frozen scores + labels."""
    scores, ys, ids, resources = _load_scores_labels(CACHE_CAL0, CAL0_LABELS, CAL0_MANIFEST)
    return scores, ys, ids, resources


def load_target10():
    """Target10 development frozen scores + labels (POST_HOC only)."""
    select_doc = json.loads(TARGET10_SELECT.read_text(encoding="utf-8"))
    label_doc = json.loads(TARGET10_LABELS.read_text(encoding="utf-8"))
    labels = {r["sample_id"]: int(r["label"]) for r in label_doc["records"]}
    sample_ids = [r["sample_id"] for r in select_doc["records"]]
    bundle, resources, resource_meta = load_resources()
    cache = FeatureCache(CACHE_TARGET)
    features = cache.load_by_id()
    scores, ys = [], []
    for sid in sample_ids:
        z0 = torch.from_numpy(features[sid][0])
        score = float(z0 @ resources.w + resources.b)
        scores.append(score)
        ys.append(labels[sid])
    return scores, ys, sample_ids, resources


def binary_metrics(scores, labels, threshold):
    from eptta.evaluation.metrics import binary_metrics as _bm
    return _bm(list(scores), list(labels), threshold)


def quantiles(values):
    import statistics
    values = sorted(values)
    n = len(values)
    def q(p):
        if n == 0:
            return None
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return values[idx]
    return {
        "count": n,
        "mean": statistics.fmean(values) if n else None,
        "std": statistics.stdev(values) if n > 1 else None,
        "median": statistics.median(values) if n else None,
        "q01": q(0.01), "q05": q(0.05), "q25": q(0.25), "q50": q(0.50),
        "q75": q(0.75), "q95": q(0.95), "q99": q(0.99),
    }
