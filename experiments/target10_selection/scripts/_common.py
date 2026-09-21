"""Shared helpers for the target10_selection pipeline.

Kept as a private module inside the experiment directory so the individual
scripts stay small and use the exact same production adaptation path
(``run_method("ep_tta", ...)``) as the project's ``run-tta``.
"""
import json
import math
from pathlib import Path

import torch

from eptta.adaptation.math import apply_adapter
from eptta.adaptation.types import EPConfig, TargetViews
from eptta.baselines.dispatch import run_method
from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/target10_selection"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_DIR = EXP_DIR / "configs"
MANIFEST_DIR = EXP_DIR / "manifests"

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
RESOURCES = ROOT / "outputs_v2/ssl_aasist/resources"
CACHE = ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"
TARGET10 = MANIFEST_DIR / "inwild_target10.json"                    # raw audit manifest (labels present)
TARGET10_SELECT = MANIFEST_DIR / "inwild_target10_select.json"      # label-free select manifest
TARGET90 = MANIFEST_DIR / "inwild_target90.json"
BEST_PARAM = RESULTS_DIR / "best_param.json"

# Label/leakage keys forbidden inside the label-free select manifest.
FORBIDDEN_LABEL_KEYS = frozenset({
    "label", "canonical_label", "original_label", "target", "class", "y", "source_labels",
})

KS = [0, 1, 3, 5, 10]
LRS = [1e-5, 5e-5, 1e-4]
RHO, GAMMA, LAMBDA_KEEP = 0.2, 0.1, 1.0  # fixed EP constants (existing config)

# Existing source-select EP choice (outputs_v2/ssl_aasist/selection.json).
BASELINE_METHOD = "source-select EP"
BASELINE_STEPS = 1
BASELINE_LR = 0.003


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def binary_entropy(p):
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def candidates():
    """K == steps in this codebase (docs/DESIGN.md: "K 步后对原输入视图评分").

    K=0 is the frozen identity path (lr/steps irrelevant) and yields a single
    candidate; K in {1,3,5,10} cross the three learning rates.
    """
    for k in KS:
        if k == 0:
            yield {"K": 0, "lr": None, "steps": 0}
        else:
            for lr in LRS:
                yield {"K": k, "lr": lr, "steps": k}


def selection_score(row):
    return (row["entropy"] / math.log(2.0) + row["consistency"] + row["stability"]) / 3.0


def best_key(row):
    return (row["selection_score"], -row["K"], -(row["lr"] if row["lr"] is not None else 0.0))


def load_context():
    bundle, _manifest, _parity, _selection = verify_frozen_export(FROZEN_BUNDLE)
    resources, _extras, resource_meta = load_frozen_resources(RESOURCES, bundle)
    threshold = float(resource_meta["scalars"]["tau0"])
    cache = FeatureCache(CACHE)
    features = cache.load_by_id()
    return bundle, resources, resource_meta, cache, features, threshold


def evaluate_candidate(cand, sample_ids, features, resources, cache_id):
    """Per-sample episodic TTA over ``sample_ids`` and label-free aggregate metrics."""
    cfg = EPConfig(steps=cand["K"], lr=cand["lr"] if cand["lr"] is not None else 1e-5,
                   rho=RHO, gamma=GAMMA, lambda_keep=LAMBDA_KEEP)
    entropy_sum = consistency_sum = stability_sum = 0.0
    n = 0
    for sample_id in sample_ids:
        z = torch.from_numpy(features[sample_id])
        target = TargetViews(sample_id, z, cache_id)
        result = run_method("ep_tta", target, resources, cfg, {})

        p_before = sigmoid(result["score_before"])
        p_after = sigmoid(result["score"])
        entropy_sum += binary_entropy(p_before) - binary_entropy(p_after)

        with torch.no_grad():
            R = result["R"].to(z.dtype).to(z.device)
            view_scores = apply_adapter(z, resources.U, R) @ resources.w + resources.b
        view_probs = [sigmoid(s) for s in view_scores.detach().cpu().tolist()]
        votes = [1 if p > 0.5 else 0 for p in view_probs]
        majority = max(votes.count(0), votes.count(1))
        consistency_sum += majority / 3.0
        confidences = [abs(p - 0.5) for p in view_probs]
        mean_c = sum(confidences) / 3.0
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in confidences) / 3.0)
        stability_sum += max(0.0, min(1.0, 1.0 - std_c / 0.5))
        n += 1

    return {
        "K": cand["K"],
        "lr": cand["lr"],
        "steps": cand["steps"],
        "entropy": entropy_sum / n,
        "consistency": consistency_sum / n,
        "stability": stability_sum / n,
    }


def score_dataset(sample_ids, features, resources, cache_id, cfg):
    scores = {}
    for sid in sample_ids:
        z = torch.from_numpy(features[sid])
        target = TargetViews(sid, z, cache_id)
        result = run_method("ep_tta", target, resources, cfg, {})
        scores[sid] = float(result["score"])
    return scores


def compute_metrics(scores, labels, threshold):
    order = sorted(scores)
    m = binary_metrics([scores[s] for s in order], [labels[s] for s in order], threshold)
    accuracy = (m["tp"] + m["tn"]) / m["count"]
    return {
        "EER": m["eer"],
        "minDCF": None,
        "minDCF_note": "t-DCF/minDCF require ASV scores, unavailable in this pipeline",
        "accuracy": accuracy,
        "AUC": m["auroc"],
        "count": m["count"],
        "threshold": threshold,
        "balanced_accuracy": m["balanced_accuracy"],
        "tpr": m["tpr"],
        "fpr": m["fpr"],
        "fnr": m["fnr"],
    }


def read_target90_records(limit=None):
    manifest = json.loads(TARGET90.read_text(encoding="utf-8"))
    records = manifest["records"]
    if limit:
        records = records[:limit]
    sample_ids = [r["sample_id"] for r in records]
    labels = {r["sample_id"]: int(r["label"]) for r in records}
    return sample_ids, labels


def load_select_sample_ids():
    """Load the label-free select manifest and return its sample_ids.

    Re-validates the exact TargetInputManifest permission contract and the
    absence of any label field / source_labels before returning, so the search
    stage can never silently consume a labelled manifest.
    """
    from eptta.data.permissions import TargetInputManifest
    from eptta.errors import PermissionDenied

    doc = json.loads(TARGET10_SELECT.read_text(encoding="utf-8"))
    if doc.get("role") != "select":
        raise PermissionDenied("select manifest role must be 'select'")
    if "source_labels" in doc:
        raise PermissionDenied("select manifest must not contain source_labels")
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != doc.get("count") or len(records) != 3178:
        raise PermissionDenied("select manifest count must equal 3178")
    sample_ids = []
    for record in records:
        TargetInputManifest.from_dict(record)  # exact 6 fields, split_role in ROLES
        if record.get("split_role") != "select":
            raise PermissionDenied("select record split_role must be 'select'")
        for key in record:
            if key.lower() in FORBIDDEN_LABEL_KEYS or "label" in key.lower():
                raise PermissionDenied("label field present in select record: %s" % key)
        sample_ids.append(record["sample_id"])
    if len(sample_ids) != len(set(sample_ids)):
        raise PermissionDenied("select manifest sample_ids are not unique")
    return sample_ids
