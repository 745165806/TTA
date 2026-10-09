"""Shared helpers for the target10_selection pipeline.

Kept as a private module inside the experiment directory so the individual
scripts stay small and use the exact same production adaptation path
(``run_method("ep_tta_guarded", ...)``) as the project's ``run-tta``.
"""
import json
from pathlib import Path

import torch

from eptta.adaptation.math import apply_adapter, margin_deficit, margin_tolerance, view_loss
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

METHOD_ID = "ep_tta_guarded"
PROTOCOL_ID = "target10-guarded-v2"
KS = [0, 1, 3, 5, 10]
LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
RHO, GAMMA, LAMBDA_KEEP = 0.2, 0.1, 1.0  # fixed EP constants (existing config)

# Existing source-select EP choice (outputs_v2/ssl_aasist/selection.json).
BASELINE_METHOD = "source-select EP"
BASELINE_STEPS = 1
BASELINE_LR = 0.003


def candidates():
    """K == steps in this codebase (docs/DESIGN.md: "K 步后对原输入视图评分").

    K=0 is the frozen identity path (lr/steps irrelevant) and yields a single
    candidate; K in {1,3,5,10} cross the six learning rates.
    """
    for k in KS:
        if k == 0:
            yield {"K": 0, "lr": None, "steps": 0}
        else:
            for lr in LRS:
                yield {"K": k, "lr": lr, "steps": k}


def selection_score(row):
    return (
        row["view_reduction"]
        + row["probability_consistency"]
        + row["source_margin_retention"]
    ) / 3.0


def best_key(row):
    return (row["selection_score"], -row["K"], -(row["lr"] if row["lr"] is not None else 0.0))


def load_context():
    bundle, _manifest, _parity, _selection = verify_frozen_export(FROZEN_BUNDLE)
    resources, _extras, resource_meta = load_frozen_resources(RESOURCES, bundle)
    threshold = float(resource_meta["scalars"]["tau0"])
    cache = FeatureCache(CACHE)
    features = cache.load_by_id()
    return bundle, resources, resource_meta, cache, features, threshold


def require_success(result, sample_id, method_id):
    """Reject per-sample numerical fallback in selection and final evaluation."""
    status = result.get("status")
    if status != "ok":
        error_type = result.get("error_type", "unknown")
        error_message = result.get("error_message", "no error message")
        raise RuntimeError(
            f"{method_id} failed for sample {sample_id}: "
            f"status={status}, {error_type}: {error_message}"
        )
    return result


def evaluate_candidate(cand, sample_ids, features, resources, cache_id, *,
                       rho=RHO, score_sink=None):
    """Run guarded EP and aggregate mechanism-aligned, label-free metrics."""
    cfg = EPConfig(steps=cand["K"], lr=cand["lr"] if cand["lr"] is not None else 0.003,
                   rho=rho, gamma=GAMMA, lambda_keep=LAMBDA_KEEP)
    view_reduction_sum = probability_consistency_sum = 0.0
    source_safety_sum = source_margin_retention_sum = 0.0
    r_norm_sum = abs_delta_score_sum = 0.0
    guard_count = guard_backtracks = guard_reverts = guard_steps = 0
    n = 0
    for sample_id in sample_ids:
        z = torch.from_numpy(features[sample_id])
        target = TargetViews(sample_id, z, cache_id)
        result = require_success(
            run_method(METHOD_ID, target, resources, cfg, {}), sample_id, METHOD_ID)
        if score_sink is not None:
            score_sink({"sample_id": sample_id, "score": float(result["score"]),
                        "score_before": float(result["score_before"])})

        with torch.no_grad():
            R = result["R"].to(z.dtype).to(z.device)
            adapted_views = apply_adapter(z, resources.U, R)

            before_view = float(view_loss(z))
            after_view = float(view_loss(adapted_views))
            relative_reduction = (before_view - after_view) / max(before_view, 1e-12)
            view_reduction_sum += max(-1.0, min(1.0, relative_reduction))

            view_probs = torch.sigmoid(adapted_views @ resources.w + resources.b)
            probability_consistency = 1.0 - float(view_probs.std(unbiased=False)) / 0.5
            probability_consistency_sum += max(
                0.0, min(1.0, probability_consistency))

            adapted_anchors = apply_adapter(resources.anchors_z, resources.U, R)
            signs = 2 * resources.anchors_y.to(adapted_anchors.dtype) - 1
            margins_after = signs * (
                adapted_anchors @ resources.w + resources.b - resources.tau0)
            retention = (margins_after / resources.anchors_m0).clamp(min=0.0, max=1.0)
            source_margin_retention_sum += float(retention.mean())
            deficit = margin_deficit(
                adapted_anchors,
                resources.w,
                resources.b,
                resources.anchors_y,
                resources.anchors_m0,
                resources.tau0,
                cfg.gamma,
            )
            tolerance = margin_tolerance(resources, deficit.dtype)
            violation_fraction = float((deficit > tolerance).to(deficit.dtype).mean())
            source_safety_sum += 1.0 - violation_fraction
            if violation_fraction:
                raise RuntimeError("guarded EP emitted an infeasible source-margin state")

            r_norm_sum += float(torch.linalg.vector_norm(R))
            abs_delta_score_sum += abs(float(result["score"] - result["score_before"]))

        trace = result.get("trace") or []
        guard_steps += len(trace)
        guard_count += sum(bool(row["margin_guard_applied"]) for row in trace)
        guard_backtracks += sum(int(row["margin_guard_backtracks"]) for row in trace)
        guard_reverts += sum(bool(row["margin_guard_reverted"]) for row in trace)
        n += 1

    return {
        "K": cand["K"],
        "lr": cand["lr"],
        "steps": cand["steps"],
        "view_reduction": view_reduction_sum / n,
        "probability_consistency": probability_consistency_sum / n,
        "source_margin_retention": source_margin_retention_sum / n,
        "source_safety": source_safety_sum / n,
        "mean_R_norm": r_norm_sum / n,
        "mean_abs_delta_score": abs_delta_score_sum / n,
        "margin_guard_count": guard_count,
        "margin_guard_backtracks": guard_backtracks,
        "margin_guard_reverts": guard_reverts,
        "margin_guard_activation_rate": guard_count / guard_steps if guard_steps else 0.0,
        "margin_guard_backtrack_count": guard_backtracks,
        "margin_guard_revert_rate": guard_reverts / guard_steps if guard_steps else 0.0,
        "numeric_fallback_count": 0,
    }


def score_dataset(sample_ids, features, resources, cache_id, cfg, method_id=METHOD_ID):
    scores = {}
    for sid in sample_ids:
        z = torch.from_numpy(features[sid])
        target = TargetViews(sid, z, cache_id)
        result = require_success(
            run_method(method_id, target, resources, cfg, {}), sid, method_id)
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
