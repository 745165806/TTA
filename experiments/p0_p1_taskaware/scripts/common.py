"""Shared helpers for the p0_p1_taskaware pipeline.

The adaptation path is the exact production ``run_method`` entry used by the
project's ``run-tta``; this module only adds orchestration and file plumbing.
Labels are strictly forbidden here: this module and the workers it feeds may
read the label-free select manifest, the feature cache, the frozen resources and
candidate parameters, and nothing else.
"""
import json
from pathlib import Path

import torch

from eptta.adaptation.types import EPConfig, TargetViews
from eptta.baselines.dispatch import run_method
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/p0_p1_taskaware"
TARGET10_EXP = ROOT / "experiments/target10_selection"

FROZEN_BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
RESOURCES = ROOT / "outputs_v2/ssl_aasist/resources"
CACHE = ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"

SELECT_MANIFEST = TARGET10_EXP / "manifests/inwild_target10_select.json"
LABEL_MANIFEST = TARGET10_EXP / "manifests/inwild_target10.json"
PARAM_SEARCH = TARGET10_EXP / "results/param_search.json"

POST_HOC_MARKER = "POST_HOC_DEVELOPMENT_ONLY"

# Label/leakage keys forbidden inside the label-free select manifest.
FORBIDDEN_LABEL_KEYS = frozenset({
    "label", "canonical_label", "original_label", "target", "class", "y", "source_labels",
})


def load_context():
    """Load frozen bundle/resources/threshold/cache/features (label-free)."""
    bundle, _manifest, _parity, _selection = verify_frozen_export(FROZEN_BUNDLE)
    resources, _extras, resource_meta = load_frozen_resources(RESOURCES, bundle)
    threshold = float(resource_meta["scalars"]["tau0"])
    cache = FeatureCache(CACHE)
    features = cache.load_by_id()
    return bundle, resources, resource_meta, cache, features, threshold


def load_candidates():
    """Read the frozen 25-candidate grid directly (no parameter re-derivation)."""
    doc = json.loads(PARAM_SEARCH.read_text(encoding="utf-8"))
    candidates = doc.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 25:
        raise RuntimeError("param_search.json must contain exactly 25 candidates")
    return list(candidates)


def load_select_sample_ids():
    """Label-free select manifest -> ordered unique sample IDs.

    Re-validates the label-free permission contract so a labelled manifest can
    never silently leak into a worker.
    """
    from eptta.data.permissions import TargetInputManifest
    from eptta.errors import PermissionDenied

    doc = json.loads(SELECT_MANIFEST.read_text(encoding="utf-8"))
    if doc.get("role") != "select":
        raise PermissionDenied("select manifest role must be 'select'")
    if "source_labels" in doc:
        raise PermissionDenied("select manifest must not contain source_labels")
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != doc.get("count"):
        raise PermissionDenied("select manifest count mismatch")
    sample_ids = []
    for record in records:
        TargetInputManifest.from_dict(record)
        if record.get("split_role") != "select":
            raise PermissionDenied("select record split_role must be 'select'")
        for key in record:
            if key.lower() in FORBIDDEN_LABEL_KEYS or "label" in key.lower():
                raise PermissionDenied("label field present in select record: %s" % key)
        sample_ids.append(record["sample_id"])
    if len(sample_ids) != len(set(sample_ids)):
        raise PermissionDenied("select manifest sample_ids are not unique")
    return sample_ids


def load_labels():
    """Read target10 ground-truth labels for post-hoc diagnostics ONLY.

    Callers are aggregators; workers must never import/call this.
    """
    doc = json.loads(LABEL_MANIFEST.read_text(encoding="utf-8"))
    labels = {}
    for record in doc.get("records", []):
        labels[record["sample_id"]] = int(record["label"])
    return labels


def ep_config(cand):
    """EPConfig for a param_search candidate (K==steps in this codebase)."""
    return EPConfig(steps=cand["K"], lr=cand["lr"] if cand["lr"] is not None else 0.003,
                    rho=0.2, gamma=0.1, lambda_keep=1.0)


def view_scores(target, resources, R=None):
    """Per-view frozen head scores (before if R is None, after if R is given)."""
    from eptta.adaptation.math import apply_adapter
    z = target.features
    if R is not None:
        z = apply_adapter(z, resources.U, R)
    return (z @ resources.w + resources.b).detach().tolist()


def make_target(sample_id, features, cache_id):
    """Wrap a numpy (3, d) feature row into a TargetViews on CPU."""
    return TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache_id)
