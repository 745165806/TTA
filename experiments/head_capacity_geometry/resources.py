"""Frozen development features and source head, without target labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs_v2/ssl_aasist"
WAVE_CACHE = (ROOT / "experiments/capacity_audit/results/"
              "wavefake_capacity_cache_20260927a/feature_cache")
SELECTS = {
    "itw": ROOT / "experiments/large_scale_confirmation/manifests/"
           "in_the_wild_confirmation_select.json",
    "wavefake": ROOT / "experiments/capacity_audit/manifests/wavefake_capacity_select.json",
}
FORBIDDEN = {"label", "canonical_label", "raw_label", "attack_id", "correct_before",
             "correct_after", "helpful_update", "harmful_update"}


def select_rows(domain: str):
    if domain not in SELECTS:
        raise ValueError("unregistered head-development domain")
    doc = json.loads(SELECTS[domain].read_text(encoding="utf-8"))
    rows = doc["records"]
    count = 3178 if domain == "itw" else 4096
    if (doc["count"] != count or len(rows) != count or
            FORBIDDEN.intersection(doc) or any(FORBIDDEN.intersection(row) for row in rows)):
        raise ValueError("label-free fixed selection invalid")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixed sample ID")
    groups = None if domain == "itw" else [row["audio_id"] for row in rows]
    return ids, groups, rows


def sparse_selected_views(cache: FeatureCache, ids: list[str]):
    """Map only selected feature rows; inspect all chunk ID metadata, no other values."""
    wanted = set(ids)
    seen = {}
    metadata_ids_scanned = 0
    for chunk in cache.index["chunks"]:
        all_ids = json.loads((cache.root / chunk["ids_ref"]).read_text(encoding="utf-8"))
        if len(all_ids) != chunk["count"] or len(set(all_ids)) != len(all_ids):
            raise ValueError("feature-cache chunk ID metadata invalid")
        metadata_ids_scanned += len(all_ids)
        matches = [(position, sample_id) for position, sample_id in enumerate(all_ids)
                   if sample_id in wanted]
        if not matches:
            continue
        array = np.load(cache.root / chunk["array_ref"], mmap_mode="r", allow_pickle=False)
        if array.shape != tuple(chunk["shape"]) or array.dtype != np.float32:
            raise ValueError("selected feature-cache chunk schema invalid")
        for position, sample_id in matches:
            if sample_id in seen:
                raise ValueError("duplicate selected feature ID")
            value = np.array(array[position], dtype=np.float32, copy=True)
            if value.shape != (3, 160) or not np.isfinite(value).all():
                raise ValueError("selected feature shape/finite failure")
            seen[sample_id] = value
        del array
    if set(seen) != wanted:
        raise ValueError("selected feature coverage mismatch")
    return np.stack([seen[sid] for sid in ids]), metadata_ids_scanned


def load_domain(domain: str):
    ids, groups, _ = select_rows(domain)
    bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
    resources, _, metadata = load_frozen_resources(BASE / "resources", bundle)
    cache_ref = BASE / "cache-target-in_the_wild" if domain == "itw" else WAVE_CACHE
    cache = FeatureCache(cache_ref)
    identity = cache.index["identity"]
    expected_dataset = "in_the_wild" if domain == "itw" else "wavefake"
    if (identity["source_run_id"] != bundle["source_run_id"] or
            identity["checkpoint_ref"] != bundle["checkpoint_ref"] or
            identity["dataset_id"] != expected_dataset or
            cache.index["num_views"] != 3 or cache.index["feature_dim"] != 160):
        raise ValueError("feature cache/source bundle provenance mismatch")
    views, metadata_ids_scanned = sparse_selected_views(cache, ids)
    provenance = {"source_bundle": str((BASE / "frozen/bundle.json").resolve()),
        "source_run_id": bundle["source_run_id"], "checkpoint_ref": bundle["checkpoint_ref"],
        "cache_ref": str(cache_ref.resolve()), "cache_id": cache.cache_id,
        "selection_ref": str(SELECTS[domain].resolve()), "selected_rows_loaded": len(ids),
        "chunk_id_metadata_rows_scanned": metadata_ids_scanned,
        "nonselected_feature_values_loaded_into_application": False,
        "target_labels_read": False, "target90_labels_read": False,
        "target90_metrics_read": False, "tau0": float(metadata["scalars"]["tau0"])}
    return ids, views, groups, resources, provenance
