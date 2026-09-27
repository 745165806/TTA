"""Extract only fixed target10 feature rows from the historical shared cache.

The source .npy chunks are memory-mapped; only selected row slices are copied.
The source index and ID sidecars are metadata, not feature values. No target
labels, scores or waveform encoder enter this worker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eptta.cache.keys import CacheIdentity
from eptta.cache.reader import FeatureCache
from eptta.cache.writer import FeatureCacheWriter
from eptta.models.frozen import verify_frozen_export
from experiments.head_capacity_geometry.resources import (BASE, ROOT, SELECTS,
    select_rows, sparse_selected_views)


HERE = Path(__file__).resolve().parent
SOURCE = BASE / "cache-target-in_the_wild"
OUTPUT = HERE / "target10_only_cache"


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def selected_by_index(cache, records):
    indices = [row["sample_index"] for row in records]
    if len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("invalid fixed target10 sample_index assignment")
    ordered = sorted((index, position) for position, index in enumerate(indices))
    values = np.empty((len(records), 3, 160), dtype=np.float32)
    cursor = offset = 0
    source_chunk_count = 0
    for chunk in cache.index["chunks"]:
        end = offset + chunk["count"]
        if cursor >= len(ordered) or ordered[cursor][0] >= end:
            offset = end
            continue
        array = np.load(cache.root / chunk["array_ref"], mmap_mode="r", allow_pickle=False)
        if array.shape != tuple(chunk["shape"]) or array.dtype != np.float32:
            raise ValueError("shared source cache chunk schema mismatch")
        source_chunk_count += 1
        while cursor < len(ordered) and ordered[cursor][0] < end:
            index, position = ordered[cursor]
            values[position] = array[index-offset]  # selected row only
            cursor += 1
        del array
        offset = end
    if cursor != len(records) or not np.isfinite(values).all():
        raise ValueError("target10 selected-row exact coverage/finite failure")
    return values, source_chunk_count


def build():
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    if OUTPUT.exists():
        raise FileExistsError("target10-only cache already exists; no in-place overwrite")
    ids, _, records = select_rows("itw")
    if len(ids) != 3178:
        raise ValueError("fixed target10 count changed")
    bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
    source = FeatureCache(SOURCE)
    original = source.index["identity"]
    if (original["dataset_id"] != "in_the_wild" or
            original["source_run_id"] != bundle["source_run_id"] or
            original["checkpoint_ref"] != bundle["checkpoint_ref"] or
            source.index["sample_count"] != 31779 or
            source.index["feature_dim"] != 160 or source.index["num_views"] != 3):
        raise ValueError("historical source cache identity/schema differs")
    selected, touched_chunks = selected_by_index(source, records)
    historical_selected, metadata_ids_scanned = sparse_selected_views(source, ids)
    if selected.shape != historical_selected.shape or not np.array_equal(selected, historical_selected):
        raise ValueError("selected-row features differ from historical ID-based selected read")
    parity = float(np.max(np.abs(selected.astype(np.float64)-historical_selected.astype(np.float64))))
    identity = CacheIdentity(**(original | {
        "cache_id": "gradient-target10-only-20260927a", "split_role": "select",
        "manifest_ref": str(SELECTS["itw"].resolve())}))
    with FeatureCacheWriter(OUTPUT, identity, ids, 3, 160) as writer:
        for begin in range(0, len(ids), 256):
            writer.add(ids[begin:begin+256], selected[begin:begin+256])
    isolated = FeatureCache(OUTPUT)
    recovered = isolated.load_by_id()
    if (isolated.index["sample_count"] != 3178 or len(recovered) != 3178 or
            set(recovered) != set(ids) or
            any(not np.array_equal(recovered[sid], selected[i]) for i, sid in enumerate(ids))):
        raise ValueError("new isolated cache exact ID/value coverage failure")
    provenance = {"status": "PASS", "sample_count": len(ids), "unique_ids": len(set(ids)),
        "feature_shape_per_sample": [3, 160], "dtype": "float32", "all_finite": True,
        "max_abs_feature_difference_vs_historical_selected_rows": parity,
        "source_cache_ref": str(SOURCE.resolve()),
        "source_cache_id": source.cache_id,
        "source_run_id": original["source_run_id"],
        "checkpoint_ref": original["checkpoint_ref"],
        "source_cache_sample_count": source.index["sample_count"],
        "source_chunk_files_mapped": touched_chunks,
        "source_id_metadata_rows_scanned": metadata_ids_scanned,
        "source_selection_ref": str(SELECTS["itw"].resolve()),
        "derived_cache_ref": str(OUTPUT.resolve()),
        "derived_cache_id": isolated.cache_id,
        "feature_value_access_policy": "only rows at the 3178 fixed sample_index positions were indexed",
        "shared_source_files_opened": True,
        "target90_label_score_metric_read": False,
        "waveform_encoder_rerun": False,
        "os_page_byte_isolation_claimed": False}
    write_new(OUTPUT / "CACHE_PROVENANCE.json", provenance)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(build(), indent=2))
