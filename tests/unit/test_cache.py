import numpy as np
import pytest

from eptta.cache import CacheIdentity, FeatureCache, FeatureCacheWriter, merge_feature_caches
from eptta.errors import DataError


def identity(checkpoint="/runs/source/checkpoints/epoch_0001.pt"):
    return CacheIdentity("cache-1", "source-run-1", checkpoint, "fixture", "select",
                         "/data/select.jsonl", {"sample_rate": 16000},
                         {"num_views": 3}, 13, "float32",
                         {"tf32_matmul": False, "tf32_cudnn": False, "dtype": "float32"})


def test_cache_roundtrip_is_pickle_free_and_exact(tmp_path):
    with FeatureCacheWriter(tmp_path / "cache", identity(), ["a", "b", "c"], 3, 4) as writer:
        writer.add(["a", "b"], np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        writer.add(["c"], np.ones((1, 3, 4), dtype=np.float32))
    loaded = FeatureCache(tmp_path / "cache", identity()).load_by_id()
    assert set(loaded) == {"a", "b", "c"}
    assert loaded["a"].shape == (3, 4)


def test_cache_rejects_missing_duplicate_and_changed_checkpoint(tmp_path):
    with pytest.raises(DataError, match="missing"):
        with FeatureCacheWriter(tmp_path / "missing", identity(), ["a", "b"], 1, 2) as writer:
            writer.add(["a"], np.ones((1, 1, 2), dtype=np.float32))
    with pytest.raises(DataError, match="duplicate"):
        with FeatureCacheWriter(tmp_path / "dup", identity(), ["a"], 1, 2) as writer:
            writer.add(["a", "a"], np.ones((2, 1, 2), dtype=np.float32))
    with FeatureCacheWriter(tmp_path / "identity", identity(), ["a"], 1, 2) as writer:
        writer.add(["a"], np.ones((1, 1, 2), dtype=np.float32))
    with pytest.raises(DataError, match="identity"):
        FeatureCache(tmp_path / "identity", identity("/runs/source/checkpoints/epoch_0002.pt"))


def test_cache_rejects_nonfinite_values(tmp_path):
    with pytest.raises(DataError, match="non-finite"):
        with FeatureCacheWriter(tmp_path / "nan", identity(), ["a"], 1, 2) as writer:
            writer.add(["a"], np.array([[[np.nan, 0.0]]], dtype=np.float32))


def test_cache_shards_merge_with_exact_coverage(tmp_path):
    for name, ids in (("s0", ["a", "c"]), ("s1", ["b"])):
        with FeatureCacheWriter(tmp_path / name, identity(), ids, 1, 2) as writer:
            writer.add(ids, np.ones((len(ids), 1, 2), dtype=np.float32))
    result = merge_feature_caches([tmp_path / "s0", tmp_path / "s1"], tmp_path / "merged", ["a", "b", "c"])
    assert result["sample_count"] == 3
    assert set(FeatureCache(tmp_path / "merged").load_by_id()) == {"a", "b", "c"}
