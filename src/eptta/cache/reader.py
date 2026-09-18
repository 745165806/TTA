from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import read_json, sha256_file
from eptta.errors import ContractError, DataError


class FeatureCache:
    def __init__(self, cache_ref, expected_identity=None):
        from eptta.cache.keys import CacheIdentity
        self.root = Path(cache_ref)
        self.index = read_json(self.root / "index.json" if self.root.is_dir() else self.root)
        if self.index.get("status") != "LOCKED" or self.index.get("format") != "sharded_npy_v1":
            raise DataError("feature cache is not locked")
        if self.index.get("allow_pickle") is not False:
            raise DataError("feature cache must disable pickle")
        try:
            identity = CacheIdentity(**self.index["identity"])
        except (KeyError, TypeError, ValueError, ContractError) as exc:
            raise DataError("feature cache identity is malformed") from exc
        if identity.cache_key != self.index.get("cache_key"):
            raise DataError("feature cache key does not match its recomputed identity")
        if expected_identity is not None and self.index.get("cache_key") != expected_identity.cache_key:
            raise DataError("feature cache identity mismatch")

    def iter_chunks(self):
        import numpy as np
        seen = set()
        root = self.root if self.root.is_dir() else self.root.parent
        for item in self.index["chunks"]:
            array_path = root / item["array_ref"]
            ids_path = root / item["ids_ref"]
            if sha256_file(array_path) != item["array_sha256"] or sha256_file(ids_path) != item["ids_sha256"]:
                raise DataError("feature cache chunk changed")
            ids = read_json(ids_path)
            with array_path.open("rb") as stream:
                array = np.load(stream, allow_pickle=False)
            if (array.shape != tuple(item["shape"]) or str(array.dtype) != item.get("dtype") or
                    len(ids) != item["count"] or array.ndim != 3 or
                    array.shape[1:] != (self.index.get("num_views"), self.index.get("feature_dim"))):
                raise DataError("feature cache chunk metadata mismatch")
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise DataError("feature cache contains object or non-finite values")
            if set(ids).intersection(seen) or len(set(ids)) != len(ids):
                raise DataError("feature cache contains duplicate IDs")
            seen.update(ids)
            yield ids, array
        if len(seen) != self.index["sample_count"]:
            raise DataError("feature cache coverage count mismatch")

    def verify_expected_ids(self, expected_ids):
        expected = list(expected_ids)
        if not expected or len(expected) != len(set(expected)):
            raise DataError("expected cache IDs must be unique and nonempty")
        actual = []
        for ids, _array in self.iter_chunks():
            actual.extend(ids)
        if set(actual) != set(expected):
            raise DataError("feature cache ID set differs from the approved manifest")
        return tuple(actual)

    def load_by_id(self):
        result = {}
        for ids, array in self.iter_chunks():
            result.update((sample_id, array[index]) for index, sample_id in enumerate(ids))
        return result
