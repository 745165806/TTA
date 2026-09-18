"""Strict reader for explicit-run NumPy feature caches."""
from pathlib import Path

from eptta.data.io import read_json
from eptta.errors import ContractError, DataError


class FeatureCache:
    def __init__(self, cache_ref, expected_identity=None):
        from eptta.cache.keys import CacheIdentity, identities_match
        self.root = Path(cache_ref)
        self.index = read_json(self.root / "index.json" if self.root.is_dir() else self.root)
        if self.index.get("allow_pickle") is not False:
            raise DataError("feature cache must disable pickle")
        fmt = self.index.get("format")
        if fmt == "sharded_npy_v2":
            if self.index.get("status") != "READY":
                raise DataError("feature cache is incomplete")
            try:
                identity = CacheIdentity(**self.index["identity"])
            except (KeyError, TypeError, ValueError, ContractError) as exc:
                raise DataError("feature cache identity is malformed") from exc
            if expected_identity is not None and not identities_match(
                    identity.as_dict(), expected_identity.as_dict()):
                raise DataError("feature cache identity/provenance differs from this run")
        elif fmt == "sharded_npy_v1":
            # Historical caches remain explicit, read-only inputs. Stored digest
            # fields are ignored and are never recomputed by the new workflow.
            if self.index.get("status") != "LOCKED" or "identity" not in self.index:
                raise DataError("legacy feature cache is incomplete")
            if expected_identity is not None:
                raise DataError("legacy cache requires an explicit compatibility conversion")
        else:
            raise DataError("unsupported feature cache format")

    @property
    def cache_id(self):
        return self.index.get("identity", {}).get("cache_id") or "legacy-cache:" + self.root.name

    def iter_chunks(self):
        import numpy as np
        seen = set()
        root = self.root if self.root.is_dir() else self.root.parent
        chunks = self.index.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise DataError("feature cache has no chunks")
        for item in chunks:
            array_path, ids_path = root / item["array_ref"], root / item["ids_ref"]
            if not array_path.is_file() or not ids_path.is_file():
                raise DataError("feature cache chunk is missing")
            ids = read_json(ids_path)
            with array_path.open("rb") as stream:
                array = np.load(stream, allow_pickle=False)
            if (array.shape != tuple(item["shape"]) or str(array.dtype) != item.get("dtype") or
                    len(ids) != item["count"] or array.ndim != 3 or
                    array.shape[1:] != (self.index.get("num_views"), self.index.get("feature_dim"))):
                raise DataError("feature cache chunk metadata mismatch")
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise DataError("feature cache contains object or non-finite values")
            if not all(isinstance(value, str) and value for value in ids):
                raise DataError("feature cache IDs must be nonempty strings")
            if set(ids).intersection(seen) or len(set(ids)) != len(ids):
                raise DataError("feature cache contains duplicate IDs")
            seen.update(ids)
            yield ids, array
        if len(seen) != self.index.get("sample_count"):
            raise DataError("feature cache coverage count mismatch")

    def verify_expected_ids(self, expected_ids):
        expected = list(expected_ids)
        if not expected or len(expected) != len(set(expected)):
            raise DataError("expected cache IDs must be unique and nonempty")
        actual = []
        for ids, _array in self.iter_chunks():
            actual.extend(ids)
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise DataError("feature cache ID set differs from the manifest")
        return tuple(actual)

    def load_by_id(self):
        result = {}
        for ids, array in self.iter_chunks():
            result.update((sample_id, array[index]) for index, sample_id in enumerate(ids))
        return result
