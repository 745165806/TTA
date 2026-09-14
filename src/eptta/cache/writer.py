"""Atomic chunked NumPy feature writer (never uses object arrays/pickle)."""
import json
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, sha256_file, write_json_new
from eptta.errors import DataError


class FeatureCacheWriter:
    def __init__(self, output, identity, expected_ids, num_views, feature_dim):
        self.output = Path(output)
        self.identity = identity
        self.expected_ids = tuple(expected_ids)
        if len(set(self.expected_ids)) != len(self.expected_ids) or not self.expected_ids:
            raise DataError("cache expected IDs must be unique and nonempty")
        if type(num_views) is not int or num_views < 1 or type(feature_dim) is not int or feature_dim < 1:
            raise DataError("invalid cache shape")
        self.num_views = num_views
        self.feature_dim = feature_dim
        self._expected = set(self.expected_ids)
        self._atomic = None
        self._temporary = None
        self._seen = set()
        self._chunks = []

    def __enter__(self):
        self._atomic = AtomicDirectory(self.output)
        self._temporary = self._atomic.__enter__()
        (self._temporary / "chunks").mkdir()
        return self

    def add(self, sample_ids, features):
        import numpy as np
        if self._temporary is None:
            raise RuntimeError("feature writer must be used as a context manager")
        ids = [str(value) for value in sample_ids]
        array = np.asarray(features)
        if array.dtype.hasobject:
            raise DataError("object arrays are forbidden in feature caches")
        if array.shape != (len(ids), self.num_views, self.feature_dim):
            raise DataError("feature chunk shape mismatch")
        if not np.isfinite(array).all():
            raise DataError("feature chunk contains non-finite values")
        duplicates = self._seen.intersection(ids)
        if len(ids) != len(set(ids)) or duplicates:
            raise DataError("duplicate IDs across feature chunks")
        unexpected = set(ids) - self._expected
        if unexpected:
            raise DataError("unexpected cache IDs: %s" % sorted(unexpected)[:3])
        index = len(self._chunks)
        array_name = "chunk-%06d.npy" % index
        ids_name = "chunk-%06d.ids.json" % index
        array_path = self._temporary / "chunks" / array_name
        ids_path = self._temporary / "chunks" / ids_name
        with array_path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        write_json_new(ids_path, ids)
        item = {"index": index, "array_ref": "chunks/" + array_name,
                "array_sha256": sha256_file(array_path), "ids_ref": "chunks/" + ids_name,
                "ids_sha256": sha256_file(ids_path), "count": len(ids),
                "shape": list(array.shape), "dtype": str(array.dtype)}
        self._chunks.append(item)
        self._seen.update(ids)

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            return self._atomic.__exit__(kind, value, traceback)
        missing = self._expected - self._seen
        if missing:
            error = DataError("feature cache incomplete: %d IDs missing" % len(missing))
            self._atomic.__exit__(type(error), error, None)
            raise error
        metadata = {"schema_version": "0.1.0", "status": "LOCKED", "format": "sharded_npy_v1",
                    "allow_pickle": False, "cache_key": self.identity.cache_key,
                    "identity": self.identity.as_dict(), "num_views": self.num_views,
                    "feature_dim": self.feature_dim, "sample_count": len(self._seen),
                    "expected_ids_sha256": content_hash(list(self.expected_ids)), "chunks": self._chunks,
                    "immutable": True}
        write_json_new(self._temporary / "index.json", metadata)
        return self._atomic.__exit__(None, None, None)
