"""Atomic chunked NumPy feature writer (object arrays/pickle forbidden)."""
from pathlib import Path

from eptta.data.io import AtomicDirectory, write_json_new
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
        self.num_views, self.feature_dim = num_views, feature_dim
        self._expected, self._seen, self._chunks = set(self.expected_ids), set(), []
        self._atomic = self._temporary = None

    def __enter__(self):
        self._atomic = AtomicDirectory(self.output)
        self._temporary = self._atomic.__enter__()
        (self._temporary / "chunks").mkdir()
        return self

    def add(self, sample_ids, features):
        import numpy as np
        if self._temporary is None:
            raise RuntimeError("feature writer must be used as a context manager")
        ids, array = [str(value) for value in sample_ids], np.asarray(features)
        if array.dtype.hasobject or array.shape != (len(ids), self.num_views, self.feature_dim):
            raise DataError("feature chunk dtype/shape mismatch")
        if not np.isfinite(array).all():
            raise DataError("feature chunk contains non-finite values")
        if len(ids) != len(set(ids)) or self._seen.intersection(ids):
            raise DataError("duplicate IDs across feature chunks")
        if set(ids) - self._expected:
            raise DataError("feature chunk contains unexpected IDs")
        index = len(self._chunks)
        array_name, ids_name = "chunk-%06d.npy" % index, "chunk-%06d.ids.json" % index
        array_path = self._temporary / "chunks" / array_name
        ids_path = self._temporary / "chunks" / ids_name
        with array_path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        write_json_new(ids_path, ids)
        self._chunks.append({"index": index, "array_ref": "chunks/" + array_name,
                             "ids_ref": "chunks/" + ids_name, "count": len(ids),
                             "shape": list(array.shape), "dtype": str(array.dtype)})
        self._seen.update(ids)

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            return self._atomic.__exit__(kind, value, traceback)
        missing = self._expected - self._seen
        if missing:
            error = DataError("feature cache incomplete: %d IDs missing" % len(missing))
            self._atomic.__exit__(type(error), error, None)
            raise error
        metadata = {"schema_version": "0.2.0", "status": "READY", "format": "sharded_npy_v2",
                    "allow_pickle": False, "identity": self.identity.as_dict(),
                    "num_views": self.num_views, "feature_dim": self.feature_dim,
                    "sample_count": len(self._seen), "expected_ids": list(self.expected_ids),
                    "chunks": self._chunks, "immutable": True}
        write_json_new(self._temporary / "index.json", metadata)
        return self._atomic.__exit__(None, None, None)
