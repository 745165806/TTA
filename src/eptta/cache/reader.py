from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import read_json, sha256_file
from eptta.errors import DataError


class FeatureCache:
    def __init__(self, cache_ref, expected_identity=None):
        self.root = Path(cache_ref)
        self.index = read_json(self.root / "index.json" if self.root.is_dir() else self.root)
        if self.index.get("status") != "LOCKED" or self.index.get("format") != "sharded_npy_v1":
            raise DataError("feature cache is not locked")
        if self.index.get("allow_pickle") is not False:
            raise DataError("feature cache must disable pickle")
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
            if array.shape != tuple(item["shape"]) or len(ids) != item["count"]:
                raise DataError("feature cache chunk metadata mismatch")
            if set(ids).intersection(seen) or len(set(ids)) != len(ids):
                raise DataError("feature cache contains duplicate IDs")
            seen.update(ids)
            yield ids, array
        if len(seen) != self.index["sample_count"]:
            raise DataError("feature cache coverage count mismatch")

    def load_by_id(self):
        result = {}
        for ids, array in self.iter_chunks():
            result.update((sample_id, array[index]) for index, sample_id in enumerate(ids))
        return result
