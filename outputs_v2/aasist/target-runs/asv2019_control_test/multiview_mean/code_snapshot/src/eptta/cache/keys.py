"""Ordinary, explicit feature-cache provenance.

The cache identifier is a run-local name, not a content digest. Compatibility is
checked from declared scientific fields plus exact IDs and array metadata.
"""
from dataclasses import asdict, dataclass
from typing import Mapping

from eptta.errors import ContractError


@dataclass(frozen=True)
class CacheIdentity:
    cache_id: str
    source_run_id: str
    checkpoint_ref: str
    dataset_id: str
    split_role: str
    manifest_ref: str
    preprocess: Mapping[str, object]
    views: Mapping[str, object]
    seed: int
    dtype: str
    numerical_mode: Mapping[str, object]

    def __post_init__(self):
        for name in ("cache_id", "source_run_id", "checkpoint_ref", "dataset_id",
                     "split_role", "manifest_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ContractError("cache identity requires nonempty %s" % name)
        if self.dtype not in ("float16", "float32", "float64"):
            raise ContractError("unsupported cache dtype")
        if type(self.seed) is not int:
            raise ContractError("cache seed must be an integer")
        if not isinstance(self.preprocess, Mapping) or not isinstance(self.views, Mapping):
            raise ContractError("cache preprocess/views must be mappings")
        if not isinstance(self.numerical_mode, Mapping):
            raise ContractError("cache numerical_mode must be a mapping")

    def as_dict(self):
        return asdict(self)


def identities_match(left, right):
    """Compare declared fields; never infer identity from filenames or bytes."""
    return dict(left) == dict(right)
