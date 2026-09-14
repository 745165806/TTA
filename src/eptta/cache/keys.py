from dataclasses import dataclass
from typing import Mapping

from eptta.config.validate import content_hash
from eptta.errors import ContractError
from eptta.training.contracts import require_hash


@dataclass(frozen=True)
class CacheIdentity:
    input_manifest_sha256: str
    baseline_id: str
    selected_checkpoint_sha256: str
    head_sha256: str
    wrapper_numerical_version: str
    preprocess_sha256: str
    probe_sha256: str
    seed: int
    dtype: str
    numerical_mode: Mapping[str, object]

    def __post_init__(self):
        for name in ("input_manifest_sha256", "selected_checkpoint_sha256", "head_sha256",
                     "preprocess_sha256", "probe_sha256"):
            require_hash(getattr(self, name), name)
        if not self.baseline_id or not self.wrapper_numerical_version:
            raise ContractError("cache identity requires baseline and wrapper identities")
        if self.dtype not in ("float16", "float32", "float64"):
            raise ContractError("unsupported cache dtype")
        if type(self.seed) is not int:
            raise ContractError("cache seed must be an integer")

    def as_dict(self):
        return dict(self.__dict__)

    @property
    def cache_key(self):
        return "features-" + content_hash(self.as_dict())
