import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class EPConfig:
    steps: int = 3
    lr: float = 0.01
    rho: float = 0.2
    gamma: float = 0.1
    lambda_keep: float = 1.0

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        values = (self.lr, self.rho, self.gamma, self.lambda_keep)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("adaptation parameters must be finite real scalars")
        if not (self.lr > 0 and 0 < self.rho < 1 and 0 <= self.gamma < 1 and self.lambda_keep >= 0):
            raise ValueError("invalid adaptation parameter range")

    @classmethod
    def from_config(cls, config):
        from eptta.config.schema import check
        from eptta.registry import validate_selection
        check(config)
        validate_selection(config)
        p = config["defaults"]
        return cls(p["steps"], p["lr"], p["rho"], p["gamma"], p["regularizer_weight"])


@dataclass(frozen=True, slots=True)
class TargetViews:
    sample_id: str
    features: "Tensor"
    feature_artifact_id: str


@dataclass(frozen=True, slots=True)
class FrozenResources:
    U: "Tensor"
    w: "Tensor"
    b: float
    anchors_z: "Tensor"
    anchors_y: "Tensor"
    anchors_m0: "Tensor"
    anchors_s0: "Tensor"
    tau0: float
    artifact_bundle_id: str
    source_role: str = "fit"
    schema_version: str = "0.1.0"
    channel: str = "synthetic_test"


@dataclass(frozen=True, slots=True)
class EpisodeOutput:
    sample_id: str
    feature_artifact_id: str
    artifact_bundle_id: str
    R: "Tensor"
    score_before: float
    score_after: float
    status: str
    reason: str | None
    steps_completed: int
    final_loss_view: float | None
    final_loss_keep: float | None
    final_violation_fraction: float | None
    final_r_fro: float
    trace: tuple[dict[str, float], ...] = field(default_factory=tuple)
    schema_version: str = "0.1.0"
    method_id: str = "ep_tta"


@dataclass(frozen=True, slots=True)
class BatchOutput:
    episodes: tuple[EpisodeOutput, ...]
    serial_replay_count: int = 0
    serial_replay_seconds: float = 0.0
