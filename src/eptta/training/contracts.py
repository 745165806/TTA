from dataclasses import dataclass, fields
import re
from typing import Any, Mapping, Protocol

from eptta.data.contracts import ContractIssue
from eptta.data.permissions import require_role
from eptta.errors import ContractError, PermissionDenied


def require_hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError(f"{name} must be a SHA-256 hex digest")


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be nonempty")


@dataclass(frozen=True)
class SourceManifestRef:
    manifest_ref: str
    snapshot_hash: str
    role: str

    def __post_init__(self):
        require_text(self.manifest_ref, "manifest_ref")
        require_hash(self.snapshot_hash, "snapshot_hash")


@dataclass(frozen=True)
class InitializationRef:
    artifact_ref: str
    sha256: str
    scope: str
    pretraining_provenance: str

    def __post_init__(self):
        if self.scope != "generic_ssl_frontend_only":
            raise PermissionDenied("initialization permits generic SSL frontend only; task checkpoints are forbidden")
        require_hash(self.sha256, "initialization.sha256")
        require_text(self.artifact_ref, "initialization.artifact_ref")
        require_text(self.pretraining_provenance, "pretraining_provenance")


@dataclass(frozen=True)
class ResumeRef:
    artifact_ref: str
    training_run_id: str
    recipe_hash: str
    fit_snapshot_hash: str
    source_val_snapshot_hash: str
    task_weight_origin: str

    def __post_init__(self):
        for name in ("artifact_ref", "training_run_id"):
            require_text(getattr(self, name), name)
        for name in ("recipe_hash", "fit_snapshot_hash", "source_val_snapshot_hash"):
            require_hash(getattr(self, name), name)


@dataclass(frozen=True)
class SourceTrainJob:
    schema_version: str
    job_type: str
    model_id: str
    recipe_lock_ref: str
    recipe_hash: str
    fit: SourceManifestRef
    source_val: SourceManifestRef
    initialization: InitializationRef | None
    output_dir: str
    training_seed: int
    phase: str
    resume: ResumeRef | None = None

    def __post_init__(self):
        if self.schema_version != "0.1.0" or self.job_type != "source_train":
            raise ContractError("invalid source training job version/type")
        require_text(self.recipe_lock_ref, "recipe_lock_ref")
        require_text(self.output_dir, "output_dir")
        require_hash(self.recipe_hash, "recipe_hash")
        if self.model_id not in ("aasist_source", "ssl_aasist_source"):
            raise ContractError("unregistered model")
        if type(self.fit) is not SourceManifestRef or type(self.source_val) is not SourceManifestRef:
            raise PermissionDenied("source manifests must have explicit typed role references")
        require_role(self.fit.role, "source_gradient")
        require_role(self.source_val.role, "source_selection")
        if self.fit.manifest_ref == self.source_val.manifest_ref:
            raise PermissionDenied("fit and source_val manifests must be separate")
        if self.initialization is not None and type(self.initialization) is not InitializationRef:
            raise PermissionDenied("untyped initialization rejected")
        if self.model_id == "aasist_source" and self.initialization is not None:
            raise PermissionDenied("AASIST starts from native initialization")
        if self.model_id == "ssl_aasist_source" and self.initialization is None and self.resume is None:
            raise ContractError("SSL training requires explicit generic initialization")
        if self.phase not in ("smoke", "full") or type(self.training_seed) is not int or self.training_seed < 0:
            raise ContractError("invalid phase or training seed")
        if self.resume is not None:
            r = self.resume
            if type(r) is not ResumeRef or r.task_weight_origin != "trained_in_project":
                raise PermissionDenied("resume requires in-project training provenance")
            if (r.recipe_hash, r.fit_snapshot_hash, r.source_val_snapshot_hash) != (self.recipe_hash, self.fit.snapshot_hash, self.source_val.snapshot_hash):
                raise ContractError("resume requires the same recipe and snapshots")

    @classmethod
    def from_dict(cls, payload):
        allowed = {f.name for f in fields(cls)}
        if set(payload) - allowed:
            raise PermissionDenied("forbidden source job fields (target/select/calibration data are not accepted)")
        data = dict(payload)
        try:
            for key in ("fit", "source_val"):
                data[key] = SourceManifestRef(**data[key])
            for key, kind in (("initialization", InitializationRef), ("resume", ResumeRef)):
                if data.get(key) is not None:
                    data[key] = kind(**data[key])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ContractError(f"invalid source job structure: {exc}") from exc


class SourceTrainer(Protocol):
    def validate_job(self, spec: SourceTrainJob) -> list[ContractIssue]: ...
    def train_smoke(self, spec: SourceTrainJob) -> Mapping[str, Any]: ...
    def train(self, spec: SourceTrainJob) -> Mapping[str, Any]: ...
    def resume(self, spec: SourceTrainJob) -> Mapping[str, Any]: ...
    def finalize(self, run_ref: str) -> Mapping[str, Any]: ...
    def export_frozen(self, finalized_training_ref: str) -> str: ...
