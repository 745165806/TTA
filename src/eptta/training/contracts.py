from dataclasses import dataclass, fields
from typing import Any, Mapping, Protocol

from eptta.data.contracts import ContractIssue
from eptta.data.permissions import require_role
from eptta.errors import ContractError, PermissionDenied


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError("%s must be nonempty" % name)


@dataclass(frozen=True)
class SourceManifestRef:
    manifest_ref: str
    dataset_id: str
    role: str

    def __post_init__(self):
        require_text(self.manifest_ref, "manifest_ref")
        require_text(self.dataset_id, "dataset_id")


@dataclass(frozen=True)
class InitializationRef:
    artifact_ref: str
    scope: str
    pretraining_provenance: str

    def __post_init__(self):
        if self.scope != "generic_ssl_frontend_only":
            raise PermissionDenied("initialization permits generic SSL frontend only")
        require_text(self.artifact_ref, "initialization.artifact_ref")
        require_text(self.pretraining_provenance, "pretraining_provenance")


@dataclass(frozen=True)
class SourceTrainJob:
    schema_version: str
    job_type: str
    run_id: str
    model_id: str
    recipe_ref: str
    fit: SourceManifestRef
    source_val: SourceManifestRef
    initialization: InitializationRef | None
    output_dir: str
    training_seed: int
    phase: str

    def __post_init__(self):
        if self.schema_version != "0.3.0" or self.job_type != "source_train":
            raise ContractError("invalid source training job version/type")
        for name in ("run_id", "recipe_ref", "output_dir"):
            require_text(getattr(self, name), name)
        if self.model_id not in ("aasist_source", "ssl_aasist_source"):
            raise ContractError("unregistered model")
        require_role(self.fit.role, "source_gradient")
        require_role(self.source_val.role, "source_selection")
        if self.fit.manifest_ref == self.source_val.manifest_ref:
            raise PermissionDenied("fit and source_val manifests must be separate")
        if self.model_id == "aasist_source" and self.initialization is not None:
            raise PermissionDenied("AASIST starts from native initialization")
        if self.model_id == "ssl_aasist_source" and self.initialization is None:
            raise ContractError("SSL training requires explicit generic initialization")
        if self.phase not in ("smoke", "full") or type(self.training_seed) is not int or self.training_seed < 0:
            raise ContractError("invalid phase or training seed")

    @classmethod
    def from_dict(cls, payload):
        if set(payload) - {field.name for field in fields(cls)}:
            raise PermissionDenied("source job contains forbidden fields")
        data = dict(payload)
        try:
            data["fit"] = SourceManifestRef(**data["fit"])
            data["source_val"] = SourceManifestRef(**data["source_val"])
            if data.get("initialization") is not None:
                data["initialization"] = InitializationRef(**data["initialization"])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ContractError("invalid source job structure: %s" % exc) from exc


class SourceTrainer(Protocol):
    def validate_job(self, spec: SourceTrainJob) -> list[ContractIssue]: ...
    def train_smoke(self, spec: SourceTrainJob) -> Mapping[str, Any]: ...
    def train(self, spec: SourceTrainJob) -> Mapping[str, Any]: ...
    def resume(self, spec: SourceTrainJob) -> Mapping[str, Any]: ...
    def finalize(self, run_ref: str) -> Mapping[str, Any]: ...
    def export_frozen(self, finalized_training_ref: str) -> str: ...
