from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from eptta.errors import ContractError, PermissionDenied
from eptta.training.contracts import InitializationRef, require_hash


class ModelFactory(Protocol):
    def build(self, architecture_lock: Mapping[str, Any], initialization: InitializationRef | None) -> Any: ...
    def class_index_map(self) -> Mapping[str, int]: ...


class FrozenDetector(Protocol):
    def encode(self, waveform: Any) -> Any: ...
    def score(self, embedding: Any) -> Any: ...
    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FrozenModelBundle:
    schema_version: str
    model_id: str
    baseline_id: str
    selected_checkpoint_sha256: str
    training_run_id: str
    fit_snapshot_hash: str
    source_val_snapshot_hash: str
    recipe_hash: str
    init_provenance: Mapping[str, Any]
    task_training_provenance: Mapping[str, Any]
    eval_preprocess_hash: str
    class_index_map: Mapping[str, int]
    head_ref: str
    embedding_dim: int
    training_status: str
    training_phase: str
    task_weight_origin: str
    source_val_selection_ref: str
    parity_report_ref: str

    def __post_init__(self):
        if self.schema_version != "0.1.0":
            raise ContractError("invalid frozen bundle version")
        if self.model_id not in ("aasist_source", "ssl_aasist_source"):
            raise ContractError("unregistered frozen model")
        for name in ("selected_checkpoint_sha256", "fit_snapshot_hash", "source_val_snapshot_hash", "recipe_hash", "eval_preprocess_hash"):
            require_hash(getattr(self, name), name)
        if self.task_weight_origin != "trained_in_project":
            raise PermissionDenied("external/synthetic task weights cannot become a production frozen bundle")
        if self.training_status != "FINALIZED" or self.training_phase != "full":
            raise ContractError("only finalized full source training is eligible for frozen export")
        if not all((self.training_run_id, self.task_training_provenance,
                    self.source_val_selection_ref, self.parity_report_ref)):
            raise ContractError("training/selection/parity evidence is required")
        if set(self.class_index_map) != {"bonafide", "spoof"} or any(type(v) is not int for v in self.class_index_map.values()) or set(self.class_index_map.values()) != {0, 1}:
            raise ContractError("class index map must be bijective")
        if type(self.embedding_dim) is not int or self.embedding_dim < 1:
            raise ContractError("embedding dimension must be verified")
