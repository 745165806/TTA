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
    model_contract: Mapping[str, Any]
    score_contract: Mapping[str, Any]
    numerical_contract: Mapping[str, Any]
    r4_validation: Mapping[str, Any]
    export_code_sha256: str

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
        require_hash(self.export_code_sha256, "export_code_sha256")
        provenance = self.task_training_provenance
        migration = provenance.get("migration")
        if migration is not None and (not isinstance(migration, Mapping) or
                                      migration.get("exact_resume_claim") is not False):
            raise ContractError("nonexact migration lineage must remain explicitly disclosed")
        endpoint = provenance.get("training_endpoint")
        if not isinstance(endpoint, Mapping):
            raise ContractError("frozen bundle must bind its actual training endpoint")
        for name in ("last_epoch", "completed_epoch_count", "scheduler_horizon_epochs"):
            if type(endpoint.get(name)) is not int or endpoint[name] < 0:
                raise ContractError("training endpoint contains an invalid %s" % name)
        if endpoint["completed_epoch_count"] != endpoint["last_epoch"] + 1:
            raise ContractError("training endpoint epoch count is inconsistent")
        legacy_r4 = self.r4_validation.get("parity_policy") is None
        if (legacy_r4 and self.model_id == "aasist_source" and
                (endpoint["last_epoch"] != 79 or endpoint["completed_epoch_count"] != 80 or
                 endpoint["scheduler_horizon_epochs"] != 100)):
            raise ContractError("legacy AASIST R4 bundle must retain its epoch79 endpoint")
        if migration is None and provenance.get("completion_evidence") != "locked_recipe_endpoint_or_approved_stop":
            raise ContractError("from-scratch training requires locked endpoint/stop evidence")
        if self.score_contract != {"formula": "native_logits[spoof]-native_logits[bonafide]",
                                   "direction": "larger_is_spoof", "output_type": "logit_difference",
                                   "unit": "dimensionless"}:
            raise ContractError("frozen score contract changed")
        if self.model_contract.get("embedding_point") != "native_out_layer_input" or self.model_contract.get(
                "freq_aug") is not False:
            raise ContractError("frozen embedding/augmentation contract is invalid")
        if self.numerical_contract.get("atol") != 1e-6 or self.numerical_contract.get("rtol") != 1e-5:
            raise ContractError("R4 numerical tolerance contract changed")
        if (self.r4_validation.get("status") != "PASS" or
                type(self.r4_validation.get("source_val_count")) is not int or
                self.r4_validation["source_val_count"] < 1 or
                type(self.r4_validation.get("fit_count")) is not int or
                self.r4_validation["fit_count"] < 2):
            raise ContractError("real R4 validation evidence is incomplete")
        if legacy_r4 and (self.r4_validation["source_val_count"] != 5654 or
                          self.r4_validation["fit_count"] != 128):
            raise ContractError("legacy real R4 validation evidence is incomplete")
        if set(self.class_index_map) != {"bonafide", "spoof"} or any(type(v) is not int for v in self.class_index_map.values()) or set(self.class_index_map.values()) != {0, 1}:
            raise ContractError("class index map must be bijective")
        if type(self.embedding_dim) is not int or self.embedding_dim < 1:
            raise ContractError("embedding dimension must be verified")
