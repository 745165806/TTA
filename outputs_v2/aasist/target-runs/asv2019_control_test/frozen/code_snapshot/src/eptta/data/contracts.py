from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class RawRecord:
    record_ref: str
    fields: Mapping[str, Any]
    source_file: str = ""
    source_row: int | None = None
    protocol_context: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ContractIssue:
    code: str
    field: str
    record_ref: str | None
    message: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class LabelMapping:
    original_label: str | int | None
    canonical_label: int | None
    mapping_reason: str
    policy_id: str


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    schema_version: str
    sample_id: str
    dataset_id: str
    dataset_release: str | None
    snapshot_id: str
    root_key: str
    audio_relpath: str
    raw_record_ref: Mapping[str, Any]
    original_label: str | int | None
    canonical_label: int | None
    label_policy_id: str
    label_mapping_status: str
    official_split: str | None
    split_role: str
    source_group_id: str | None
    group_quality: str
    speaker_id: str | None = None
    generator_id: str | None = None
    generator_family: str | None = None
    parent_id: str | None = None
    treatment_id: str = "identity"
    codec_id: str | None = None
    status: str = "staged"
    arrival_batch_id: str | None = None
    source_release: str | None = None


class LabelMapper(Protocol):
    def map_label(self, value: str | int | None, approved_policy: Mapping[str, Any]) -> LabelMapping: ...


class DatasetAdapter(Protocol):
    def inspect(self, roots: Mapping[str, Path], hints: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def propose_contract(self, inventory: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def iter_raw(self, approved_contract: Mapping[str, Any]) -> Iterable[RawRecord]: ...
    def normalize(self, record: RawRecord, label_policy: Mapping[str, Any],
                  group_policy: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, list[ContractIssue]]: ...


class SplitPlanner(Protocol):
    def propose(self, snapshot_ref: str, policy: Mapping[str, Any], prior_assignments_ref: str | None) -> Mapping[str, Any]: ...
    def validate(self, proposal_ref: str) -> list[ContractIssue]: ...
    def commit(self, proposal_ref: str, approval_ref: str) -> str: ...


class IncrementalReconciler(Protocol):
    def diff(self, parent_snapshot_ref: str, candidate_records_ref: str) -> Mapping[str, Any]: ...
    def reconcile(self, diff_ref: str, grouping_policy_ref: str, role_policy_ref: str) -> Mapping[str, Any]: ...
    def commit_snapshot(self, reviewed_delta_ref: str, approval_ref: str) -> str: ...
