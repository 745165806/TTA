"""No-formula-guessing status for published audio TTA ports."""
from dataclasses import dataclass
from typing import Mapping

from eptta.errors import ContractError, NotImplementedStage


PORT_IDS = frozenset({"tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full",
                      "eata_audio_ep", "t2a_audio_ep"})


@dataclass(frozen=True)
class PublishedPortAudit:
    method_id: str
    status: str
    repository: str | None
    repo_commit: str | None
    license_ref: str | None
    patch_ref: str | None
    updated_parameter_names: tuple[str, ...]
    normalization_policy: str | None
    reset_policy: str
    target_history: bool
    parity_report_ref: str | None
    protocol_deviations: tuple[str, ...]

    def __post_init__(self):
        if self.method_id not in PORT_IDS or self.reset_policy != "per_sample" or self.target_history:
            raise ContractError("episodic published port identity/reset/history contract mismatch")
        if self.status not in ("NOT_IMPLEMENTED", "VERIFIED"):
            raise ContractError("invalid published port audit status")
        if self.status == "VERIFIED" and not all((self.repository, self.repo_commit, self.license_ref,
                                                   self.patch_ref, self.updated_parameter_names,
                                                   self.normalization_policy, self.parity_report_ref)):
            raise ContractError("VERIFIED port lacks source/license/patch/parameter/parity evidence")


def require_verified_port(audit):
    if type(audit) is not PublishedPortAudit or audit.status != "VERIFIED":
        raise NotImplementedStage("published port remains NOT_IMPLEMENTED; no substitute formula is allowed")
    return audit
