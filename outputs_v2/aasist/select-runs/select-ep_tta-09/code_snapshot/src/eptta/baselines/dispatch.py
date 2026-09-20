"""Route registered controls while enforcing one shared frozen source model."""
from dataclasses import dataclass

from eptta.adaptation.adapter import run_cache_method
from eptta.errors import ContractError, NotImplementedStage
from eptta.baselines.registry import get_method_contract


@dataclass(frozen=True)
class ComparisonGroup:
    comparison_group_id: str
    baseline_id: str
    checkpoint_ref: str

    def validate_run(self, metadata):
        if metadata.get("baseline_id") != self.baseline_id or metadata.get(
                "checkpoint_ref") != self.checkpoint_ref:
            raise ContractError("all compared methods must share the same frozen source checkpoint")


def run_method(method_id, target, resources, cfg, params=None):
    spec = get_method_contract(method_id)
    status = spec.get("implementation_status")
    if spec["comparison_track"] == "published_port":
        raise NotImplementedStage("published method is an audit-bound port, not a guessed fallback: %s" % method_id)
    if status not in ("IMPLEMENTED_SYNTHETIC_VERIFIED", "IMPLEMENTED_UNVERIFIED"):
        raise NotImplementedStage("method is not implemented: %s" % method_id)
    return run_cache_method(method_id, target, resources, cfg, params)
