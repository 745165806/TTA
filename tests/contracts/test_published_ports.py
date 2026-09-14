import pytest

from eptta.baselines.dispatch import ComparisonGroup, run_method
from eptta.baselines.ports.contracts import PORT_IDS, PublishedPortAudit, require_verified_port
from eptta.errors import ContractError, NotImplementedStage
from eptta.baselines.registry import get_method_contract
from eptta.config.schema import read_document
from pathlib import Path


@pytest.mark.parametrize("method_id", sorted(PORT_IDS))
def test_unverified_published_ports_fail_explicitly(method_id):
    audit = PublishedPortAudit(method_id, "BLOCKED_AUDIT", None, None, None, None, (), None,
                               "per_sample", False, None, ())
    with pytest.raises(NotImplementedStage, match="BLOCKED_AUDIT"):
        require_verified_port(audit)


def test_comparison_group_rejects_checkpoint_advantage():
    group = ComparisonGroup("group", "baseline-a", "a" * 64)
    group.validate_run({"baseline_id": "baseline-a", "selected_checkpoint_sha256": "a" * 64})
    with pytest.raises(ContractError, match="same frozen"):
        group.validate_run({"baseline_id": "baseline-b", "selected_checkpoint_sha256": "b" * 64})


def test_every_registered_method_has_the_complete_contract_fields():
    root = Path(__file__).resolve().parents[2]
    methods = read_document(root / "configs/methods/registry.yaml")["methods"]
    required = {"method_id", "comparison_track", "route", "parameterization", "objective",
                "regularizer", "subspace", "solver", "source_resources", "reset_policy",
                "final_output", "implementation_status"}
    for method_id in methods:
        assert set(get_method_contract(method_id)) == required
