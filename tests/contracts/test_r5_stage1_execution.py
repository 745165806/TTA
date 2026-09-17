import json

import pytest
import torch

from eptta.adaptation.episode import run_k0_episode
from eptta.adaptation.types import TargetViews
from eptta.errors import ContractError
from eptta.execution.r5_stage1 import lock_stage1_proposal


def _target(views):
    return TargetViews("sample", views, "artifact")


def test_k0_entrypoint_returns_original_view_score_not_mean():
    z0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    z1 = torch.tensor([100.0, 100.0, 100.0, 100.0])
    z2 = torch.tensor([-50.0, -50.0, -50.0, -50.0])
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = 0.0
    target = _target(torch.stack([z0, z1, z2]))
    result = run_k0_episode(target, w, b, "bundle")
    assert result.score_after == float(z0 @ w + b)
    assert result.score_before == result.score_after
    assert result.status == "no_adaptation"
    assert result.steps_completed == 0
    assert result.final_r_fro == 0.0
    # A three-view mean would be very different; assert the entrypoint did not mean views.
    mean_score = float(torch.stack([z0, z1, z2]).mean(dim=0) @ w + b)
    assert result.score_after != mean_score


def test_k0_entrypoint_rejects_wrong_view_count():
    w = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="three"):
        run_k0_episode(_target(torch.zeros(2, 2)), w, 0.0, "bundle")
    with pytest.raises(ValueError, match="three"):
        run_k0_episode(_target(torch.zeros(4, 2)), w, 0.0, "bundle")


def test_k0_entrypoint_rejects_non_finite_and_bad_head():
    z = torch.stack([torch.tensor([1.0, 2.0]), torch.zeros(2), torch.zeros(2)])
    w = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        run_k0_episode(_target(torch.stack([torch.tensor([float("nan"), 0.0]), z[1], z[2]])), w, 0.0, "b")
    with pytest.raises(ValueError, match="head"):
        run_k0_episode(_target(z), torch.tensor([1.0, 0.0, 0.0]), 0.0, "b")
    with pytest.raises(ValueError, match="bias"):
        run_k0_episode(_target(z), w, float("inf"), "b")


def _proposal_dir(tmp_path):
    probe = {"num_views": 3, "seed": 13, "noise_snr_db": 30.0, "fir_side_gain": 0.05}
    numerical = {"dtype": "float32", "tf32": False, "block_units": 256}
    roles = {}
    for role in ("fit", "cal0", "select"):
        extraction = {"schema_version": "0.1.0", "status": "PROPOSED",
                      "purpose": "select" if role == "select" else "source_prepare",
                      "input_role": role, "frozen_bundle_ref": "/unused/bundle.json",
                      "manifest_ref": "/unused/" + role + ".jsonl",
                      "manifest_sha256": "a" * 64, "data_roots": {"k": "/unused"},
                      "probe": probe, "numerical_mode": numerical, "output_root": "/unused/caches/" + role}
        (tmp_path / (role + ".extraction.proposal.json")).write_text(json.dumps(extraction))
        roles[role] = {"purpose": extraction["purpose"], "count": 128,
                       "unlabeled_manifest_ref": role + ".unlabeled.jsonl",
                       "unlabeled_manifest_sha256": "b" * 64,
                       "selection_evidence_ref": role + ".selection.json",
                       "selection_evidence_sha256": "c" * 64,
                       "extraction_proposal_ref": role + ".extraction.proposal.json",
                       "extraction_proposal_sha256": "d" * 64}
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
                "locks_published": False, "proposal_id": "proposal",
                "parent_plan_ref": "/unused/plan.md", "parent_plan_sha256": "e" * 64,
                "bundle_ref": "/unused/bundle.json", "bundle_sha256": "f" * 64,
                "bundle_id": "baseline", "export_manifest_sha256": "1" * 64,
                "checkpoint_sha256": "2" * 64, "training_seed": 13,
                "source_snapshot_ref": "/unused/snap", "source_snapshot_sha256": "3" * 64,
                "worker_ref": "/unused/worker.py", "worker_sha256": "4" * 64,
                "views": ["identity", "deterministic_noise", "deterministic_fir"],
                "view_index": {"z0": 0}, "probe": probe, "probe_sha256": "5" * 64,
                "numerical_mode": numerical, "transform_contract": {"z0_storage": "view_index_0"},
                "roles": roles, "commands_after_review_lock": {},
                "real_block_coverage": "tail", "implementation_gaps_before_lock": ["gap"],
                "budget": {}, "forbidden": ["U", "M"]}
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal, sort_keys=True))
    import hashlib
    proposal_sha256 = hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    return proposal_path, proposal_sha256, roles


def test_lock_stage1_proposal_publishes_locked_plans(tmp_path):
    proposal_path, proposal_sha256, roles = _proposal_dir(tmp_path)
    for role in roles:
        roles[role]["extraction_proposal_sha256"] = _sha(proposal_path.parent /
                                                          roles[role]["extraction_proposal_ref"])
    # rewrite proposal with the real role hashes so the lock verifies them
    proposal = json.loads(proposal_path.read_text())
    for role in roles:
        proposal["roles"][role]["extraction_proposal_sha256"] = roles[role]["extraction_proposal_sha256"]
    proposal_path.write_text(json.dumps(proposal, sort_keys=True))
    proposal_sha256 = _sha(proposal_path)
    lock = lock_stage1_proposal(proposal_path, proposal_sha256)
    assert lock["status"] == "LOCKED"
    locked_dir = proposal_path.parent / "locked"
    assert locked_dir.is_dir()
    for role in roles:
        plan = json.loads((locked_dir / (role + ".extraction.lock.json")).read_text())
        assert plan["status"] == "LOCKED" and plan["input_role"] == role
    with pytest.raises(ContractError, match="overwrite"):
        lock_stage1_proposal(proposal_path, proposal_sha256)


def test_lock_stage1_proposal_rejects_wrong_hash(tmp_path):
    proposal_path, proposal_sha256, _roles = _proposal_dir(tmp_path)
    with pytest.raises(ContractError, match="changed"):
        lock_stage1_proposal(proposal_path, "0" * 64)


def _sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
