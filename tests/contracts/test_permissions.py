import ast
import copy
import csv
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from eptta.adaptation.types import FrozenResources, TargetViews
from eptta.config.schema import read_document
from eptta.config.validate import content_hash
from eptta.data.permissions import ExplicitLabelMapper, TargetInputManifest, project_target, require_role, safe_relative, resolve_under_root
from eptta.errors import ContractError, PermissionDenied
from eptta.models.contracts import FrozenModelBundle
from eptta.training.contracts import InitializationRef, ResumeRef, SourceManifestRef, SourceTrainJob

ROOT = Path(__file__).resolve().parents[2]


def policy():
    payload = {"policy_id": "fixture_only_reverse_numeric", "raw_to_canonical": {"1": 0, "0": 1}, "unknown_policy": "quarantine"}
    return {"schema_version": "0.1.0", "status": "LOCKED", "payload": payload,
            "approval": {"reviewer": "synthetic_test", "approved_at": "2026-09-13T00:00:00Z",
                         "report_ref": "fixture_only", "sample_evidence_ref": "tests/fixtures/protocol.csv",
                         "content_sha256": content_hash(payload)}}


def test_d03_explicit_reverse_mapping_and_unknown_quarantine():
    with (ROOT / "tests/fixtures/protocol.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))  # Test-side fixture parser; not a real DatasetAdapter.
    mapped = [ExplicitLabelMapper().map_label(row["raw_label"], policy()) for row in records]
    assert [r.canonical_label for r in mapped] == [0, 1, None, None]
    assert mapped[0].original_label == "1"
    assert mapped[2].mapping_reason == "quarantine_unmapped"
    assert ExplicitLabelMapper().map_label(None, policy()).canonical_label is None
    assert ExplicitLabelMapper().map_label(True, policy()).canonical_label is None
    assert all(r.policy_hash == content_hash(policy()["payload"]) for r in mapped)


def test_unreviewed_label_policy_rejected():
    p = policy()
    p["status"] = "PROPOSED"
    with pytest.raises(ContractError):
        ExplicitLabelMapper().map_label("1", p)


def test_changed_mapping_requires_new_review():
    p = policy()
    p["payload"]["raw_to_canonical"]["1"] = 1
    with pytest.raises(ContractError):
        ExplicitLabelMapper().map_label("1", p)


@pytest.mark.parametrize("path", ["../audio.wav", "/data/a.wav", "C:/a.wav", "C:a.wav", "a/../../x", "a\\x", "//server/share", ""])
def test_path_escape_rejected(path):
    with pytest.raises(PermissionDenied):
        safe_relative(path)


def test_root_confinement(tmp_path):
    assert resolve_under_root(tmp_path, "fixture/a.wav") == tmp_path / "fixture/a.wav"
    with pytest.raises(PermissionDenied):
        resolve_under_root(tmp_path, "../outside.wav")


def test_c03_manifest_whitelist():
    record = dict(schema_version="0.1.0", sample_id="opaque-1", root_key="fixture", audio_relpath="fixture/a.wav",
                  input_sha256=None, decode_profile_id="fixture_decode", probe_profile_id="fixture_probe",
                  canonical_label=0, generator_id="fixture_attack", clean_path="secret.wav", parent_id="secret-parent")
    first = project_target(record)
    record.update(canonical_label=1, generator_id="other_attack", clean_path="other_secret.wav")
    assert project_target(record) == first
    with pytest.raises(PermissionDenied):
        TargetInputManifest.from_dict(record)
    assert {f.name for f in fields(TargetViews)} == {"sample_id", "features", "feature_artifact_id"}
    with pytest.raises(TypeError):
        TargetViews("id", None, "artifact", canonical_label=1)


def job():
    return SourceTrainJob("0.1.0", "source_train", "aasist_source", "fixture_recipe", "a" * 64,
                          SourceManifestRef("fixture_fit", "b" * 64, "fit"),
                          SourceManifestRef("fixture_val", "c" * 64, "source_val"),
                          None, "fixture_output", 13, "smoke")


@pytest.mark.parametrize("role", ["select", "cal0", "audit", "target_test", "control_test", "unassigned", "quarantine"])
def test_d04_training_role_firewall(role):
    j = job()
    with pytest.raises(PermissionDenied):
        replace(j, fit=replace(j.fit, role=role))
    with pytest.raises(PermissionDenied):
        replace(j, source_val=replace(j.source_val, role=role))


def test_source_job_wire_contract():
    j = job()
    assert SourceTrainJob.from_dict(asdict(j)) == j
    for key in ("target_manifest_ref", "select", "cal0", "labels", "R", "lambda_keep"):
        data = asdict(j)
        data[key] = "forbidden"
        with pytest.raises((PermissionDenied, ContractError)):
            SourceTrainJob.from_dict(data)


def test_ssl_init_is_not_task_checkpoint():
    with pytest.raises(PermissionDenied):
        InitializationRef("fixture_external_task_checkpoint", "a" * 64, "task_weights", "external")
    ssl = InitializationRef("fixture_generic_ssl", "a" * 64, "generic_ssl_frontend_only", "fixture")
    assert replace(job(), model_id="ssl_aasist_source", initialization=ssl).initialization == ssl
    with pytest.raises(ContractError):
        replace(job(), model_id="ssl_aasist_source")


def test_resume_cannot_inject_external_or_changed_snapshot():
    j = job()
    resume = ResumeRef("fixture_resume", "fixture_run", j.recipe_hash, j.fit.snapshot_hash, j.source_val.snapshot_hash, "trained_in_project")
    assert replace(j, resume=resume).resume == resume
    with pytest.raises(PermissionDenied):
        replace(j, resume=replace(resume, task_weight_origin="external_task_checkpoint"))
    with pytest.raises(ContractError):
        replace(j, resume=replace(resume, fit_snapshot_hash="d" * 64))


def bundle():
    return FrozenModelBundle("0.1.0", "aasist_source", "fixture_baseline", "a" * 64, "fixture_run", "b" * 64,
                             "c" * 64, "d" * 64, {"native": True},
                             {"fixture_only": True, "migration": {"exact_resume_claim": False},
                              "training_endpoint": {"last_epoch": 79, "completed_epoch_count": 80,
                                                    "scheduler_horizon_epochs": 100}}, "e" * 64,
                             {"bonafide": 1, "spoof": 0}, "fixture_head", 7, "FINALIZED", "full",
                             "trained_in_project", "fixture_selection", "fixture_parity",
                             {"embedding_point": "native_out_layer_input", "freq_aug": False},
                             {"formula": "native_logits[spoof]-native_logits[bonafide]",
                              "direction": "larger_is_spoof", "output_type": "logit_difference",
                              "unit": "dimensionless"},
                             {"atol": 1e-6, "rtol": 1e-5},
                             {"status": "PASS", "fit_count": 128, "source_val_count": 5654}, "f" * 64)


@pytest.mark.parametrize("changes", [{"training_phase": "smoke"}, {"training_status": "RUNNING"},
    {"task_weight_origin": "external_task_checkpoint"}, {"task_weight_origin": "synthetic_test"},
    {"source_val_selection_ref": ""}, {"parity_report_ref": ""}])
def test_s05_export_contract_rejects_ineligible(changes):
    with pytest.raises((ContractError, PermissionDenied)):
        replace(bundle(), **changes)


def test_adaptation_import_boundary():
    for path in (ROOT / "src/eptta/adaptation").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any((node.module or "").startswith(prefix) for prefix in ("eptta.evaluation", "eptta.data", "eptta.training"))


def test_source_roles_distinct_from_adaptation_selection():
    require_role("fit", "source_gradient")
    require_role("source_val", "source_selection")
    require_role("select", "method_selection")
    with pytest.raises(PermissionDenied):
        require_role("select", "source_selection")
