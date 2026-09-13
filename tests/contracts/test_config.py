import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eptta.config.resolve import expand_env, resolve, resolve_training
from eptta.config.schema import check, loads, read_document
from eptta.config.validate import check_contract, content_hash, report, validate_stage
from eptta.errors import EPTTAError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    return read_document(ROOT / "configs/base.yaml")


def test_d01_null_structure_no_resource_access(cfg, monkeypatch):
    cfg["paths"]["dataset_roots"]["asvspoof2019_la"] = "${NEVER_READ}"
    original_stat = Path.stat
    def guarded_stat(path, *a, **kw):
        if "NEVER_READ" in str(path):
            raise AssertionError("resource I/O attempted")
        return original_stat(path, *a, **kw)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    assert validate_stage(cfg) == []
    assert cfg["contracts"]["datasets"]["asvspoof2019_la"]["raw"]["payload"]["columns"] is None


def test_c01_cli_no_torch_import_or_network():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    script = "import sys; from eptta.cli import main; assert main(['validate']) == 0; assert 'torch' not in sys.modules; assert 'socket' not in sys.modules"
    out = subprocess.run([sys.executable, "-S", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"outer":{"x":1,"x":2}}', '{"lr":NaN}', '{"lr":Infinity}'])
def test_c02_duplicate_or_nonfinite(text):
    with pytest.raises(EPTTAError):
        loads(text)


def test_yaml_duplicate_if_available():
    pytest.importorskip("yaml", reason="NOT_RUN: PyYAML absent; JSON duplicate checks still execute")
    with pytest.raises(EPTTAError):
        loads("a: 1\na: 2\n")


@pytest.mark.parametrize("section,key,value", [("defaults", "steps", True), ("defaults", "rho", 1),
    ("defaults", "lr", float("nan")), ("defaults", "weight_decay", .1), ("probe", "num_views", 2),
    ("probe", "regenerate_per_step", True), ("protocol", "target_labels_visible", True),
    ("model_policy", "allow_external_task_checkpoint", True), ("safety", "automatic_ssh", True)])
def test_c02_invalid_fields(cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises(EPTTAError):
        check(cfg)


def test_unknown_nested_key(cfg):
    cfg["contracts"]["datasets"]["asvspoof2019_la"]["raw"]["payload"]["guess_columns"] = True
    with pytest.raises(EPTTAError, match="unknown key"):
        check(cfg)


@pytest.mark.parametrize("layer", ["profile", "paths", "experiment"])
def test_illegal_override(cfg, layer):
    with pytest.raises(EPTTAError, match="illegal"):
        resolve(cfg, **{layer: {"schema_version": "0.1.0", "defaults": {"rho": .8}}})


def test_resolution_provenance_and_immutability(cfg):
    previous = copy.deepcopy(cfg)
    remote = read_document(ROOT / "configs/profiles/remote_a6000.yaml")
    resolved, sources = resolve(cfg, profile=remote, params={"steps": 5})
    assert resolved["defaults"]["steps"] == 5
    assert sources["runtime"] == "profile"
    assert sources["defaults.steps"] == "explicit_params"
    assert cfg == previous


def test_stage_inventory_no_checkpoint_or_label_requirement(cfg):
    cfg["runtime"] = read_document(ROOT / "configs/profiles/remote_a6000.yaml")["runtime"]
    cfg["paths"]["dataset_roots"]["asvspoof2019_la"] = "/fixture/audio"
    cfg["paths"]["protocol_files"]["asvspoof2019_la"] = "/fixture/protocol"
    result = report(cfg, "inventory", "resources")
    assert result["issues"] == []
    assert result["resource_io_performed"] is False and result["execution_ready"] is False


def test_d01_stage_dependencies(cfg):
    train = validate_stage(cfg, "source_training", "resources")
    fields = [i.field for i in train]
    assert any("recipe" in f for f in fields)
    assert any("split" in f for f in fields)
    assert any("initialization" in f for f in fields)
    assert not any("frozen_bundle" in f or "source_resources" in f for f in fields)
    assert not any("wavefake" in f for f in fields)
    build = validate_stage(cfg, "data_build", "resources")
    assert not any("split" in i.field or "snapshot" in i.field for i in build)
    adapt = validate_stage(cfg, "adaptation", "resources")
    assert any("frozen_bundle" in i.field for i in adapt)
    assert any("source_resources" in i.field for i in adapt)
    assert not any("raw" in i.field for i in adapt)


def test_aasist_does_not_require_ssl(cfg):
    cfg["selection"]["model_id"] = "aasist_source"
    assert not any("initialization" in i.field for i in validate_stage(cfg, "source_training", "resources"))


def test_source_training_external_dataset_forbidden(cfg):
    cfg["selection"]["dataset_ids"] = ["in_the_wild"]
    assert any(i.code == "PERMISSION_DENIED" and "dataset_ids" in i.field for i in validate_stage(cfg, "source_training", "resources"))


def test_approval_alone_is_not_lock(cfg):
    contract = cfg["contracts"]["recipe"]
    contract["status"] = "APPROVED"
    assert any(i.field.endswith("status") for i in check_contract(contract, "recipe", "recipe"))
    contract["status"] = "LOCKED"
    assert any("approval" in i.field for i in check_contract(contract, "recipe", "recipe"))


def test_changed_payload_rejects_approval(cfg):
    contract = cfg["contracts"]["recipe"]
    contract["status"] = "LOCKED"
    contract["approval"] = {"reviewer": "fixture reviewer", "approved_at": "2026-09-13T00:00:00Z", "report_ref": "fixture-report", "sample_evidence_ref": "fixture-samples", "content_sha256": content_hash(contract["payload"])}
    contract["payload"]["lr"] = 0.1
    assert any(i.field.endswith("content_sha256") for i in check_contract(contract, "recipe", "recipe"))


def test_local_permission_escalation_forbidden(cfg):
    cfg["runtime"]["permissions"]["real_data_io"] = True
    assert any(i.code == "PERMISSION_DENIED" for i in validate_stage(cfg))


def test_env_expansion_is_literal():
    assert expand_env("${ROOT}/audio", {"ROOT": "/data"}) == "/data/audio"
    for value in ("$(command)", "`command`", "${MISSING}"):
        with pytest.raises(EPTTAError):
            expand_env(value, {})


def test_all_templates_validate():
    for kind in ("models", "datasets", "methods"):
        assert read_document(ROOT / f"configs/{kind}/registry.yaml") == read_document(ROOT / f"src/eptta/catalogs/{kind}.json")
    for path in (ROOT / "configs/data_contracts").glob("*.example"):
        for kind, contract in read_document(path).items():
            check(contract, kind)
    check(read_document(ROOT / "configs/source_training_plan.yaml"), "source_training_plan")
    for path in (ROOT / "configs/training").glob("*.yaml"):
        check(read_document(path), "recipe")
    for path in (ROOT / "configs/preprocess").glob("*.example"):
        check(read_document(path), "preprocess")
    for path in (ROOT / "configs/splits").glob("*.example"):
        check(read_document(path), "split")


def test_training_resolver_does_not_inherit_ep_defaults(cfg):
    plan = read_document(ROOT / "configs/source_training_plan.yaml")
    recipe = read_document(ROOT / "configs/training/aasist.yaml")
    contracts = {**cfg["contracts"]["datasets"]["asvspoof2019_la"],
                 **{k: cfg["contracts"][k] for k in ("preprocess", "split", "architecture")}}
    result = resolve_training("aasist_source", plan, recipe, contracts, cfg["runtime"], cfg["paths"])
    assert "defaults" not in result and "lambda_keep" not in result
    assert result["recipe"]["payload"]["lr"] is None
    assert result["status"] == "PREVIEW_ONLY" and result["execution_ready"] is False
    assert result["model"]["pretrained_frontend"] is None
