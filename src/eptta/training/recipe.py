"""Resolve and validate hash-bound source-training recipes."""
from copy import deepcopy
from pathlib import Path

from eptta.config.schema import check, read_document
from eptta.config.validate import check_contract, content_hash
from eptta.data.io import sha256_file, write_json_new
from eptta.data.source_manifests import validate_source_snapshot
from eptta.errors import ContractError, DataError, ResourceError
from eptta.models.author import inspect_author_repository


AUTHOR_TRAIN_ENTRYPOINTS = {
    "aasist_source": "main.py",
    "ssl_aasist_source": "main_SSL_LA.py",
}


def _training_orchestration_identity(model_id, source_repo_ref):
    """Bind the project worker that replaces author train/eval orchestration."""
    root = Path(__file__).parents[3]
    worker = root / "workers/source_train_bridge.py"
    compat = root / "workers/compat/author_training.py"
    author_entrypoint = Path(source_repo_ref) / AUTHOR_TRAIN_ENTRYPOINTS[model_id]
    for path in (worker, compat, author_entrypoint):
        if not path.is_file():
            raise ResourceError("training orchestration input is missing: %s" % path)
    payload = {
        "kind": "project_source_worker_replaces_author_train_eval_orchestration",
        "author_training_entrypoint": AUTHOR_TRAIN_ENTRYPOINTS[model_id],
        "author_training_sha256": sha256_file(author_entrypoint),
        "project_worker_ref": str(worker.resolve()),
        "project_worker_sha256": sha256_file(worker),
        "project_compat_ref": str(compat.resolve()),
        "project_compat_sha256": sha256_file(compat),
        "allowed_roles": ["fit", "source_val"],
        "forbidden_roles": ["select", "cal0", "audit", "control_test", "target_test", "cal1"],
        "author_eval_path_disabled": True,
    }
    return {"payload": payload, "patch_sha256": content_hash(payload)}


def _locked_preprocess(path):
    value = read_document(path)
    issues = check_contract(value, "preprocess", "preprocess")
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))
    return value


def resolve_training_recipe(model_id, snapshot_ref, preprocess_ref, template_ref,
                            source_repo_ref, data_roots, initialization_ref=None, output=None):
    """Bind reviewed inputs while leaving unresolved scientific choices unresolved."""
    source = validate_source_snapshot(snapshot_ref)
    preprocess = _locked_preprocess(preprocess_ref)
    architecture = inspect_author_repository(model_id, source_repo_ref)
    template = read_document(template_ref)
    if template.get("schema_version") != "0.1.0":
        raise ContractError("training recipe template version mismatch")
    # Accept the compact contract templates shipped by the current repository.
    if "payload" in template:
        payload = deepcopy(template["payload"])
    else:
        training = template.get("training", {})
        payload = {"optimizer": training.get("optimizer"), "lr": training.get("lr"),
                   "max_epochs": training.get("max_epochs"), "scheduler": training.get("scheduler"),
                   "loss": training.get("loss"), "class_weights_by_name": training.get("class_weights_by_name"),
                   "sampler_policy": training.get("sampler_policy"),
                   "augmentation_recipe_ref": training.get("augmentation_recipe_ref"),
                   "trainable_scope": template.get("trainable_scope"),
                   "selection_metric": "source_val_eer", "tie_break": "earliest_epoch",
                   "weight_decay": training.get("weight_decay")}
    initialization = None
    if model_id == "ssl_aasist_source":
        if not initialization_ref:
            raise ResourceError("SSL-AASIST requires an explicit generic SSL initialization")
        init_path = Path(initialization_ref)
        if not init_path.is_file():
            raise ResourceError("generic SSL initialization is missing: %s" % init_path)
        initialization = {"artifact_ref": str(init_path.resolve()), "sha256": sha256_file(init_path),
                          "scope": "generic_ssl_frontend_only",
                          "pretraining_provenance": "user_bound_generic_xlsr"}
    elif initialization_ref:
        raise ContractError("AASIST must use native initialization")
    required_roots = sorted(set(source["roles"]["fit"]["root_keys"]) |
                            set(source["roles"]["source_val"]["root_keys"]))
    missing_roots = [key for key in required_roots if not data_roots.get(key)]
    if missing_roots:
        raise ResourceError("source manifest roots are unbound: %s" % missing_roots)
    source_roots = {key: data_roots[key] for key in required_roots}
    payload.update({
        "model_id": model_id,
        "class_index_map": architecture["class_index_map"],
        "bindings": {"source": source, "preprocess_ref": str(Path(preprocess_ref).resolve()),
                     "preprocess_hash": preprocess["approval"]["content_sha256"],
                     "architecture": architecture, "initialization": initialization,
                     "data_roots": source_roots,
                     "training_orchestration": _training_orchestration_identity(model_id, source_repo_ref)},
        "runtime": payload.get("runtime") if "runtime" in payload else None,
    })
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval": None, "payload": payload}
    check(proposal, "recipe")
    if output:
        write_json_new(output, proposal)
    return proposal


def load_locked_training_recipe(path):
    document = read_document(path)
    # Resolver annotations must be removed before approval, preventing a proposal
    # from being passed directly to a worker.
    if set(document) != {"schema_version", "status", "approval", "payload"}:
        raise ContractError("locked recipe has unknown top-level fields")
    issues = check_contract(document, "recipe", "recipe")
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))
    payload = document["payload"]
    required = {"model_id", "class_index_map", "bindings", "runtime"}
    if not required.issubset(payload) or payload["bindings"] is None or payload["runtime"] is None:
        raise ContractError("recipe is approved but lacks resolved execution bindings/runtime")
    if payload["model_id"] not in ("aasist_source", "ssl_aasist_source"):
        raise ContractError("recipe model is unregistered")
    if payload["class_index_map"] != {"spoof": 0, "bonafide": 1}:
        raise ContractError("recipe native class map does not match audited author source")
    source = payload["bindings"].get("source", {})
    for role in ("fit", "source_val"):
        item = source.get("roles", {}).get(role, {})
        ref = Path(item.get("manifest_ref", ""))
        if not ref.is_file() or sha256_file(ref) != item.get("manifest_sha256"):
            raise DataError("locked %s manifest is missing or changed" % role)
    orchestration = payload["bindings"].get("training_orchestration")
    expected_orchestration = _training_orchestration_identity(
        payload["model_id"], payload["bindings"]["architecture"]["repository_ref"])
    if orchestration != expected_orchestration:
        raise DataError("training orchestration patch changed after recipe lock")
    init = payload["bindings"].get("initialization")
    if payload["model_id"] == "ssl_aasist_source":
        if not init or init.get("scope") != "generic_ssl_frontend_only":
            raise ContractError("SSL recipe lacks generic-only initialization")
        if sha256_file(init["artifact_ref"]) != init.get("sha256"):
            raise DataError("generic SSL initialization changed after recipe lock")
    elif init is not None:
        raise ContractError("AASIST recipe cannot bind initialization weights")
    return document


def recipe_hash(recipe):
    return content_hash(recipe["payload"])
