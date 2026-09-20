"""Load ordinary source-training recipes while preserving scientific checks."""
from pathlib import Path

from eptta.config.schema import read_document
from eptta.errors import ContractError, DataError, ResourceError


def _required_file(value, field):
    path = Path(value or "")
    if not path.is_file():
        raise ResourceError("%s is missing: %s" % (field, path))
    return str(path.resolve())


def load_training_recipe(path):
    document = read_document(path)
    if not isinstance(document, dict) or "payload" not in document:
        raise ContractError("training recipe must contain payload")
    # New READY documents and historical LOCKED documents share the scientific
    # payload. Approval/digest fields in historical files are ignored.
    if document.get("status") not in ("READY", "LOCKED"):
        raise ContractError("training recipe must be READY (historical LOCKED is read-only compatible)")
    payload = document["payload"]
    required = {"model_id", "class_index_map", "bindings", "runtime", "optimizer", "lr",
                "max_epochs", "scheduler", "loss", "class_weights_by_name", "sampler_policy",
                "augmentation_recipe_ref", "trainable_scope", "weight_decay"}
    if not required.issubset(payload):
        raise ContractError("training recipe lacks resolved scientific/runtime fields")
    if payload["model_id"] not in ("aasist_source", "ssl_aasist_source"):
        raise ContractError("recipe model is unregistered")
    if payload["class_index_map"] != {"spoof": 0, "bonafide": 1}:
        raise ContractError("recipe native class map differs from the author model")
    bindings = payload["bindings"]
    source = bindings.get("source", {})
    for role in ("fit", "source_val"):
        item = source.get("roles", {}).get(role, {})
        item["manifest_ref"] = _required_file(item.get("manifest_ref"), role + " manifest")
        if not item.get("dataset_id"):
            # Historical snapshot files use snapshot_id as the ordinary dataset/split label.
            item["dataset_id"] = source.get("snapshot_id") or "legacy-explicit-manifest"
    architecture = bindings.get("architecture")
    if (not isinstance(architecture, dict) or not architecture.get("repository_ref") or
            architecture.get("model_id") != payload["model_id"]):
        raise ContractError("recipe architecture repository is missing")
    repository = Path(architecture["repository_ref"])
    if not repository.is_dir():
        raise ResourceError("author repository is missing: %s" % repository)
    # Historical architecture documents may carry approval/digest metadata.
    # Normalize only the ordinary construction/provenance fields into new jobs.
    bindings["architecture"] = {
        "model_id": payload["model_id"],
        "repository_ref": str(repository.resolve()),
        "repository_revision": architecture.get("repository_revision", architecture.get("repo_commit")),
        "entrypoint": architecture.get("entrypoint"),
        "class_index_map": dict(payload["class_index_map"]),
        "embedding_dim": architecture.get("embedding_dim", 160),
    }
    init = bindings.get("initialization")
    if payload["model_id"] == "ssl_aasist_source":
        if not init or init.get("scope") != "generic_ssl_frontend_only":
            raise ContractError("SSL recipe lacks generic-only initialization")
        init["artifact_ref"] = _required_file(init.get("artifact_ref"), "generic SSL initialization")
    elif init is not None:
        raise ContractError("AASIST recipe cannot bind initialization weights")
    preprocess = bindings.get("preprocess")
    if preprocess is None:
        preprocess_ref = bindings.get("preprocess_ref")
        if not preprocess_ref:
            raise ContractError("recipe must contain preprocess or preprocess_ref")
        preprocess = read_document(_required_file(preprocess_ref, "preprocess config"))
        if "payload" in preprocess:
            preprocess = preprocess["payload"]
        bindings["preprocess"] = preprocess
    if not isinstance(bindings.get("data_roots"), dict) or not bindings["data_roots"]:
        raise ContractError("recipe data_roots must be explicitly bound")
    return document


# Compatibility name for callers; no locking or digest validation is performed.
load_locked_training_recipe = load_training_recipe


def resolve_training_recipe(*_args, **_kwargs):
    raise ContractError("recipe proposal/approval is retired; create one resolved READY recipe")
