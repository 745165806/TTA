"""Read-only registry. Registration does not imply an executable implementation."""
from pathlib import Path

from eptta.config.schema import read_document
from eptta.errors import EPTTAError, NotImplementedStage


def registry(kind):
    if kind not in ("models", "datasets", "methods"):
        raise EPTTAError(f"unknown registry: {kind}")
    data = read_document(Path(__file__).parent / "catalogs" / f"{kind}.json")
    if set(data) != {"registry_version", kind} or data["registry_version"] != "0.1.0":
        raise EPTTAError("invalid registry header")
    return data[kind]


def get_spec(kind, identifier):
    entries = registry(kind)
    if identifier not in entries:
        raise EPTTAError(f"unknown {kind} ID: {identifier}")
    return entries[identifier]


_EP_TTA_IDENTITY = {"route": "feature_cache", "parameterization": "matrix", "objective": "view_variance",
                    "regularizer": "margin", "subspace": "response", "final_output": "original",
                    "reset_policy": "per_sample", "projection": "frobenius"}

_TASKAWARE_IDENTITY = {"route": "feature_cache", "parameterization": "matrix",
                       "objective": "calibrated_pseudo_bce+task_logit_consistency",
                       "regularizer": "source_logit+parameter_l2", "subspace": "response",
                       "final_output": "original", "reset_policy": "per_sample",
                       "projection": "frobenius"}

_CALIBRATED_SELECTIVE_IDENTITY = {
    "route": "feature_cache", "parameterization": "matrix",
    "objective": "target_calibrated_pseudo_bce+task_logit_consistency",
    "regularizer": "source_logit+parameter_l2", "subspace": "response",
    "final_output": "original", "reset_policy": "per_sample",
    "projection": "frobenius",
}


def validate_selection(cfg):
    sel = cfg["selection"]
    get_spec("models", sel["model_id"])
    method = get_spec("methods", sel["method_id"])
    for identifier in sel["dataset_ids"]:
        get_spec("datasets", identifier)
    method_id = sel["method_id"]
    if method_id == "ep_tta":
        expected = _EP_TTA_IDENTITY
    elif method_id == "ep_tta_taskaware_v1":
        expected = _TASKAWARE_IDENTITY
    elif method_id == "ep_tta_calibrated_selective_v1":
        expected = _CALIBRATED_SELECTIVE_IDENTITY
    else:
        raise NotImplementedStage(f"{method_id} is registered TODO; its dedicated configuration is not implemented")
    if any(method.get(k) != v for k, v in expected.items()):
        raise EPTTAError(f"{method_id} method identity mismatch")
