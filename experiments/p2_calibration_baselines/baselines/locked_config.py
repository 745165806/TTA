"""Strict loader for the P2 published-port configuration."""
import json
from pathlib import Path


PUBLISHED_METHODS = ("tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full")
ALL_METHODS = (*PUBLISHED_METHODS, "norm_only_audio")
REQUIRED_FIELDS = {
    "tent_audio_ep": ("method", "optimizer", "lr", "steps", "weight_decay"),
    "sar_audio_ep": ("method", "optimizer", "lr", "steps", "rho", "momentum",
                     "entropy_margin"),
    "memo_audio_ep_full": ("method", "optimizer", "lr", "steps", "niter",
                           "weight_decay"),
    "norm_only_audio": ("method", "optimizer", "lr", "steps", "control"),
}
OPTIMIZER_CONTRACTS = {
    "tent_audio_ep": "Adam",
    "sar_audio_ep": "SGD + SAM",
    "memo_audio_ep_full": "SGD",
    "norm_only_audio": "none",
}


def load_locked_document(path):
    """Return the complete validated config as path-independent JSON content."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("locked config is not readable JSON: %s" % path) from exc
    methods = document.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("locked config has no methods object")
    for method_id in ALL_METHODS:
        _validate_entry(methods, method_id)
    # JSON round-trip normalizes the content without calculating a digest.
    return json.loads(json.dumps(document, sort_keys=True, separators=(",", ":")))


def _validate_entry(methods, method_id):
    if method_id not in REQUIRED_FIELDS:
        raise ValueError("unsupported locked method: %s" % method_id)
    if method_id not in methods:
        raise ValueError("locked config has no method: %s" % method_id)
    entry = methods[method_id]
    if not isinstance(entry, dict):
        raise ValueError("locked config method is not an object: %s" % method_id)
    missing = [field for field in REQUIRED_FIELDS[method_id] if field not in entry]
    if missing:
        raise ValueError("locked config %s missing fields: %s" %
                         (method_id, ", ".join(missing)))
    if entry["method"] != method_id:
        raise ValueError("locked config method id mismatch: expected %s, got %r" %
                         (method_id, entry["method"]))
    expected_optimizer = OPTIMIZER_CONTRACTS[method_id]
    if entry["optimizer"] != expected_optimizer:
        raise ValueError("locked config optimizer mismatch for %s: expected %s" %
                         (method_id, expected_optimizer))
    if method_id == "memo_audio_ep_full" and entry["steps"] != entry["niter"]:
        raise ValueError("locked MEMO steps and niter must agree")
    return dict(entry)


def load_locked_method(path, method_id):
    """Return a validated method entry, failing closed on malformed input."""
    if method_id not in REQUIRED_FIELDS:
        raise ValueError("unsupported locked method: %s" % method_id)
    document = load_locked_document(path)
    return dict(document["methods"][method_id])


def adaptation_kwargs(method_id, entry):
    """Map a validated locked entry to explicit adaptation-call arguments."""
    if method_id == "tent_audio_ep":
        return {key: entry[key] for key in ("lr", "steps", "weight_decay")}
    if method_id == "sar_audio_ep":
        return {key: entry[key] for key in
                ("lr", "steps", "rho", "momentum", "entropy_margin")}
    if method_id == "memo_audio_ep_full":
        return {"lr": entry["lr"], "niter": entry["niter"],
                "weight_decay": entry["weight_decay"]}
    return {}


def invoke_locked(method_id, entry, adaptation_function, *args):
    """Invoke an adaptation function using only explicit locked parameters."""
    return adaptation_function(*args, **adaptation_kwargs(method_id, entry))
