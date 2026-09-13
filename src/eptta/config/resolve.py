import copy
import re
from pathlib import Path

from eptta.config.schema import check, read_document
from eptta.errors import EPTTAError


def expand_env(value, environ):
    """Explicit helper for future remote binding; never called by local resolution."""
    if value is None:
        return None
    pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
    def replace(match):
        key = match.group(1)
        if not environ.get(key):
            raise EPTTAError(f"unresolved environment variable: {key}")
        return environ[key]
    result = re.sub(pattern, replace, value)
    if "$" in result or "`" in result:
        raise EPTTAError("only literal ${ENV_NAME} expansion is supported")
    return result


def resolve(base, *, experiment=None, profile=None, paths=None, method=None, params=None):
    cfg = copy.deepcopy(check(base))
    provenance = {"$": "base"}
    for name, overlay, allowed in (
        ("experiment", experiment, {"selection"}),
        ("profile", profile, {"runtime"}),
        ("paths", paths, {"paths"}),
    ):
        if overlay is None:
            continue
        if not isinstance(overlay, dict) or set(overlay) - allowed - {"schema_version"}:
            raise EPTTAError(f"illegal {name} override")
        if overlay.get("schema_version") != "0.1.0":
            raise EPTTAError(f"invalid {name} schema_version")
        for key in allowed & overlay.keys():
            cfg[key] = copy.deepcopy(overlay[key])
            provenance[key] = name
    if method is not None:
        cfg["selection"]["method_id"] = method
        provenance["selection.method_id"] = "registered_method"
    if params is not None:
        if set(params) - {"rank", "steps", "lr", "rho", "gamma", "regularizer_weight"}:
            raise EPTTAError("illegal method parameter")
        cfg["defaults"].update(params)
        provenance.update({f"defaults.{k}": "explicit_params" for k in params})
    check(cfg)
    from eptta.registry import validate_selection
    validate_selection(cfg)
    return cfg, provenance


def load_resolved(base, **overlays):
    return resolve(read_document(base), **{
        key: read_document(value) if value is not None else None
        for key, value in overlays.items()
    })


def resolve_training(model_id, plan, recipe, contracts, runtime, paths):
    """Independent source preview, not a worker job or recipe lock (L4).

    EP defaults and target parameters never enter this resolver.
    """
    from eptta.registry import get_spec
    from eptta.config.schema import validate
    check(plan, "source_training_plan")
    check(recipe, "recipe")
    if model_id not in plan["model_ids"]:
        raise EPTTAError("model is not selected in the source training plan")
    required = {"raw", "label", "group", "preprocess", "split", "architecture", "snapshot"}
    if set(contracts) != required:
        raise EPTTAError("source training requires explicit raw/label/group/preprocess/split/architecture/snapshot contracts")
    for kind, contract in contracts.items():
        check(contract, kind)
    spec = read_document(Path(__file__).parents[1] / "schemas/config.json")
    for key, value in (("runtime", runtime), ("paths", paths)):
        errors = validate(value, spec["properties"][key], key)
        if errors:
            raise EPTTAError("; ".join(errors))
    if not {"fit", "source_val"}.issubset(get_spec("datasets", plan["source_dataset_id"])["roles"]):
        raise EPTTAError("source dataset is not permitted for fit/source_val")
    return copy.deepcopy({"schema_version": "0.1.0", "status": "PREVIEW_ONLY", "model_id": model_id,
                          "model": get_spec("models", model_id), "source_plan": plan, "recipe": recipe,
                          "contracts": contracts, "runtime": runtime, "paths": paths,
                          "field_sources": {"model": "model_registry", "source_plan": "source_training_plan",
                                            "recipe": "selected_recipe", "contracts": "explicit_contracts",
                                            "runtime": "profile", "paths": "private_paths"},
                          "execution_ready": False})
