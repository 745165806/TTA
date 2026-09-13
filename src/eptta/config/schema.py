"""Strict, small JSON Schema subset used by this project's own schemas.

No remote schema resolution, code evaluation, implicit defaults or coercion.
"""
import json
import math
from pathlib import Path

from eptta.errors import EPTTAError, MissingDependency


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EPTTAError(f"duplicate key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise EPTTAError(f"non-finite JSON number: {value}")


def loads(text):
    try:
        return json.loads(text, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except json.JSONDecodeError:
        pass
    try:
        import yaml
    except ImportError as exc:
        raise MissingDependency("YAML syntax requires PyYAML; shipped .yaml templates use JSON syntax") from exc

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        # Reject merge keys too: config layering must be explicit.
        return _unique([(loader.construct_object(k, deep=True),
                         loader.construct_object(v, deep=True)) for k, v in node.value])

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        return yaml.load(text, Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise EPTTAError(f"invalid YAML: {exc}") from exc


def read_document(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def validate(value, spec, path="$"):
    """Collect all violations, including nested unknown keys and bool-as-int."""
    errors = []
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    allowed = spec.get("type", list(types))
    allowed = [allowed] if isinstance(allowed, str) else allowed
    if not any(isinstance(value, types[t]) and
               not (t in ("integer", "number") and isinstance(value, bool)) for t in allowed):
        return [f"{path}: expected {allowed}"]
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}: must be finite")
    if "enum" in spec and not any(type(value) is type(v) and value == v for v in spec["enum"]):
        errors.append(f"{path}: invalid enum {value!r}")
    if isinstance(value, dict):
        props = spec.get("properties", {})
        errors.extend(f"{path}.{k}: required" for k in spec.get("required", []) if k not in value)
        for key, item in value.items():
            if key in props:
                errors.extend(validate(item, props[key], f"{path}.{key}"))
            elif spec.get("additionalProperties", True) is False:
                errors.append(f"{path}.{key}: unknown key")
            elif isinstance(spec.get("additionalProperties"), dict):
                errors.extend(validate(item, spec["additionalProperties"], f"{path}.{key}"))
    elif isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if spec.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            errors.append(f"{path}: duplicate items")
        for i, item in enumerate(value):
            errors.extend(validate(item, spec.get("items", {}), f"{path}[{i}]"))
    elif isinstance(value, str) and len(value) < spec.get("minLength", 0):
        errors.append(f"{path}: empty string")
    elif type(value) in (int, float):
        for op, failed in (("minimum", lambda v: value < v), ("maximum", lambda v: value > v),
                           ("exclusiveMinimum", lambda v: value <= v),
                           ("exclusiveMaximum", lambda v: value >= v)):
            if op in spec and failed(spec[op]):
                errors.append(f"{path}: violates {op}={spec[op]}")
    return errors


def check(value, schema_name="config"):
    spec = read_document(Path(__file__).parents[1] / "schemas" / f"{schema_name}.json")
    errors = validate(value, spec)
    if errors:
        raise EPTTAError("; ".join(errors))
    return value
