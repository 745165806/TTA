"""Duplicate-key-safe JSON/YAML reader for ordinary experiment files."""
import json
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
