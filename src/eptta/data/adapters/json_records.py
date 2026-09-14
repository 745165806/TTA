import json
from pathlib import Path

from eptta.data.adapters.base import field_by_path, protocol_context
from eptta.data.contracts import RawRecord
from eptta.data.io import sha256_file
from eptta.errors import DataError


class JsonRecordsAdapter:
    def iter_file(self, path, payload):
        source = Path(path)
        digest = sha256_file(source)
        with source.open(encoding=payload["encoding"]) as stream:
            if payload["format"] == "json":
                try:
                    values = json.load(stream)
                except json.JSONDecodeError as exc:
                    raise DataError(f"invalid JSON at {source}: {exc}") from exc
                if isinstance(values, dict):
                    values = [values]
            else:
                values = []
                for row, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            values.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise DataError(f"invalid JSONL at {source}:{row}: {exc}") from exc
        if not isinstance(values, list):
            raise DataError(f"JSON protocol must contain object or array: {source}")
        context = protocol_context(source, payload.get("protocol_contexts"))
        for index, value in enumerate(values, 1):
            if not isinstance(value, dict):
                raise DataError(f"JSON record must be object at {source}:{index}")
            fields = {logical: field_by_path(value, path) for logical, path in payload["json_paths"].items()}
            yield RawRecord(f"{source}:{index}", fields, digest, str(source), index, context)
