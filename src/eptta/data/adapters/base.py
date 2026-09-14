"""Shared, contract-driven parsing primitives; no inferred columns or labels."""
import fnmatch
from pathlib import Path

from eptta.errors import ContractError, DataError


def protocol_context(path, contexts):
    matches = [(pattern, value) for pattern, value in (contexts or {}).items()
               if fnmatch.fnmatch(Path(path).name, pattern) or fnmatch.fnmatch(str(path), pattern)]
    if len(matches) != 1:
        raise ContractError(f"protocol file must match exactly one reviewed context: {path}; matches={[m[0] for m in matches]}")
    return matches[0][1]


def select_protocols(candidates, globs):
    if not globs:
        raise ContractError("LOCKED raw contract requires nonempty protocol_globs")
    selected = []
    for candidate in candidates:
        value = candidate["path"] if isinstance(candidate, dict) else str(candidate)
        if any(fnmatch.fnmatch(Path(value).name, pattern) or fnmatch.fnmatch(value, pattern) for pattern in globs):
            selected.append(candidate)
    if not selected:
        raise DataError("reviewed protocol_globs matched no inventoried files")
    return selected


def field_by_path(value, path):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise DataError(f"missing reviewed JSON path: {path}")
        current = current[part]
    return current
