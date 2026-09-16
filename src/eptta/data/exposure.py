"""Append-only, hash-chained records of target-data and metric exposure."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from eptta.config.schema import check
from eptta.data.io import iter_jsonl, write_json_new
from eptta.errors import ContractError, DataError


EVENT_TYPES = frozenset({"metadata_view", "sample_debug", "model_scoring", "effect_view", "method_selection"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_exposure_ledger(path):
    ledger = Path(path)
    anchor = Path(str(ledger) + ".head.json")
    previous = None
    count = 0
    if not ledger.exists():
        if anchor.exists():
            raise DataError("exposure ledger is missing but its retained head anchor exists")
        return {"event_count": 0, "head_sha256": None}
    for number, event in enumerate(iter_jsonl(ledger), 1):
        required = {"schema_version", "event_id", "recorded_at", "dataset_id", "scope",
                    "event_type", "artifact_ref", "artifact_sha256", "effect_visible",
                    "decision_impact", "previous_event_sha256", "event_sha256"}
        if set(event) != required or event.get("schema_version") != "0.1.0":
            raise DataError(f"exposure event {number} has an invalid field set/version")
        try:
            check(event, "exposure_event")
        except Exception as exc:
            raise DataError(f"exposure event {number} fails schema: {exc}") from exc
        if event["event_type"] not in EVENT_TYPES or type(event["effect_visible"]) is not bool:
            raise DataError(f"exposure event {number} has invalid type/visibility")
        if event["previous_event_sha256"] != previous:
            raise DataError(f"exposure history was truncated or reordered at event {number}")
        claimed = event.pop("event_sha256")
        actual = _digest(event)
        event["event_sha256"] = claimed
        if claimed != actual:
            raise DataError(f"exposure event {number} hash mismatch")
        previous = claimed
        count += 1
    if anchor.exists():
        with anchor.open(encoding="utf-8") as stream:
            expected = json.load(stream)
        if expected != {"schema_version": "0.1.0", "event_count": count, "head_sha256": previous}:
            raise DataError("exposure history was silently cleared, truncated, or replaced")
    return {"event_count": count, "head_sha256": previous}


def append_exposure_event(path, event):
    ledger = Path(path)
    state = validate_exposure_ledger(ledger)
    required = {"recorded_at", "dataset_id", "scope", "event_type", "artifact_ref",
                "artifact_sha256", "effect_visible", "decision_impact"}
    if set(event) != required:
        raise ContractError("exposure event has unknown or missing fields")
    if event["event_type"] not in EVENT_TYPES or type(event["effect_visible"]) is not bool:
        raise ContractError("invalid exposure event type/visibility")
    value = {"schema_version": "0.1.0", "event_id": f"exposure-{state['event_count'] + 1:06d}",
             **event, "previous_event_sha256": state["head_sha256"]}
    value["event_sha256"] = _digest(value)
    check(value, "exposure_event")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    anchor = Path(str(ledger) + ".head.json")
    anchor_value = {"schema_version": "0.1.0", "event_count": state["event_count"] + 1,
                    "head_sha256": value["event_sha256"]}
    fd, temporary = tempfile.mkstemp(prefix="." + anchor.name, dir=str(anchor.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(anchor_value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, anchor)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return value
