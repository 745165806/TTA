"""Explicit capability boundaries; no ambient target manifest in adaptation."""
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath

from eptta.config.schema import check
from eptta.config.validate import check_contract, content_hash
from eptta.data.contracts import LabelMapping
from eptta.errors import ContractError, PermissionDenied

ROLES = frozenset({"fit", "source_val", "select", "cal0", "audit", "control_test", "target_test", "cal1"})


def safe_relative(path):
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or "\x00" in path:
        raise PermissionDenied("audio path must be a safe POSIX relative path")
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute() or ".." in path.split("/"):
        raise PermissionDenied("absolute/traversing audio path rejected")
    return path


def resolve_under_root(root, relative):
    """For future authorized I/O only; detects symlinks escaping a real root."""
    root_path = Path(root).resolve()
    resolved = (root_path / safe_relative(relative)).resolve()
    if not resolved.is_relative_to(root_path):
        raise PermissionDenied("resolved path escapes root")
    return resolved


@dataclass(frozen=True, slots=True)
class TargetInputManifest:
    schema_version: str
    sample_id: str
    root_key: str
    audio_relpath: str
    input_sha256: str | None
    decode_profile_id: str
    probe_profile_id: str

    def __post_init__(self):
        if self.schema_version != "0.1.0":
            raise ContractError("invalid target manifest version")
        safe_relative(self.audio_relpath)

    @classmethod
    def from_dict(cls, record):
        if set(record) != {field.name for field in fields(cls)}:
            raise PermissionDenied("target manifest contains forbidden or missing fields")
        return cls(**record)


def project_target(record):
    """Trusted data-building boundary. Deliberate whitelist, never forwarding raw metadata."""
    return TargetInputManifest.from_dict({field.name: record[field.name] for field in fields(TargetInputManifest)})


class ExplicitLabelMapper:
    """Minimal contract primitive. Production protocol parsing remains L3.

    Raw numeric values require an explicit string representation in the reviewed
    mapping; None and bool are never coerced to labels.
    """
    def map_label(self, value, approved_policy):
        check(approved_policy, "label")
        issues = check_contract(approved_policy, "label", "label")
        if issues:
            raise ContractError("; ".join(i.message for i in issues))
        policy = approved_policy["payload"]
        mapped = None
        if type(value) in (str, int):
            mapped = policy["raw_to_canonical"].get(str(value))
        if mapped is None and policy["unknown_policy"] == "error":
            raise ContractError(f"unmapped raw label: {value!r}")
        return LabelMapping(value, mapped, "explicit_mapping" if mapped is not None else "quarantine_unmapped",
                            policy["policy_id"], content_hash(policy))


def require_role(role, purpose):
    allowed = {"source_gradient": {"fit"}, "source_selection": {"source_val"},
               "source_resources": {"fit"}, "threshold": {"cal0"}, "method_selection": {"select"}}
    if purpose not in allowed or role not in allowed[purpose]:
        raise PermissionDenied(f"{role} is forbidden for {purpose}")
