"""Small scientific capability and path boundaries."""
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath

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
    root_path = Path(root).resolve()
    resolved = (root_path / safe_relative(relative)).resolve()
    if not resolved.is_relative_to(root_path):
        raise PermissionDenied("resolved path escapes root")
    return resolved


@dataclass(frozen=True, slots=True)
class TargetInputManifest:
    schema_version: str
    sample_id: str
    sample_index: int
    root_key: str
    audio_relpath: str
    split_role: str

    def __post_init__(self):
        if self.schema_version not in ("0.1.0", "0.2.0"):
            raise ContractError("invalid target manifest version")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ContractError("sample_index must be a non-negative integer")
        if self.split_role not in ROLES:
            raise ContractError("unknown split role")
        safe_relative(self.audio_relpath)

    @classmethod
    def from_dict(cls, record):
        if set(record) != {field.name for field in fields(cls)}:
            raise PermissionDenied("target manifest contains forbidden or missing fields")
        return cls(**record)


def project_target(record):
    return TargetInputManifest.from_dict({field.name: record[field.name] for field in fields(TargetInputManifest)})


class ExplicitLabelMapper:
    def map_label(self, value, policy):
        required = {"policy_id", "raw_to_canonical", "unknown_policy"}
        if not isinstance(policy, dict) or required - set(policy):
            raise ContractError("label policy is incomplete")
        mapped = None
        if type(value) in (str, int):
            mapped = policy["raw_to_canonical"].get(str(value))
        if mapped is not None and mapped not in (0, 1):
            raise ContractError("canonical label must be 0 or 1")
        if mapped is None and policy["unknown_policy"] == "error":
            raise ContractError("unmapped raw label: %r" % value)
        return LabelMapping(value, mapped, "explicit_mapping" if mapped is not None else "quarantine_unmapped",
                            policy["policy_id"])


def require_role(role, purpose):
    allowed = {"source_gradient": {"fit"}, "source_selection": {"source_val"},
               "source_resources": {"fit"}, "threshold": {"cal0"}, "method_selection": {"select"}}
    if purpose not in allowed or role not in allowed[purpose]:
        raise PermissionDenied("%s is forbidden for %s" % (role, purpose))
