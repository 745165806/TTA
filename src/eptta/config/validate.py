"""Offline stage-aware readiness checks. Never stat/open any resource binding."""
import hashlib
import json
import re
from dataclasses import asdict, dataclass

from eptta.config.schema import check
from eptta.errors import EPTTAError
from eptta.registry import get_spec, validate_selection

STAGES = ("development", "inventory", "data_build", "source_training", "frozen_extract", "adaptation", "evaluation")


@dataclass(frozen=True)
class Issue:
    code: str
    field: str
    message: str


def content_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def unresolved(value):
    return value is None or value == "" or (isinstance(value, str) and
        any(marker in value.lower() for marker in ("${", "/replace", "placeholder", "todo", "<replace", "example.invalid")))


def check_contract(contract, kind, field):
    check(contract, kind)
    issues = []
    def fail(key, message):
        issues.append(Issue("BLOCKED_CONTRACT", f"{field}.{key}", message))
    if contract["status"] != "LOCKED":
        fail("status", "real execution requires a reviewed LOCKED contract")
    approval = contract["approval"]
    if approval is None:
        fail("approval", "reviewer, time, report, sampled evidence and payload hash required")
    else:
        for key, value in approval.items():
            if unresolved(value):
                fail(f"approval.{key}", "unresolved approval evidence")
        if approval["content_sha256"] != content_hash(contract["payload"]):
            fail("approval.content_sha256", "payload changed or review hash invalid")
        if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$", approval["approved_at"]):
            fail("approval.approved_at", "ISO timestamp with timezone required")
    payload = contract["payload"]
    optional = {"split": {"ratios", "counts"}, "raw": {"columns", "json_paths", "delimiter", "header"}}.get(kind, set())
    for key, value in payload.items():
        if key not in optional and unresolved(value):
            fail(f"payload.{key}", "unresolved required field; no inferred defaults")
        if key.endswith("sha256") and value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            fail(f"payload.{key}", "expected SHA-256 hex")
    if kind == "raw":
        if payload["format"] == "delimited":
            for key in ("columns", "delimiter", "header"):
                if payload[key] is None or payload[key] == {}:
                    fail(f"payload.{key}", "explicit delimited format contract required")
        elif payload["format"] in ("json", "jsonl", "sidecar") and not payload["json_paths"]:
            fail("payload.json_paths", "explicit JSON/sidecar paths required")
        if not payload["allowed_values"]:
            fail("payload.allowed_values", "empty label vocabulary")
    if kind == "label" and not payload["raw_to_canonical"]:
        fail("payload.raw_to_canonical", "explicit nonempty label mapping required")
    if kind == "group" and payload["group_quality"] in ("unknown", "synthetic_fixture"):
        fail("payload.group_quality", "real source grouping requires review")
    if kind == "architecture":
        mapping = payload["class_index_map"]
        if mapping is not None and set(mapping.values()) != {0, 1}:
            fail("payload.class_index_map", "native class map must be bijective")
        if payload["repo_commit"] is not None and not re.fullmatch(r"[0-9a-f]{40}", payload["repo_commit"]):
            fail("payload.repo_commit", "full author commit required")
    if kind == "recipe":
        if payload["lr"] is not None and payload["lr"] <= 0:
            fail("payload.lr", "learning rate must be positive")
        if payload["max_epochs"] is not None and payload["max_epochs"] < 1:
            fail("payload.max_epochs", "training budget must be positive")
    return issues


def validate_stage(cfg, stage="development", level="structure"):
    if stage not in STAGES or level not in ("structure", "resources"):
        raise EPTTAError("unknown stage or validation level")
    check(cfg)
    validate_selection(cfg)
    runtime = cfg["runtime"]
    permissions = runtime["permissions"]
    issues = []
    if runtime["profile_id"] == "local_dev" and (runtime["environment"] != "local" or any(permissions.values()) or runtime["device"] != "cpu" or runtime["physical_gpu_ids"]):
        issues.append(Issue("PERMISSION_DENIED", "runtime", "local_dev forbids real I/O, GPU, weights, network and submission"))
    if runtime["profile_id"] == "remote_a6000" and runtime["environment"] != "remote":
        issues.append(Issue("PERMISSION_DENIED", "runtime.environment", "remote profile must declare remote environment"))
    for name in ("network_download", "remote_submission"):
        if permissions[name]:
            issues.append(Issue("PERMISSION_DENIED", f"runtime.permissions.{name}", "automatic download/submission is disabled"))
    if level == "structure" or stage == "development":
        return issues
    if runtime["environment"] != "remote":
        issues.append(Issue("PERMISSION_DENIED", "runtime.environment", "real stages require remote execution; validation performs no remote I/O"))
    needed_permissions = ["real_data_io"]
    if stage in ("source_training", "frozen_extract"):
        needed_permissions.append("model_weight_io")
    if stage == "source_training":
        needed_permissions.append("real_source_training")
    for permission in needed_permissions:
        if not permissions[permission]:
            issues.append(Issue("PERMISSION_DENIED", f"runtime.permissions.{permission}", "permission disabled"))

    def require_path(field, value):
        if unresolved(value):
            issues.append(Issue("BLOCKED_CONTRACT", field, "binding is null or placeholder; not accessed"))

    datasets = cfg["selection"]["dataset_ids"]
    if stage in ("inventory", "data_build", "source_training", "frozen_extract"):
        for ds in datasets:
            require_path(f"paths.dataset_roots.{ds}", cfg["paths"]["dataset_roots"][ds])
            if stage in ("inventory", "data_build"):
                require_path(f"paths.protocol_files.{ds}", cfg["paths"]["protocol_files"][ds])
    contracts = cfg["contracts"]
    if stage in ("data_build", "source_training", "frozen_extract"):
        for ds in datasets:
            kinds = ["raw", "label", "group"]
            if stage != "data_build":
                kinds.append("snapshot")
            for kind in kinds:
                issues.extend(check_contract(contracts["datasets"][ds][kind], kind, f"contracts.datasets.{ds}.{kind}"))
    required = {
        "inventory": [], "data_build": [],
        "source_training": ["preprocess", "split", "architecture", "recipe"],
        "frozen_extract": ["preprocess", "split", "frozen_bundle"],
        "adaptation": ["frozen_bundle", "source_resources"],
        "evaluation": ["score_seal"],
    }[stage]
    if stage == "source_training":
        model = cfg["selection"]["model_id"]
        require_path("paths.work_root", cfg["paths"]["work_root"])
        repo = "ssl_aasist" if model == "ssl_aasist_source" else "aasist"
        require_path(f"paths.source_repos.{repo}", cfg["paths"]["source_repos"][repo])
        for ds in datasets:
            if not {"fit", "source_val"}.issubset(get_spec("datasets", ds)["roles"]):
                issues.append(Issue("PERMISSION_DENIED", f"selection.dataset_ids.{ds}", "external dataset is not approved for source training"))
        if model == "ssl_aasist_source":
            required.append("initialization")
            require_path("paths.generic_ssl_initialization", cfg["paths"]["generic_ssl_initialization"])
            scope = contracts["recipe"]["payload"]["trainable_scope"]
            if scope is not None and scope != "joint_ssl_frontend_backend_head":
                issues.append(Issue("BLOCKED_CONTRACT", "contracts.recipe.payload.trainable_scope", "SSL main model requires joint finetuning; frozen frontend needs a separate model ID"))
    for kind in required:
        issues.extend(check_contract(contracts[kind], kind, f"contracts.{kind}"))
    return issues


def report(cfg, stage, level):
    issues = validate_stage(cfg, stage, level)
    return {"schema_version": "0.1.0", "stage": stage, "level": level,
            "status": "BLOCKED_CONTRACT" if issues else "STRUCTURE_VALID" if level == "structure" else "CONTRACT_FIELDS_VALID",
            "resource_io_performed": False, "execution_ready": False,
            "issues": [asdict(issue) for issue in issues],
            "notice": "L0-L2 only; resource existence, review evidence and real workers are not verified"}
