"""Hash-bound descriptions of the two unmodified author architectures."""
from dataclasses import dataclass
from pathlib import Path
import subprocess

from eptta.data.io import sha256_file
from eptta.errors import ContractError, ResourceError


@dataclass(frozen=True)
class AuthorArchitecture:
    model_id: str
    repository: str
    commit: str
    entrypoint: str
    entrypoint_sha256: str
    class_index_map: dict
    embedding_dim: int
    initialization_scope: str


ARCHITECTURES = {
    "aasist_source": AuthorArchitecture(
        "aasist_source", "https://github.com/clovaai/aasist.git",
        "a04c9863f63d44471dde8a6abcb3b082b07cd1d1", "models/AASIST.py",
        "9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50",
        {"spoof": 0, "bonafide": 1}, 160, "native_initialization"),
    "ssl_aasist_source": AuthorArchitecture(
        "ssl_aasist_source", "https://github.com/TakHemlata/SSL_Anti-spoofing.git",
        "4acaa61dcef5f7610f43aa4d0b29c4559b970cd2", "model.py",
        "08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6",
        {"spoof": 0, "bonafide": 1}, 160, "generic_ssl_frontend_only"),
}


def get_author_architecture(model_id):
    try:
        return ARCHITECTURES[model_id]
    except KeyError as exc:
        raise ContractError("unregistered author architecture: %s" % model_id) from exc


def inspect_author_repository(model_id, repository_ref):
    spec = get_author_architecture(model_id)
    root = Path(repository_ref)
    entrypoint = root / spec.entrypoint
    if not entrypoint.is_file():
        raise ResourceError("author entrypoint is missing: %s" % entrypoint)
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResourceError("cannot identify author repository commit: %s" % root) from exc
    digest = sha256_file(entrypoint)
    if commit != spec.commit or digest != spec.entrypoint_sha256:
        raise ContractError("author source identity mismatch for %s" % model_id)
    return {"schema_version": "0.1.0", "status": "VERIFIED", "model_id": model_id,
            "repository_ref": str(root.resolve()), "repository": spec.repository,
            "repo_commit": commit, "entrypoint": spec.entrypoint,
            "entrypoint_sha256": digest, "class_index_map": dict(spec.class_index_map),
            "embedding_dim": spec.embedding_dim, "initialization_scope": spec.initialization_scope}


def canonical_to_native(labels, class_index_map):
    if set(class_index_map) != {"bonafide", "spoof"} or set(class_index_map.values()) != {0, 1}:
        raise ContractError("native class map must be a bonafide/spoof bijection")
    native = []
    for label in labels:
        if type(label) is not int or label not in (0, 1):
            raise ContractError("canonical labels must be 0/1 integers")
        native.append(class_index_map["bonafide" if label == 0 else "spoof"])
    return native


def class_weights_native(class_weights_by_name, class_index_map):
    if set(class_weights_by_name) != {"bonafide", "spoof"}:
        raise ContractError("class weights must be keyed by semantic class names")
    result = [None, None]
    for name, value in class_weights_by_name.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ContractError("class weights must be positive")
        result[class_index_map[name]] = float(value)
    return result
