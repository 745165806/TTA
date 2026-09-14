"""Read-only layout inspection. Observations are never approvals."""
import collections
import os
from datetime import datetime, timezone
from pathlib import Path

from eptta.data.io import sha256_file
from eptta.errors import ResourceError

PROTOCOL_SUFFIXES = frozenset({".csv", ".tsv", ".txt", ".json", ".jsonl", ".keys"})
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac"})


def _files(root):
    for base, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not Path(base, d).is_symlink())
        for name in sorted(names):
            yield Path(base, name)


def _sample_lines(path, limit=5, max_chars=1000):
    lines = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.strip():
                    lines.append(line.rstrip("\r\n")[:max_chars])
                if len(lines) == limit:
                    break
    except OSError as exc:
        return [], str(exc)
    return lines, None


def inspect_dataset(dataset_id, root, protocol_source):
    root = Path(root).resolve()
    protocol_source = Path(protocol_source).resolve()
    if not root.is_dir():
        raise ResourceError(f"dataset root is not a directory: {root}")
    if not protocol_source.exists():
        raise ResourceError(f"protocol source does not exist: {protocol_source}")
    extensions = collections.Counter()
    audio_count = 0
    symlink_count = 0
    for path in _files(root):
        extensions[path.suffix.lower() or "<none>"] += 1
        audio_count += path.suffix.lower() in AUDIO_SUFFIXES
        symlink_count += path.is_symlink()
    protocol_paths = [protocol_source] if protocol_source.is_file() else [
        p for p in _files(protocol_source) if p.suffix.lower() in PROTOCOL_SUFFIXES]
    protocols = []
    for path in protocol_paths:
        sample, error = _sample_lines(path)
        protocols.append({"path": str(path), "relative_to_protocol_source": path.name if protocol_source.is_file()
                          else path.relative_to(protocol_source).as_posix(), "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path), "sample_lines": sample, "sample_error": error})
    return {"dataset_id": dataset_id, "root": str(root), "protocol_source": str(protocol_source),
            "audio_file_count": audio_count, "extension_counts": dict(sorted(extensions.items())),
            "symlink_file_count": symlink_count, "protocol_files": protocols,
            "observed_only": True, "contract_status": "UNRESOLVED"}


def inspect_selected(dataset_ids, roots, protocol_files):
    datasets = {}
    for dataset_id in dataset_ids:
        if not roots.get(dataset_id) or not protocol_files.get(dataset_id):
            raise ResourceError(f"missing root/protocol binding for selected dataset: {dataset_id}")
        datasets[dataset_id] = inspect_dataset(dataset_id, roots[dataset_id], protocol_files[dataset_id])
    return {"schema_version": "0.1.0", "status": "INVENTORIED",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "read_only": True, "datasets": datasets}
