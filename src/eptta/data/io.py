"""Small deterministic JSON/JSONL and atomic-output helpers."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

from eptta.errors import DataError, EPTTAError


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    from eptta.config.schema import read_document
    return read_document(path)


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for row, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid JSONL at {path}:{row}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataError(f"JSONL record must be an object at {path}:{row}")
            yield value


def write_json_new(path, value):
    destination = Path(path)
    if destination.exists():
        raise EPTTAError(f"output exists; overwrite is forbidden: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class AtomicDirectory:
    """Publish a new directory only after all validation succeeds."""
    def __init__(self, destination):
        self.destination = Path(destination)
        self.temporary = None

    def __enter__(self):
        if self.destination.exists():
            raise EPTTAError(f"output exists; overwrite is forbidden: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(tempfile.mkdtemp(prefix=f".{self.destination.name}.", dir=self.destination.parent))
        return self.temporary

    def __exit__(self, kind, value, traceback):
        if kind is None:
            os.replace(self.temporary, self.destination)
            return False
        if self.temporary and self.temporary.exists():
            import shutil
            shutil.rmtree(self.temporary)
        return False
