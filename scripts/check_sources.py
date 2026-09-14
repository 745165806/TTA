"""Dependency-free source/schema audit. Does not import numerical code."""
import ast
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eptta.config.schema import read_document, check

paths = []
for directory in ("src", "tests", "scripts"):
    for path in sorted((ROOT / directory).rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
        paths.append(str(path.relative_to(ROOT)))
worker_paths = []
for path in sorted((ROOT / "workers").rglob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 7))
    worker_paths.append(str(path.relative_to(ROOT)))
check(read_document(ROOT / "configs/base.yaml"))
schemas = sorted((ROOT / "src/eptta/schemas").glob("*.json"))
for path in schemas:
    read_document(path)
result = {"schema_version": "0.1.0", "status": "PASS", "python": sys.version,
          "platform": platform.platform(), "syntax_target": "Python 3.10",
          "worker_syntax_target": "Python 3.7", "worker_syntax_files": worker_paths,
          "syntax_files": paths, "schema_files": len(schemas),
          "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("torch", "pytest", "numpy", "yaml", "setuptools")},
          "design_sha256": hashlib.sha256((ROOT / "docs/DESIGN.md").read_bytes()).hexdigest(),
          "tensor_execution": "NOT_RUN" if importlib.util.find_spec("torch") is None else "NOT_TESTED_BY_THIS_SCRIPT"}
print(json.dumps(result, ensure_ascii=False, indent=2))
