"""Verify the built wheel matches sources and print a reproducible input manifest."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
wheel = ROOT / "dist/ep_tta-0.1.0-py3-none-any.whl"
with zipfile.ZipFile(wheel) as archive:
    metadata = archive.read("ep_tta-0.1.0.dist-info/METADATA").decode()
    assert "Version: 0.1.0" in metadata.splitlines()
    assert "Requires-Python: >=3.10" in metadata.splitlines()
    checked = 0
    for path in (ROOT / "src/eptta").rglob("*"):
        if path.is_file() and path.suffix in (".py", ".json"):
            assert archive.read(path.relative_to(ROOT / "src").as_posix()) == path.read_bytes(), str(path)
            checked += 1
design = (ROOT / "docs/DESIGN.md").read_text(encoding="utf-8")
for marker, filename in [("CORE_REFERENCE", "core_reference.py"), ("BATCH_REFERENCE", "batch_reference.py")]:
    code = design.split(f"# {marker}_BEGIN\n", 1)[1].split(f"# {marker}_END", 1)[0]
    assert (ROOT / "tests/reference" / filename).read_text(encoding="utf-8") == code
files = [ROOT / name for name in ("pyproject.toml", "README.md", "AGENTS.md", "IMPLEMENTATION_STATUS.md", "docs/DESIGN.md", "docs/DECISIONS.md", "docs/TEST_REPORT.md")]
for directory in ("src/eptta", "configs", "tests", "scripts"):
    files += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
print(json.dumps({"schema_version": "0.1.0", "status": "PASS", "wheel_matched_source_files": checked,
                  "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "reference_extraction": "MATCHES_DESIGN_A_B", "files_sha256": hashes,
                  "remote_runs": "NOT_RUN", "tensor_tests": "NOT_RUN_MISSING_TORCH"}, ensure_ascii=False, indent=2))
