"""Extract unmodified Appendix A/B reference code; fail rather than overwrite."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
design = (ROOT / "docs/DESIGN.md").read_text(encoding="utf-8")
for marker, filename in [("CORE_REFERENCE", "core_reference.py"), ("BATCH_REFERENCE", "batch_reference.py")]:
    code = design.split(f"# {marker}_BEGIN\n", 1)[1].split(f"# {marker}_END", 1)[0]
    dest = ROOT / "tests/reference" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(code)
    print(f"{filename}: {hashlib.sha256(code.encode()).hexdigest()}")
