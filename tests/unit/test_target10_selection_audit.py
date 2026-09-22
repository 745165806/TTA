import importlib.util
from pathlib import Path

import pytest


COMMON_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "target10_selection"
    / "scripts"
    / "_common.py"
)
SPEC = importlib.util.spec_from_file_location("target10_selection_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)


def test_target10_pipeline_rejects_numeric_fallback():
    ok = {"status": "ok", "score": 1.0}
    assert COMMON.require_success(ok, "sample-ok", "ep_tta_guarded") is ok

    failed = {
        "status": "fallback_numeric",
        "error_type": "FloatingPointError",
        "error_message": "non-finite mechanism loss",
    }
    with pytest.raises(RuntimeError, match="sample-bad.*fallback_numeric.*non-finite"):
        COMMON.require_success(failed, "sample-bad", "ep_tta_guarded")
