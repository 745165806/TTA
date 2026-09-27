"""Selected development labels; import only in supervised/offline analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "itw": ROOT / "experiments/target10_selection/manifests/inwild_target10.json",
    "wavefake": ROOT / "experiments/capacity_audit/manifests/wavefake_capacity_labels.json",
}


def load_labels(domain: str, ids: list[str]) -> np.ndarray:
    doc = json.loads(PATHS[domain].read_text(encoding="utf-8"))
    mapping = {row["sample_id"]: row["label"] for row in doc["records"]}
    if len(mapping) != len(ids) or set(mapping) != set(ids) or set(mapping.values()) != {0, 1}:
        raise ValueError("selected development label coverage invalid")
    return np.asarray([mapping[sid] for sid in ids], dtype=np.int64)
