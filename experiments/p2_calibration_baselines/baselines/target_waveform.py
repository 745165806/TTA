#!/usr/bin/env python
"""Strict label-free target waveform loader (P2 published ports).

Rejects any label/canonical_label/attack/class/y field.  Reuses the project's
existing inference audio preprocessing (16 kHz, length 64600, repeat-or-crop).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))

import torch

from author_training import load_audio

ALLOWED_FIELDS = {"schema_version", "sample_id", "root_key", "audio_relpath",
                  "split_role", "sample_index"}
FORBIDDEN_LABEL_KEYS = {"label", "canonical_label", "attack", "class", "target", "y"}


class TargetWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_ref, data_roots, role="target_test", length=64600,
                 format="jsonl"):
        self.rows = []
        seen = set()
        if format == "json_records":
            doc = json.loads(Path(manifest_ref).read_text(encoding="utf-8"))
            records = doc.get("records")
            if not isinstance(records, list):
                raise ValueError("json_records manifest has no records list")
            for number, row in enumerate(records, 1):
                self._append(row, number, data_roots, role, seen)
        else:
            with open(manifest_ref, encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    self._append(json.loads(line), number, data_roots, role, seen)
        if not self.rows:
            raise ValueError("empty manifest")
        self.length = length

    def _append(self, row, number, data_roots, role, seen):
        if not ALLOWED_FIELDS.issubset(row) or set(row) - ALLOWED_FIELDS:
            raise ValueError("unsafe manifest row %d: unknown fields" % number)
        for key in row:
            if key.lower() in FORBIDDEN_LABEL_KEYS or "label" in key.lower():
                raise ValueError("label field present in row %d: %s" % (number, key))
        if row["split_role"] != role or row["sample_id"] in seen:
            raise ValueError("mismatched role or duplicate ID in row %d" % number)
        if row["root_key"] not in data_roots:
            raise ValueError("unbound data root: %s" % row["root_key"])
        root = os.path.realpath(data_roots[row["root_key"]])
        audio_path = os.path.realpath(os.path.join(root, row["audio_relpath"]))
        if os.path.commonpath((root, audio_path)) != root:
            raise ValueError("audio_relpath escapes its approved data root")
        seen.add(row["sample_id"])
        self.rows.append({"sample_id": row["sample_id"], "audio_path": audio_path,
                          "sample_index": row.get("sample_index", 0)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform = torch.from_numpy(load_audio(row["audio_path"], length=self.length))
        return waveform, row["sample_id"]
