"""Fix a paired WaveFake development assignment from audited content IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
COUNT = 2048
SEED = 2026
CODES = tuple(f"WF{i}" for i in range(1, 8))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main(root: Path):
    audit = json.loads((HERE / "wavefake_audit.json").read_text())
    if (audit["root"] != str(root) or audit["unique_audio_ids"] != 13100 or
            audit["complete_eight_way_ids"] != 13100 or audit["invalid_wav"] or
            set(audit["label_values"]) != {"R", *CODES}):
        raise ValueError("WaveFake full audit does not license paired assignment")
    locations = {}
    files = sorted(root.glob("*.parquet"))
    for file in files:
        table = pq.read_table(file, columns=["audio_id", "real_or_fake"])
        for index, (audio_id, code) in enumerate(zip(table["audio_id"].to_pylist(),
                                                      table["real_or_fake"].to_pylist())):
            key = (audio_id, code)
            if key in locations:
                raise ValueError("duplicate content/generator row")
            locations[key] = (str(file), index)
    group_ids = sorted({audio_id for audio_id, _ in locations})
    if len(group_ids) != 13100:
        raise ValueError("audit and metadata group counts disagree")
    selected = random.Random(SEED).sample(group_ids, COUNT)
    records, label_records = [], []
    for group_index, audio_id in enumerate(selected):
        fake = CODES[group_index % len(CODES)]
        for code in ("R", fake):
            file, row_index = locations[(audio_id, code)]
            sample_id = f"wavefake:{audio_id}:{code}"
            records.append({"sample_id": sample_id, "audio_id": audio_id,
                            "parquet_ref": file, "parquet_row_index": row_index,
                            "sample_index": len(records), "original_split": "local_wavefake_resource",
                            "selection_seed": SEED, "selection_policy": "paired_2048_seeded_groups_generator_cycle"})
            label_records.append({"sample_id": sample_id, "label": 0 if code == "R" else 1})
    if len(records) != 4096 or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("paired assignment coverage invalid")
    out = HERE / "manifests"
    write_new(out / "wavefake_capacity_select.json", {"role": "capacity_development_select",
        "dataset_id": "wavefake", "count": len(records), "content_groups": COUNT,
        "seed": SEED, "selection_policy": "random.Random(2026).sample(sorted_audio_ids,2048); R plus WF1..WF7 cycle",
        "records": records})
    write_new(out / "wavefake_capacity_labels.json", {"role": "supervised_development_labels",
        "dataset_id": "wavefake", "count": len(label_records), "records": label_records})
    print(f"fixed {COUNT} related-content groups / {len(records)} waveforms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/media/dell/data/fakedata/WaveFake/data"))
    main(parser.parse_args().root)
