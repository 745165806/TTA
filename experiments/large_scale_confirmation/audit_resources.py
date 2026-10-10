"""Label-free audio header and WaveFake reader audit for fixed assignments."""
import importlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from experiments.large_scale_confirmation.make_assignments import HERE, write_once


def run():
    rates = {}
    for domain in ("codecfake", "asv2019_la_dev"):
        manifest = json.loads((HERE / "manifests" / (domain + "_confirmation_select.json"))
                              .read_text(encoding="utf-8"))
        counts = Counter()
        for record in manifest["records"]:
            root = Path(record["root_ref"]).resolve()
            path = (root / record["audio_relpath"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("fixed selected audio missing/escaped: " + record["sample_id"])
            info = sf.info(path)
            if info.frames < 1 or info.channels < 1:
                raise ValueError("empty selected audio: " + record["sample_id"])
            counts[info.samplerate] += 1
        rates[domain] = dict(sorted(counts.items()))
    modules = {}
    for name in ("pyarrow", "pandas", "polars", "duckdb"):
        modules[name] = (getattr(importlib.import_module(name), "__version__", "unknown")
                         if importlib.util.find_spec(name) else "MISSING")
    wavefake = Path("/media/dell/data/fakedata/WaveFake/data")
    parquet_files = sorted(wavefake.glob("*.parquet")) if wavefake.is_dir() else []
    report = {"role": "label_free_resource_audit", "selected_audio_sample_rate_counts": rates,
              "target_labels_read": False,
              "wavefake": {"root": str(wavefake), "parquet_file_count": len(parquet_files),
                           "readers": modules,
                           "status": "READER_AVAILABLE_PIPELINE_NOT_MATERIALIZED"
                           if modules["pyarrow"] != "MISSING" else "WAVEFAKE_BLOCKED_BY_READER"}}
    write_once(HERE / "resource_audit.json", report)
    print(report)


if __name__ == "__main__":
    run()
