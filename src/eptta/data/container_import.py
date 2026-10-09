"""Materialize explicitly described embedded audio into a prepare-data input.

This is an opt-in bridge for local Parquet/Arrow releases. It never reads target
labels while scoring; labels are written only to the separate prepare-data input.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

from eptta.data.io import AtomicDirectory
from eptta.errors import ContractError, DataError, ResourceError


def _require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be explicit nonempty text")
    return value.strip()


def _source_files(config):
    root = Path(_require_text(config.get("source_root"), "source_root")).resolve()
    if not root.is_dir():
        raise DataError(f"source_root is missing: {root}")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ContractError("sources must be a nonempty list of explicit glob/count pairs")
    result = []
    for source in sources:
        if set(source) != {"pattern", "expected_count"}:
            raise ContractError("each source requires pattern and expected_count")
        pattern = _require_text(source["pattern"], "source.pattern")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ContractError("source.pattern must stay inside source_root")
        expected = source["expected_count"]
        if type(expected) is not int or expected < 1:
            raise ContractError("source.expected_count must be a positive integer")
        found = sorted(root.glob(pattern))
        if len(found) != expected or any(not p.is_file() or not p.resolve().is_relative_to(root)
                                          for p in found):
            raise DataError(f"source pattern {pattern!r} has {len(found)} files; expected {expected}")
        result.extend(found)
    if len(result) != len(set(result)):
        raise DataError("source patterns overlap")
    return root, result


def _rows(path, kind, columns):
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ResourceError("pyarrow is required to read embedded Parquet/Arrow audio in the tta environment") from exc
    if kind == "parquet":
        reader = parquet.ParquetFile(path)
        absent = set(columns) - set(reader.schema_arrow.names)
        if absent:
            raise DataError(f"missing columns in {path}: {sorted(absent)}")
        batches = reader.iter_batches(batch_size=64, columns=columns)
    elif kind == "arrow":
        source = pa.memory_map(str(path), "r")
        reader = ipc.open_stream(source)
        absent = set(columns) - set(reader.schema.names)
        if absent:
            raise DataError(f"missing columns in {path}: {sorted(absent)}")
        batches = reader
    else:
        raise ContractError("format must be parquet or arrow")
    try:
        for batch in batches:
            names = batch.schema.names
            for row in batch.to_pylist():
                yield {name: row[name] for name in columns if name in names}
    finally:
        if kind == "arrow":
            source.close()


def _group_map(path, key_column, group_column):
    if not path:
        return None
    mapping = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {key_column, group_column}.issubset(reader.fieldnames or []):
            raise DataError("group map lacks its explicit key/group columns")
        for row in reader:
            key = _require_text(row[key_column], "group map key")
            group = _require_text(row[group_column], "group map group")
            if key in mapping:
                raise DataError(f"duplicate group map key: {key}")
            mapping[key] = group
    return mapping


def _audio_bytes(audio, preprocess, location):
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes) or not audio["bytes"]:
        raise DataError(f"audio.bytes missing at {location}")
    import soundfile as sf
    import numpy as np
    raw = audio["bytes"]
    try:
        info = sf.info(io.BytesIO(raw))
    except Exception as exc:
        raise DataError(f"embedded audio cannot be decoded at {location}") from exc
    expected = preprocess["input_sample_rate"]
    if info.samplerate != expected or info.frames < 1 or info.format not in {"WAV", "FLAC", "OGG"}:
        raise DataError(f"unexpected embedded audio at {location}: {info}")
    if preprocess["resampler"] == "identity":
        if expected != 16000:
            raise ContractError("identity import requires 16 kHz input")
        return raw, {"WAV": ".wav", "FLAC": ".flac", "OGG": ".ogg"}[info.format]
    if preprocess["resampler"] != "scipy_polyphase_v1" or preprocess["output_sample_rate"] != 16000:
        raise ContractError("unsupported explicit resampling recipe")
    from scipy.signal import resample_poly
    samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if rate != expected or not np.isfinite(samples).all():
        raise DataError(f"invalid decoded audio at {location}")
    if preprocess["channels"] != "mono_mean":
        raise ContractError("resampling requires channels=mono_mean")
    mono = samples.mean(axis=1)
    divisor = math.gcd(expected, 16000)
    converted = resample_poly(mono, 16000 // divisor, expected // divisor).astype("float32")
    if converted.size == 0 or not np.isfinite(converted).all():
        raise DataError(f"invalid resampled audio at {location}")
    output = io.BytesIO()
    sf.write(output, converted, 16000, format="WAV", subtype="FLOAT")
    return output.getvalue(), ".wav"


def materialize(config):
    required = {"schema_version", "dataset_id", "release", "subset", "format", "source_root",
                "sources", "audio_column", "label_column", "group", "label_map", "preprocess", "output"}
    optional = {"select_groups", "assert_columns", "expected_sample_count"}
    if not required.issubset(config) or set(config) - required - optional or config["schema_version"] != "0.1.0":
        raise ContractError(f"container import requires exactly {sorted(required)}")
    dataset_id = _require_text(config["dataset_id"], "dataset_id")
    release = _require_text(config["release"], "release")
    subset = _require_text(config["subset"], "subset")
    kind = config["format"]
    audio_key = _require_text(config["audio_column"], "audio_column")
    label_key = _require_text(config["label_column"], "label_column")
    label_map = config["label_map"]
    if not isinstance(label_map, dict) or not label_map or any(type(v) is not int or v not in (0, 1)
                                                                for v in label_map.values()):
        raise ContractError("label_map must explicitly map source values to 0/1")
    preprocess = config["preprocess"]
    if (not isinstance(preprocess, dict) or set(preprocess) != {"input_sample_rate", "output_sample_rate", "channels", "resampler"}
            or type(preprocess["input_sample_rate"]) is not int or preprocess["input_sample_rate"] < 1
            or preprocess["output_sample_rate"] != 16000 or preprocess["channels"] != "mono_mean"
            or preprocess["resampler"] not in {"identity", "scipy_polyphase_v1"}):
        raise ContractError("preprocess requires explicit input/output rate, mono_mean, and supported resampler")
    selected = config.get("select_groups")
    if selected is not None and (not isinstance(selected, list) or not selected or
                                 any(not isinstance(x, str) or not x for x in selected) or len(selected) != len(set(selected))):
        raise ContractError("select_groups must contain unique explicit group IDs")
    selected = set(selected) if selected is not None else None
    asserted = config.get("assert_columns", {})
    if not isinstance(asserted, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in asserted.items()):
        raise ContractError("assert_columns must map explicit columns to exact string values")
    expected_count = config.get("expected_sample_count")
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ContractError("expected_sample_count must be a positive integer")
    group = config["group"]
    if not isinstance(group, dict) or set(group) not in ({"column"}, {"map_ref", "key_column", "map_key_column", "map_group_column"}):
        raise ContractError("group requires a source column or explicit external key/group map")
    group_column = _require_text(group["column"], "group.column") if "column" in group else None
    mapping = None if group_column else _group_map(group["map_ref"], group["map_key_column"], group["map_group_column"])
    if mapping is None and not group_column:
        raise ContractError("group.map_ref must be supplied")
    source_root, sources = _source_files(config)
    destination = Path(_require_text(config["output"], "output"))
    columns = list(dict.fromkeys([audio_key, label_key, group_column or group["key_column"], *asserted]))
    count = 0
    observed_groups = set()
    with AtomicDirectory(destination) as temporary:
        audio_dir = temporary / "audio"
        audio_dir.mkdir()
        with (temporary / "input.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "audio_relpath", "label", "source_group_id", "split_role"])
            writer.writeheader()
            for source in sources:
                relative_source = source.relative_to(source_root).as_posix()
                for index, row in enumerate(_rows(source, kind, columns)):
                    if any(str(row[key]) != value for key, value in asserted.items()):
                        raise DataError(f"asserted source field mismatch in {source}:{index}")
                    raw_label = row[label_key]
                    if raw_label not in label_map:
                        raise DataError(f"unmapped label {raw_label!r} in {source}:{index}")
                    raw_group = row[group_column] if group_column else mapping.get(str(row[group["key_column"]]))
                    group_id = _require_text(raw_group, f"group at {source}:{index}")
                    if selected is not None and group_id not in selected:
                        continue
                    observed_groups.add(group_id)
                    output_bytes, suffix = _audio_bytes(row[audio_key], preprocess, f"{source}:{index}")
                    sample_id = f"{dataset_id}:{relative_source}:{index}"
                    audio_name = f"{count:09d}{suffix}"
                    with (audio_dir / audio_name).open("xb") as output:
                        output.write(output_bytes)
                    writer.writerow({"sample_id": sample_id, "audio_relpath": "audio/" + audio_name,
                                     "label": raw_label, "source_group_id": group_id,
                                     "split_role": "target_test"})
                    count += 1
        if not count:
            raise DataError("container import produced no rows")
        if selected is not None and observed_groups != selected:
            raise DataError(f"selected groups not covered: {sorted(selected - observed_groups)}")
        if expected_count is not None and count != expected_count:
            raise DataError(f"imported {count} rows; expected {expected_count}")
        (temporary / "summary.json").write_text(json.dumps({"schema_version": "0.1.0", "status": "READY",
            "dataset_id": dataset_id, "release": release, "subset": subset,
            "sample_count": count, "source_files": len(sources), "group_count": len(observed_groups),
            "preprocess": preprocess}, indent=2) + "\n", encoding="utf-8")
    return {"status": "READY", "sample_count": count, "input_manifest": str(destination / "input.csv"),
            "audio_root": str(destination)}


def main():
    parser = argparse.ArgumentParser(description="Import explicit embedded-audio data into a new directory")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(materialize(config), indent=2))


if __name__ == "__main__":
    main()
