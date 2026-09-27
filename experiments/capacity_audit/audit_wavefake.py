"""Read local WaveFake Parquet metadata without extracting or modifying audio."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import wave

import pyarrow.parquet as pq


def audit(root: Path) -> dict:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(root)
    schemas = Counter()
    labels = Counter()
    path_suffixes = Counter()
    paths_by_label = defaultdict(Counter)
    rates_by_label = defaultdict(Counter)
    codecs_by_label = defaultdict(Counter)
    durations_by_label = defaultdict(list)
    channels_by_label = defaultdict(Counter)
    by_id = defaultdict(Counter)
    id_parts = defaultdict(set)
    invalid_wav = Counter()
    partition_rows = {}
    for index, file in enumerate(files):
        parquet = pq.ParquetFile(file)
        schemas[str(parquet.schema_arrow.remove_metadata())] += 1
        partition_rows[file.name] = parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=32, columns=["audio_id", "real_or_fake", "audio"]):
            for row in batch.to_pylist():
                sample_id = row["audio_id"]
                label = row["real_or_fake"]
                audio = row["audio"]
                if not isinstance(sample_id, str) or not isinstance(label, str) or not isinstance(audio, dict):
                    raise ValueError(f"Invalid row in {file}")
                labels[label] += 1
                by_id[sample_id][label] += 1
                id_parts[sample_id].add(file.name)
                path = audio.get("path") or ""
                path_suffixes[Path(path).suffix.lower()] += 1
                paths_by_label[label][Path(path).suffix.lower()] += 1
                payload = audio.get("bytes")
                if not isinstance(payload, bytes):
                    invalid_wav["missing_bytes"] += 1
                    continue
                try:
                    with wave.open(BytesIO(payload), "rb") as wav:
                        rate = wav.getframerate()
                        channels = wav.getnchannels()
                        frames = wav.getnframes()
                        codec = f"{wav.getcomptype()}_{wav.getsampwidth() * 8}bit"
                except (wave.Error, EOFError) as exc:
                    invalid_wav[type(exc).__name__] += 1
                    continue
                rates_by_label[label][str(rate)] += 1
                channels_by_label[label][str(channels)] += 1
                codecs_by_label[label][codec] += 1
                durations_by_label[label].append(frames / rate)
        if (index + 1) % 10 == 0:
            print(f"audited {index + 1}/{len(files)} partitions", flush=True)
    expected = set(labels)
    complete = sum(set(counts) == expected and all(n == 1 for n in counts.values()) for counts in by_id.values())
    incomplete_examples = [name for name, counts in by_id.items() if set(counts) != expected or any(n != 1 for n in counts.values())][:20]
    duration_summary = {}
    for label, values in durations_by_label.items():
        values.sort()
        duration_summary[label] = {
            "count": len(values), "min": values[0], "p05": values[int(.05 * (len(values)-1))],
            "median": values[len(values)//2], "p95": values[int(.95 * (len(values)-1))],
            "max": values[-1],
        }
    return {
        "root": str(root), "partition_count": len(files), "bytes": sum(f.stat().st_size for f in files),
        "partition_rows": partition_rows, "schema_variants": dict(schemas), "row_count": sum(labels.values()),
        "label_values": dict(labels), "unique_audio_ids": len(by_id), "complete_eight_way_ids": complete,
        "incomplete_id_examples": incomplete_examples,
        "ids_spanning_partitions": sum(len(parts) > 1 for parts in id_parts.values()),
        "path_suffixes": dict(path_suffixes), "path_suffixes_by_label": {k:dict(v) for k,v in paths_by_label.items()},
        "rates_by_label": {k:dict(v) for k,v in rates_by_label.items()},
        "channels_by_label": {k:dict(v) for k,v in channels_by_label.items()},
        "codecs_by_label": {k:dict(v) for k,v in codecs_by_label.items()},
        "duration_seconds_by_label": duration_summary, "invalid_wav": dict(invalid_wav),
        "speaker_field": "ABSENT", "language_field": "ABSENT", "generator_field": "real_or_fake has WF1..WF7/R codes; no independent generator metadata",
        "pairing_key": "audio_id", "audio_storage": "embedded bytes in audio struct; audio.path is a basename only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/media/dell/data/fakedata/WaveFake/data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("row_count", "unique_audio_ids", "complete_eight_way_ids", "label_values", "rates_by_label", "invalid_wav")}, indent=2))


if __name__ == "__main__":
    main()
