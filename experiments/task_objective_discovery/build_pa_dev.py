"""Fix one auxiliary ASVspoof2019 PA official-dev assignment by speaker group."""
import json
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = Path("/media/dell/data/fakedata/asvspoof2019/PA/ASVspoof2019_PA_cm_protocols/"
                "ASVspoof2019.PA.cm.dev.trl.txt")
AUDIO_ROOT = Path("/media/dell/data/fakedata/asvspoof2019/PA/ASVspoof2019_PA_dev")
OUT = HERE / "manifests"
SEED = 2026


def write_new(path, doc):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build():
    if OUT.exists():
        raise FileExistsError("auxiliary development assignment already exists; redraw forbidden")
    groups = defaultdict(list)
    seen = set()
    with PROTOCOL.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("unexpected official PA dev protocol row")
            speaker, sample_id = fields[:2]  # Never inspect the label column to select.
            if sample_id in seen:
                raise ValueError("duplicate official PA dev ID")
            seen.add(sample_id)
            groups[speaker].append(sample_id)
    short_groups = []
    for group in sorted(groups):
        ids = groups[group]
        if len(ids) != 270:
            continue
        paths = [AUDIO_ROOT / ("flac/%s.flac" % sample_id) for sample_id in ids]
        if all(path.is_file() for path in paths):
            short_groups.append(group)
    if not short_groups:
        raise ValueError("no audio-complete 270-sample speaker group")
    random.Random(SEED).shuffle(short_groups)
    selected_groups = sorted(short_groups[:1])
    selected = sorted((sample_id, group) for group in selected_groups
                      for sample_id in groups[group])
    if len(selected) != 270:
        raise ValueError("full selected speaker group must contain 270 samples")
    records, sample_rates = [], {}
    for sample_id, group in selected:
        relative = "flac/%s.flac" % sample_id
        info = sf.info(AUDIO_ROOT / relative)
        sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
        if info.samplerate != 16000 or info.frames <= 0:
            raise ValueError("official PA dev waveform violates production sample rate")
        records.append({"sample_id": sample_id, "dataset_id": "asv2019_pa_dev",
                        "original_split": "official_dev", "source_group_id": group,
                        "root_key": "asv2019_pa_dev", "root_ref": str(AUDIO_ROOT),
                        "audio_relpath": relative, "selection_seed": SEED,
                        "selection_policy": "one_audio_complete_270_speaker_group_random2026"})
    OUT.mkdir(parents=True, exist_ok=False)
    shared = {"schema_version": "0.1.0", "dataset_id": "asv2019_pa_dev",
              "selection_seed": SEED, "selection_policy": "one_audio_complete_270_speaker_group_random2026",
              "source_ref": str(PROTOCOL), "count": len(records),
              "selected_groups": selected_groups, "sample_rates": sample_rates,
              "records": records}
    write_new(OUT / "asv2019_pa_dev_mechanism_select.json",
              {**shared, "role": "mechanism_select"})
    write_new(OUT / "asv2019_pa_dev_mechanism_audit.json",
              {**shared, "role": "mechanism_audit", "label_source_ref": str(PROTOCOL),
               "label_status": "DEFERRED_UNTIL_SCORES_COMPLETE"})
    print({"groups": selected_groups, "count": len(records),
           "sample_rates": sample_rates})


if __name__ == "__main__":
    build()
