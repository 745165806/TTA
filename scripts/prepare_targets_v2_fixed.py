#!/usr/bin/env python3
"""
EP-TTA target dataset preparation helper, fixed version.

Put at:
  /media/dell/data/fakeAudioDection/TTA/scripts/prepare_targets_v2.py

Run:
  cd /media/dell/data/fakeAudioDection/TTA
  conda activate tta
  python scripts/prepare_targets_v2.py 2>&1 | tee logs/prepare_targets_v2.log

This version:
  - keeps already READY ITW / ASV2021 LA / ASV2021 DF untouched;
  - fixes old Codecfake label-pack compatibility:
      sample_id       <- sample_id or utt_id
      source_group_id <- source_group_id or speaker/attack/source/file
  - searches WaveFake resources more broadly;
  - if generated_audio.zip already exists locally, it extracts it automatically;
  - never downloads data from the network.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

EXPECTED = {
    "in_the_wild": 31_779,
    "asv2021_la_eval": 148_176,
    "asv2021_df_eval": 533_928,
}

DATASETS = (
    "in_the_wild",
    "asv2021_la_eval",
    "asv2021_df_eval",
    "codecfake_xie",
    "wavefake",
)

SCRIPT = Path(__file__).resolve()
DEFAULT_TTA = SCRIPT.parents[1] if SCRIPT.parent.name == "scripts" else Path(
    "/media/dell/data/fakeAudioDection/TTA"
)
TTA_ROOT = Path(os.environ.get("TTA_ROOT", DEFAULT_TTA)).expanduser().resolve()
FAKEDATA_ROOT = Path(
    os.environ.get("FAKEDATA_ROOT", "/media/dell/data/fakedata")
).expanduser().resolve()

SOURCE_DIR = TTA_ROOT / "data/source_manifests_v2"
TARGET_DIR = TTA_ROOT / "data/manifests_v2"
CONFIG_DIR = TTA_ROOT / "configs/local_v2/targets"

SOURCE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS: dict[str, str] = {}


def info(msg: str = "") -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def mark_incomplete(name: str, msg: str) -> None:
    warn(f"{name}: {msg}")
    RESULTS[name] = f"INCOMPLETE: {msg}"


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def target_files(name: str) -> tuple[Path, Path]:
    base = TARGET_DIR / name
    return base / "inference/target_test.jsonl", base / "labels/target_test.jsonl"


def is_ready(name: str, expected: int | None = None) -> bool:
    inf, lab = target_files(name)
    if not inf.is_file() or not lab.is_file():
        return False
    ni = count_lines(inf)
    nl = count_lines(lab)
    if ni <= 0 or ni != nl:
        warn(f"{name}: existing outputs mismatch inference={ni} labels={nl}")
        return False
    if expected is not None and ni != expected:
        warn(f"{name}: existing count={ni}, expected={expected}")
        return False
    info(f"[READY] {name}: {ni:,}")
    RESULTS[name] = f"READY ({ni:,})"
    return True


def backup_output(name: str) -> None:
    dst = TARGET_DIR / name
    if not dst.exists():
        return
    root = TARGET_DIR / "_backup_before_prepare_targets"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = root / f"{name}-{stamp}"
    k = 1
    while backup.exists():
        backup = root / f"{name}-{stamp}-{k}"
        k += 1
    shutil.move(str(dst), str(backup))
    info(f"[BACKUP] {dst.relative_to(TTA_ROOT)} -> {backup.relative_to(TTA_ROOT)}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_config(
    name: str,
    dataset_id: str,
    release: str,
    subset: str,
    source_manifest: Path,
    audio_root: Path,
    label_map: dict[str, int],
) -> Path:
    cfg = CONFIG_DIR / f"{name}_prepare.yaml"
    payload = {
        "schema_version": "0.1.0",
        "command": "prepare-data",
        "dataset": {
            "dataset_id": dataset_id,
            "release": release,
            "subset": subset,
            "manifest": str(source_manifest.relative_to(TTA_ROOT)),
            "audio_root": str(audio_root),
            "root_key": dataset_id,
        },
        "columns": {
            "sample_id": "sample_id",
            "audio_relpath": "audio_relpath",
            "label": "label",
            "group_id": "source_group_id",
            "split_role": "split_role",
        },
        "label_map": label_map,
        "split": {
            "reuse_existing": True,
            "seed": 13,
        },
        "output": str((TARGET_DIR / name).relative_to(TTA_ROOT)),
    }
    cfg.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg


def run_prepare(name: str, cfg: Path, expected: int | None = None) -> bool:
    backup_output(name)
    cmd = [
        sys.executable,
        "-m",
        "eptta.cli",
        "prepare-data",
        "--config",
        str(cfg.relative_to(TTA_ROOT)),
    ]
    info("[RUN] " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=TTA_ROOT).returncode
    if rc != 0:
        mark_incomplete(name, f"prepare-data exited with code {rc}")
        return False
    if not is_ready(name, expected):
        mark_incomplete(name, "prepare-data finished but target_test validation failed")
        return False
    return True


def first_nonempty(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def check_existing_known_targets() -> None:
    for name, expected in EXPECTED.items():
        if not is_ready(name, expected):
            mark_incomplete(
                name,
                "was previously prepared but current target manifest is missing or count differs",
            )


def codecfake_group(row: dict[str, str], src: Path) -> tuple[str, str]:
    group = first_nonempty(row, "source_group_id")
    if group:
        return group, "source_group_id"

    speaker = first_nonempty(row, "speaker", "speaker_id")
    if speaker and speaker.lower() not in {"-", "none", "null", "unknown"}:
        return f"codecfake:speaker:{speaker}", "speaker"

    attack = first_nonempty(row, "attack", "generator_id", "codec_id")
    source = first_nonempty(row, "source", "dataset_source")

    if attack and source:
        return f"codecfake:source:{source}:attack:{attack}", "source+attack"
    if attack:
        return f"codecfake:attack:{attack}", "attack"
    if source:
        return f"codecfake:source:{source}", "source"

    return f"codecfake:manifest:{src.stem}", "manifest"


def resolve_audio_path(row: dict[str, str]) -> Path | None:
    raw = first_nonempty(row, "path")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = TTA_ROOT / p
        return p.resolve()

    rel = first_nonempty(row, "audio_relpath")
    if not rel:
        return None
    return (FAKEDATA_ROOT / rel).resolve()


def prepare_codecfake() -> None:
    name = "codecfake_xie"
    if is_ready(name):
        return

    srcdir = TTA_ROOT / "fakedata/current/manifests"
    files = sorted(srcdir.glob("codecfake_xie_*.csv"))
    if not files:
        mark_incomplete(name, f"no codecfake_xie_*.csv under {srcdir}")
        return

    records: list[dict] = []
    seen: set[str] = set()
    abs_paths: list[str] = []
    id_sources = Counter()
    group_sources = Counter()
    per_file = Counter()

    for src in files:
        low = src.stem.lower()
        if "train" in low or "dev" in low:
            info(f"[SKIP] source split manifest: {src.name}")
            continue

        with src.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if "label" not in fields and "canonical_label" not in fields:
                mark_incomplete(name, f"{src}: missing label/canonical_label")
                return
            if "path" not in fields and "audio_relpath" not in fields:
                mark_incomplete(name, f"{src}: missing path/audio_relpath")
                return

            for rownum, row in enumerate(reader, 2):
                split = first_nonempty(row, "split", "official_split").lower()
                if split in {"train", "training", "dev", "development"}:
                    continue

                sid = first_nonempty(row, "sample_id")
                if sid:
                    id_sources["sample_id"] += 1
                else:
                    sid = first_nonempty(row, "utt_id", "trial_id", "id")
                    if sid:
                        id_sources["utt_id"] += 1

                if not sid:
                    mark_incomplete(name, f"{src}:{rownum}: no sample_id or utt_id")
                    return

                if sid in seen:
                    prefixed = f"{src.stem}:{sid}"
                    if prefixed in seen:
                        mark_incomplete(name, f"duplicate sample ID even after prefix: {sid}")
                        return
                    sid = prefixed
                    id_sources["manifest_prefix"] += 1

                label = first_nonempty(row, "label", "canonical_label")
                if label not in {"0", "1", "bonafide", "spoof"}:
                    mark_incomplete(name, f"{src}:{rownum}: unexpected label {label!r}")
                    return

                full = resolve_audio_path(row)
                if full is None:
                    mark_incomplete(name, f"{src}:{rownum}: no usable audio path")
                    return
                if not full.is_file():
                    mark_incomplete(name, f"{src}:{rownum}: audio not found: {full}")
                    return

                group, kind = codecfake_group(row, src)
                group_sources[kind] += 1

                seen.add(sid)
                abs_paths.append(str(full))
                records.append(
                    {
                        "sample_id": sid,
                        "fullpath": full,
                        "label": label,
                        "source_group_id": group,
                        "split_role": "target_test",
                    }
                )
                per_file[src.name] += 1

    if not records:
        mark_incomplete(name, "no Codecfake target records remained after filtering")
        return

    common_root = Path(os.path.commonpath(abs_paths)).resolve()

    rows = [
        {
            "sample_id": r["sample_id"],
            "audio_relpath": r["fullpath"].relative_to(common_root).as_posix(),
            "label": r["label"],
            "source_group_id": r["source_group_id"],
            "split_role": "target_test",
        }
        for r in records
    ]

    out = SOURCE_DIR / "codecfake_xie_target.csv"
    write_csv(out, rows)
    (SOURCE_DIR / "codecfake_xie.root").write_text(str(common_root) + "\n", encoding="utf-8")

    info("[OK] Codecfake manifest inputs:")
    for filename, n in sorted(per_file.items()):
        info(f"     {filename}: {n:,}")
    info(f"[OK] total target rows: {len(rows):,}")
    info(f"[OK] ID sources       : {dict(id_sources)}")
    info(f"[OK] group sources    : {dict(group_sources)}")
    info(f"[OK] common audio root: {common_root}")

    cfg = write_config(
        name=name,
        dataset_id="codecfake_xie",
        release="Codecfake-Xie",
        subset="official-test-available-audio",
        source_manifest=out,
        audio_root=common_root,
        label_map={"0": 0, "1": 1, "bonafide": 0, "spoof": 1},
    )
    run_prepare(name, cfg)


def find_dirs(name: str) -> list[Path]:
    result = []
    try:
        for p in FAKEDATA_ROOT.rglob(name):
            if p.is_dir():
                result.append(p.resolve())
    except OSError:
        pass
    return result


def find_files(patterns: tuple[str, ...]) -> list[Path]:
    result = []
    seen = set()
    for pattern in patterns:
        try:
            for p in FAKEDATA_ROOT.rglob(pattern):
                if p.is_file():
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        result.append(rp)
        except OSError:
            pass
    return sorted(result)


def maybe_extract_wavefake() -> Path | None:
    manual = os.environ.get("WAVEFAKE_GEN")
    if manual:
        p = Path(manual).expanduser().resolve()
        if p.is_dir():
            return p

    dirs = find_dirs("generated_audio")
    if dirs:
        return dirs[0]

    archives = find_files(
        ("generated_audio.zip", "*generated*audio*.zip", "*WaveFake*.zip", "*wavefake*.zip")
    )
    if not archives:
        return None

    archive = next((p for p in archives if p.name.lower() == "generated_audio.zip"), archives[0])
    info(f"[FOUND] WaveFake archive candidate: {archive}")

    if not zipfile.is_zipfile(archive):
        warn(f"not a readable ZIP: {archive}")
        return None

    dest = FAKEDATA_ROOT / "wavefake"
    dest.mkdir(parents=True, exist_ok=True)
    info(f"[EXTRACT] {archive} -> {dest}")

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    dirs = find_dirs("generated_audio")
    if dirs:
        return dirs[0]

    count = sum(1 for p in dest.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"})
    if count:
        info(f"[INFO] extracted {count:,} WaveFake audio files under {dest}")
        return dest
    return None


def locate_ljspeech() -> Path | None:
    manual = os.environ.get("LJSPEECH_WAV")
    if manual:
        p = Path(manual).expanduser().resolve()
        if p.is_dir():
            return p
    try:
        for p in FAKEDATA_ROOT.rglob("wavs"):
            if p.is_dir() and p.parent.name.lower().startswith("ljspeech"):
                return p.resolve()
    except OSError:
        pass
    return None


def locate_jsut() -> Path | None:
    manual = os.environ.get("JSUT_WAV")
    if manual:
        p = Path(manual).expanduser().resolve()
        if p.is_dir():
            return p
    try:
        for p in FAKEDATA_ROOT.rglob("wav"):
            if p.is_dir() and p.parent.name.lower() == "basic5000":
                return p.resolve()
    except OSError:
        pass
    return None


def iter_audio(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".wav", ".flac"}:
            yield p.resolve()


def print_wavefake_inventory() -> None:
    info("[INFO] Local WaveFake/reference archive candidates:")
    candidates = find_files(
        (
            "*generated*audio*.zip",
            "*wavefake*.zip",
            "*WaveFake*.zip",
            "*LJSpeech*.tar*",
            "*ljspeech*.tar*",
            "*LJSpeech*.zip",
            "*jsut*.zip",
            "*JSUT*.zip",
            "*jsut*.tar*",
        )
    )
    if not candidates:
        info("       <none found under FAKEDATA_ROOT>")
        return
    for p in candidates[:50]:
        try:
            size = p.stat().st_size / (1024 ** 3)
            info(f"       {p}  ({size:.2f} GiB)")
        except OSError:
            info(f"       {p}")


def prepare_wavefake() -> None:
    name = "wavefake"
    if is_ready(name):
        return

    gen = maybe_extract_wavefake()
    lj = locate_ljspeech()
    js = locate_jsut()

    info("[INFO] WaveFake resource detection:")
    info(f"       generated_audio = {gen or '<missing>'}")
    info(f"       LJSpeech wavs    = {lj or '<missing>'}")
    info(f"       JSUT basic5000   = {js or '<missing>'}")

    missing = []
    if gen is None or not gen.is_dir():
        missing.append("WAVEFAKE_GEN=/path/to/generated_audio")
    if lj is None or not lj.is_dir():
        missing.append("LJSPEECH_WAV=/path/to/LJSpeech-1.1/wavs")
    if js is None or not js.is_dir():
        missing.append("JSUT_WAV=/path/to/JSUT/basic5000/wav")

    if missing:
        print_wavefake_inventory()
        mark_incomplete(name, "missing resources; set " + ", ".join(missing))
        return

    common_root = Path(
        os.path.commonpath([str(gen.resolve()), str(lj.resolve()), str(js.resolve())])
    ).resolve()

    rows = []
    counts = Counter()

    for p in iter_audio(gen):
        local = p.relative_to(gen)
        generator = local.parts[0] if len(local.parts) > 1 else "generated"
        rows.append(
            {
                "sample_id": "wavefake:spoof:" + local.with_suffix("").as_posix().replace("/", "::"),
                "audio_relpath": p.relative_to(common_root).as_posix(),
                "label": "spoof",
                "source_group_id": f"wavefake:generator:{generator}",
                "split_role": "target_test",
            }
        )
        counts["spoof"] += 1

    for tag, root in (("ljspeech", lj), ("jsut", js)):
        for p in iter_audio(root):
            local = p.relative_to(root)
            rows.append(
                {
                    "sample_id": f"wavefake:bonafide:{tag}:"
                    + local.with_suffix("").as_posix().replace("/", "::"),
                    "audio_relpath": p.relative_to(common_root).as_posix(),
                    "label": "bonafide",
                    "source_group_id": f"wavefake:reference:{tag}",
                    "split_role": "target_test",
                }
            )
            counts["bonafide"] += 1

    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        mark_incomplete(name, "duplicate generated sample IDs")
        return
    if counts["spoof"] == 0 or counts["bonafide"] == 0:
        mark_incomplete(
            name,
            f"both classes required; spoof={counts['spoof']} bonafide={counts['bonafide']}",
        )
        return

    out = SOURCE_DIR / "wavefake_target.csv"
    write_csv(out, rows)

    info(
        f"[OK] WaveFake source manifest: spoof={counts['spoof']:,} "
        f"bonafide={counts['bonafide']:,} total={len(rows):,}"
    )
    info(f"[OK] common audio root: {common_root}")

    cfg = write_config(
        name=name,
        dataset_id="wavefake",
        release="WaveFake-1.2.0",
        subset="constructed-LJSpeech-JSUT-reference-target-test",
        source_manifest=out,
        audio_root=common_root,
        label_map={"bonafide": 0, "spoof": 1},
    )
    run_prepare(name, cfg)


def summary() -> int:
    info()
    info("=" * 72)
    info("FINAL TARGET MANIFEST SUMMARY")
    info("=" * 72)

    incomplete = 0
    for name in DATASETS:
        inf, lab = target_files(name)
        if inf.is_file() and lab.is_file():
            ni = count_lines(inf)
            nl = count_lines(lab)
            if ni == nl and ni > 0:
                info(f"{name:22s} READY ({ni:,})")
                continue
        incomplete += 1
        info(f"{name:22s} {RESULTS.get(name, 'INCOMPLETE')}")

    info()
    if incomplete:
        info(
            f"Finished with {incomplete} incomplete dataset(s). "
            "Existing READY datasets were not modified."
        )
        return 1

    info("All target datasets are prepared.")
    return 0


def main() -> int:
    if not (TTA_ROOT / "pyproject.toml").is_file() or not (TTA_ROOT / "src/eptta").is_dir():
        print(f"[FATAL] not a TTA repository: {TTA_ROOT}", file=sys.stderr)
        return 2

    os.chdir(TTA_ROOT)

    info("=" * 72)
    info("EP-TTA target dataset preparation -- fixed")
    info(f"TTA_ROOT      = {TTA_ROOT}")
    info(f"FAKEDATA_ROOT = {FAKEDATA_ROOT}")
    info(f"Python        = {sys.executable}")
    info("=" * 72)

    info("\n-- Existing proven targets --")
    check_existing_known_targets()

    info("\n-- Codecfake Xie --")
    try:
        prepare_codecfake()
    except Exception as exc:
        mark_incomplete("codecfake_xie", f"{type(exc).__name__}: {exc}")

    info("\n-- WaveFake --")
    try:
        prepare_wavefake()
    except Exception as exc:
        mark_incomplete("wavefake", f"{type(exc).__name__}: {exc}")

    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
