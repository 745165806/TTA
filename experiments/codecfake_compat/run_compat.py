"""One-shot, label-free Codecfake 16-kHz compatibility and production cache run."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
import torch

from eptta.cache.keys import CacheIdentity
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources
from experiments.codecfake_compat.compat import TARGET_RATE, inspect_audio, materialize_float_wav
from workers.compat.author_training import load_audio


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECT = ROOT / "experiments/multidomain_mechanism/manifests/codecfake_mechanism_select.json"
BASE = ROOT / "outputs_v2/ssl_aasist"
BUNDLE = BASE / "frozen/bundle.json"
REFERENCE_INDEX = BASE / "cache-target-in_the_wild/index.json"
FORBIDDEN = {"label", "raw_label", "canonical_label", "attack_id", "correct_before",
             "correct_after", "helpful_update", "harmful_update"}
EXPECTED_RATES = {16000: 222, 24000: 209, 44100: 52, 48000: 29}
PARITY_ATOL = 1e-5


def write_json_new(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv_new(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_fixed_selection():
    document = json.loads(SELECT.read_text(encoding="utf-8"))
    rows = document.get("records")
    if (document.get("role") != "mechanism_select" or document.get("dataset_id") != "codecfake"
            or document.get("count") != 512 or not isinstance(rows, list) or len(rows) != 512
            or FORBIDDEN.intersection(document) or any(FORBIDDEN.intersection(row) for row in rows)):
        raise ValueError("fixed Codecfake label-free select contract mismatch")
    ids = [row["sample_id"] for row in rows]
    if ids != sorted(ids) or len(set(ids)) != 512:
        raise ValueError("fixed Codecfake IDs must be sorted and unique")
    audited, counts = [], Counter()
    for index, row in enumerate(rows):
        if row.get("dataset_id") != "codecfake" or row.get("selection_seed") != 2026:
            raise ValueError("selected row provenance mismatch")
        source = Path(row["root_ref"]) / row["audio_relpath"]
        if source.resolve() != source or not source.is_file():
            raise ValueError("selected source path is missing or noncanonical")
        info = inspect_audio(source)
        counts[info.samplerate] += 1
        audited.append({"sample_id": row["sample_id"], "sample_index": index,
                        "source_path": str(source), "original_sample_rate": info.samplerate,
                        "original_channels": info.channels, "original_frames": info.frames,
                        "target_sample_rate": TARGET_RATE,
                        "resampled": info.samplerate != TARGET_RATE,
                        "resampler": "none" if info.samplerate == TARGET_RATE else
                        "scipy.signal.resample_poly",
                        "resampler_version": "none" if info.samplerate == TARGET_RATE else scipy.__version__})
    if dict(counts) != EXPECTED_RATES:
        raise ValueError("fixed sample-rate composition differs from prior audit")
    return rows, audited, counts


def check_cache(cache_ref, expected, identity, embedding_dim):
    cache = FeatureCache(cache_ref)
    index = cache.index
    if (index.get("identity") != identity or index.get("format") != "sharded_npy_v2" or
            index.get("sample_count") != len(expected) or index.get("feature_dim") != embedding_dim or
            index.get("num_views") != 3 or index.get("allow_pickle") is not False):
        raise ValueError("production cache identity/schema mismatch")
    actual = cache.verify_expected_ids(expected)
    if tuple(actual) != tuple(expected):
        raise ValueError("production cache order differs from fixed selection order")
    return cache


def extract_stage(out, stage, selected, rows, derived, bundle, reference):
    """Launch the unchanged production SSL-AASIST extractor with the tta Python."""
    manifest = out / "manifests" / (stage + ".jsonl")
    source_root = rows[0]["root_ref"]
    record_by_id = {row["sample_id"]: row for row in rows}
    with manifest.open("x", encoding="utf-8") as stream:
        for item in selected:
            row = record_by_id[item["sample_id"]]
            sample_id = item["sample_id"]
            root_key = "derived" if sample_id in derived else "source"
            relative = derived[sample_id] if sample_id in derived else row["audio_relpath"]
            worker_row = {"schema_version": "0.3.0", "sample_id": sample_id,
                          "sample_index": item["sample_index"], "root_key": root_key,
                          "audio_relpath": relative, "split_role": "select"}
            stream.write(json.dumps(worker_row, allow_nan=False) + "\n")
    identity = CacheIdentity(
        cache_id="codecfake-compat-%s-%s" % (out.name, stage),
        source_run_id=bundle["source_run_id"], checkpoint_ref=bundle["checkpoint_ref"],
        dataset_id="codecfake", split_role="select", manifest_ref=str(manifest.resolve()),
        preprocess=bundle["preprocess"], views=reference["views"],
        seed=reference["seed"], dtype=reference["dtype"],
        numerical_mode=reference["numerical_mode"]).as_dict()
    cache_ref = out / "feature_cache" / stage
    job = {"schema_version": "0.3.0", "job_type": "inference", "purpose": "select",
           "input_role": "select", "bundle_ref": str(BUNDLE.resolve()),
           "manifest_ref": str(manifest.resolve()),
           "data_roots": {"source": source_root,
                          "derived": str((out / "derived_audio").resolve())},
           "probe": reference["views"], "numerical_mode": reference["numerical_mode"],
           "worker_slot": 0, "worker_count": 1,
           "expected_ids": [item["sample_id"] for item in selected],
           "cache_identity": identity, "output_dir": str(cache_ref.resolve())}
    job_ref = out / "diagnostics" / (stage + ".job.json")
    write_json_new(job_ref, job)
    command = [sys.executable, str(ROOT / "workers/baseline_bridge.py"),
               "extract", "--job", str(job_ref)]
    with (out / "logs" / (stage + ".log")).open("x", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError("production extractor %s failed with exit %d" % (stage, completed.returncode))
    check_cache(cache_ref, job["expected_ids"], identity, bundle["embedding_dim"])
    print("extracted", stage, len(selected), flush=True)
    return cache_ref


def parity_rows(reference_ref, compatible_ref, selected, resources):
    reference = FeatureCache(reference_ref).load_by_id()
    compatible = FeatureCache(compatible_ref).load_by_id()
    ids = [item["sample_id"] for item in selected]
    if set(reference) != set(ids) or set(compatible) != set(ids):
        raise ValueError("native parity cache ID coverage mismatch")
    comparison = []
    weight = resources.w.detach().cpu().numpy()
    bias = float(resources.b)
    for item in selected:
        sample_id = item["sample_id"]
        path = item["source_path"]
        if item["original_sample_rate"] != TARGET_RATE:
            raise ValueError("non-native item entered native parity")
        old_waveform = load_audio(path)
        # The compatibility contract requires a direct production-path bypass.
        new_waveform = load_audio(path)
        old_features, new_features = reference[sample_id], compatible[sample_id]
        old_score = float(old_features[0] @ weight + bias)
        new_score = float(new_features[0] @ weight + bias)
        row = {"sample_id": sample_id,
               "max_waveform_difference": float(np.max(np.abs(old_waveform - new_waveform))),
               "max_feature_difference": float(np.max(np.abs(old_features - new_features))),
               "max_score_difference": abs(old_score - new_score)}
        if any(not math.isfinite(value) or value > PARITY_ATOL
               for value in list(row.values())[1:]):
            raise ValueError("native-16-kHz parity failed for %s" % sample_id)
        comparison.append(row)
    return comparison


def summarize_parity(rows, stage):
    return {"status": "PASS", "stage": stage, "count": len(rows), "atol": PARITY_ATOL,
            "max_waveform_difference": max(row["max_waveform_difference"] for row in rows),
            "max_feature_difference": max(row["max_feature_difference"] for row in rows),
            "max_score_difference": max(row["max_score_difference"] for row in rows)}


def run(run_id):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment required")
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for name in ("manifests", "logs", "diagnostics", "analysis", "feature_cache", "derived_audio"):
        (out / name).mkdir()
    start = time.perf_counter()
    try:
        rows, audited, counts = audit_fixed_selection()
        bundle, *_ = verify_frozen_export(BUNDLE)
        resources, _, _ = load_frozen_resources(BASE / "resources", bundle)
        reference = json.loads(REFERENCE_INDEX.read_text(encoding="utf-8"))["identity"]
        if (reference["source_run_id"] != bundle["source_run_id"] or
                reference["checkpoint_ref"] != bundle["checkpoint_ref"] or
                reference["preprocess"] != bundle["preprocess"] or
                reference["views"].get("num_views") != 3 or bundle["embedding_dim"] != 160):
            raise ValueError("Frozen bundle/reference feature provenance mismatch")
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                         text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        write_json_new(out / "run_config.json", {
            "run_id": run_id, "branch": branch, "commit": commit, "python": sys.version,
            "python_executable": sys.executable, "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "gpu_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": 2026, "dataset": "codecfake", "sample_count": 512,
            "sample_rate_counts": {str(k): v for k, v in sorted(counts.items())},
            "method": "explicit deterministic 16-kHz compatibility then unchanged production extraction",
            "parameters": {"target_rate": TARGET_RATE, "resampler": "scipy.signal.resample_poly",
                           "resampler_version": scipy.__version__, "window": ["kaiser", 5.0],
                           "padtype": "constant", "dtype": "float32", "parity_atol": PARITY_ATOL,
                           "views": reference["views"], "numerical_mode": reference["numerical_mode"]},
            "command": sys.argv, "target_labels_read": False})
        write_json_new(out / "provenance.json", {
            "source_bundle": str(BUNDLE), "baseline_id": bundle["baseline_id"],
            "source_run_id": bundle["source_run_id"], "checkpoint_ref": bundle["checkpoint_ref"],
            "fixed_select_manifest": str(SELECT), "source_resources": str(BASE / "resources"),
            "production_worker": str(ROOT / "workers/baseline_bridge.py"),
            "resampler_version": scipy.__version__, "soundfile_version": sf.__version__,
            "target_labels_read": False})
        fields = list(audited[0])
        write_csv_new(out / "sample_rate_audit.csv", audited, fields)
        native = [item for item in audited if item["original_sample_rate"] == TARGET_RATE]
        first32 = native[:32]
        empty = {}
        ref32 = extract_stage(out, "production_native32", first32, rows, empty, bundle, reference)
        compat32 = extract_stage(out, "compat_native32", first32, rows, empty, bundle, reference)
        compared32 = parity_rows(ref32, compat32, first32, resources)
        write_json_new(out / "native16k_parity_32.json", summarize_parity(compared32, "native32"))
        print("native32 parity PASS", flush=True)
        ref222 = extract_stage(out, "production_native222", native, rows, empty, bundle, reference)
        compat222 = extract_stage(out, "compat_native222", native, rows, empty, bundle, reference)
        compared222 = parity_rows(ref222, compat222, native, resources)
        write_csv_new(out / "native16k_parity.csv", compared222, list(compared222[0]))
        write_json_new(out / "native16k_parity.json", summarize_parity(compared222, "native222"))
        print("native222 parity PASS", flush=True)
        derived = {}
        resample_rows = []
        for item in audited:
            if item["resampled"]:
                sample_id = item["sample_id"]
                name = Path(sample_id).name
                if name in derived.values():
                    raise ValueError("derived basename collision")
                info = materialize_float_wav(item["source_path"], out / "derived_audio" / name)
                derived[sample_id] = name
                resample_rows.append({"sample_id": sample_id, **info})
        if len(derived) != 290:
            raise ValueError("non-native resampling coverage mismatch")
        write_csv_new(out / "diagnostics/resampling_details.csv", resample_rows,
                      list(resample_rows[0]))
        full = extract_stage(out, "compat_full512", audited, rows, derived, bundle, reference)
        full_features = FeatureCache(full).load_by_id()
        native_features = FeatureCache(compat222).load_by_id()
        native_final_max = max(float(np.max(np.abs(full_features[item["sample_id"]] -
                                               native_features[item["sample_id"]])))
                               for item in native)
        if native_final_max > PARITY_ATOL:
            raise ValueError("final mixed cache changed native 16-kHz features")
        validation = {"status": "PASS", "sample_count": 512,
                      "unique_ids": len(full_features), "exact_selected_id_coverage": True,
                      "feature_shape_per_sample": [3, bundle["embedding_dim"]],
                      "dtype": "float32", "all_finite": True,
                      "native16k_count": len(native), "resampled_count": len(derived),
                      "max_native_feature_difference_vs_parity_cache": native_final_max,
                      "cache_ref": str(full.resolve()), "elapsed_seconds": time.perf_counter() - start,
                      "target_labels_read": False}
        write_json_new(out / "cache_validation.json", validation)
        print("full512 cache PASS", out, flush=True)
    except BaseException:
        write_json_new(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc(),
                                                "target_labels_read": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run(args.run_id or "compat_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
