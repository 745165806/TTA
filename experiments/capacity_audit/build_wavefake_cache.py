"""Materialize fixed WaveFake pairs through explicit 22.05→16-kHz production compatibility."""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import scipy
from scipy.signal import resample_poly
import soundfile as sf
import torch

from eptta.cache.keys import CacheIdentity
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "outputs_v2/ssl_aasist"
SELECT = HERE / "manifests/wavefake_capacity_select.json"


def write_new(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def materialize(rows, out):
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["parquet_ref"]].append(row)
    realized = {}
    for file, entries in by_file.items():
        table = pq.read_table(file, columns=["audio"])
        for row in entries:
            audio = table["audio"][row["parquet_row_index"]].as_py()
            samples, rate = sf.read(BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            if (rate != 22050 or samples.ndim != 1 or samples.size < 1 or
                    not np.isfinite(samples).all()):
                raise ValueError("WaveFake source audio violates audited 22.05-kHz mono contract")
            data = resample_poly(samples, up=320, down=441,
                                 window=("kaiser", 5.0), padtype="constant").astype(np.float32)
            if data.size < 1 or not np.isfinite(data).all():
                raise FloatingPointError("WaveFake resampling produced invalid waveform")
            name = row["sample_id"].replace(":", "__") + ".wav"
            destination = out / "derived_audio" / name
            with destination.open("xb") as stream:
                sf.write(stream, data, 16000, format="WAV", subtype="FLOAT")
            info = sf.info(destination)
            if info.samplerate != 16000 or info.channels != 1 or info.frames != len(data):
                raise ValueError("WaveFake derived audio header mismatch")
            realized[row["sample_id"]] = {"filename": name, "source_ref": file,
                "source_row_index": row["parquet_row_index"], "source_rate": rate,
                "target_rate": 16000, "source_frames": len(samples), "target_frames": len(data),
                "resampler": "scipy.signal.resample_poly", "scipy_version": scipy.__version__}
    if len(realized) != len(rows):
        raise ValueError("derived waveform coverage mismatch")
    return realized


def run(run_id: str, smoke: bool):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment required")
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("manifests", "logs", "scores", "diagnostics", "analysis", "derived_audio"):
        (out / part).mkdir()
    try:
        start = time.monotonic()
        assignment = json.loads(SELECT.read_text())
        if (assignment["role"] != "capacity_development_select" or
                assignment["dataset_id"] != "wavefake" or assignment["count"] != 4096):
            raise ValueError("fixed WaveFake selection invalid")
        all_rows = assignment["records"]
        if len(all_rows) != 4096 or any("label" in row for row in all_rows):
            raise ValueError("WaveFake materializer received labels or wrong count")
        rows = all_rows[:32] if smoke else all_rows
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate selected WaveFake sample")
        shutil.copyfile(SELECT, out / "manifests" / SELECT.name)
        bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
        reference = json.loads((BASE / "cache-target-in_the_wild/index.json").read_text())["identity"]
        if (reference["source_run_id"] != bundle["source_run_id"] or
                reference["checkpoint_ref"] != bundle["checkpoint_ref"] or
                reference["preprocess"] != bundle["preprocess"] or
                reference["views"]["num_views"] != 3 or bundle["embedding_dim"] != 160):
            raise ValueError("source/reference frozen provenance mismatch")
        write_new(out / "run_config.json", {"run_id": run_id,
            "branch": subprocess.check_output(["git","branch","--show-current"], cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
            "role": "wavefake_32_waveform_smoke_no_labels" if smoke else "wavefake_fixed_development_cache_no_labels",
            "count": len(rows), "python": sys.version, "torch": torch.__version__,
            "scipy": scipy.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "command": sys.argv, "sample_rate": {"source": 22050, "target": 16000},
            "resampling": {"algorithm": "resample_poly", "up": 320, "down": 441,
                           "window": ["kaiser", 5.0], "padtype": "constant", "dtype": "float32"},
            "labels_read": False, "target90_accessed": False, "final_holdout_accessed": False})
        realized = materialize(rows, out)
        write_new(out / "diagnostics/materialization.json", realized)
        worker_manifest = out / "manifests/worker_select.jsonl"
        with worker_manifest.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"schema_version": "0.3.0", "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"], "root_key": "derived",
                    "audio_relpath": realized[row["sample_id"]]["filename"], "split_role": "select"}) + "\n")
        identity = CacheIdentity(cache_id=f"capacity-wavefake-{run_id}",
            source_run_id=bundle["source_run_id"], checkpoint_ref=bundle["checkpoint_ref"],
            dataset_id="wavefake", split_role="select", manifest_ref=str(worker_manifest.resolve()),
            preprocess=bundle["preprocess"], views=reference["views"], seed=reference["seed"],
            dtype=reference["dtype"], numerical_mode=reference["numerical_mode"]).as_dict()
        cache_ref = out / "feature_cache"
        job = {"schema_version":"0.3.0", "job_type":"inference", "purpose":"select",
            "input_role":"select", "bundle_ref":str((BASE / "frozen/bundle.json").resolve()),
            "manifest_ref":str(worker_manifest.resolve()),
            "data_roots":{"derived":str((out / "derived_audio").resolve())},
            "probe":reference["views"], "numerical_mode":reference["numerical_mode"],
            "worker_slot":0, "worker_count":1, "expected_ids":ids,
            "cache_identity":identity, "output_dir":str(cache_ref.resolve())}
        job_path = out / "diagnostics/extraction_job.json"
        write_new(job_path, job)
        with (out / "logs/extraction.log").open("x") as stream:
            completed = subprocess.run([sys.executable, str(ROOT / "workers/baseline_bridge.py"),
                "extract", "--job", str(job_path)], cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"production extractor failed: {completed.returncode}")
        cache = FeatureCache(cache_ref)
        if (cache.index["identity"] != identity or cache.index["sample_count"] != len(rows) or
                cache.index["num_views"] != 3 or cache.index["feature_dim"] != 160 or
                tuple(cache.verify_expected_ids(ids)) != tuple(ids)):
            raise ValueError("WaveFake production cache schema/coverage invalid")
        features = cache.load_by_id()
        if set(features) != set(ids) or any(z.shape != (3,160) or z.dtype != np.float32 or
                         not np.isfinite(z).all() for z in features.values()):
            raise ValueError("WaveFake feature values invalid")
        validation = {"status":"PASS", "count":len(ids), "unique_ids":len(set(ids)),
            "exact_selected_id_coverage":True, "feature_shape_per_sample":[3,160],
            "all_finite":True, "source_rate":22050, "target_rate":16000,
            "cache_ref":str(cache_ref.resolve()), "elapsed_seconds":time.monotonic()-start,
            "labels_read":False}
        write_new(out / "cache_validation.json", validation)
        print(validation, flush=True)
    except BaseException:
        write_new(out / "failure.json", {"status":"FAIL", "traceback":traceback.format_exc(),
                                          "labels_read":False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.run_id, args.smoke)
