"""Frozen production feature extraction for fixed large development assignments."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
import torch

from eptta.cache.keys import CacheIdentity
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from experiments.codecfake_compat.compat import inspect_audio, materialize_float_wav


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "outputs_v2/ssl_aasist"
PRIOR_CODEC = (ROOT.parent.parent / ".worktrees/exp-local-distribution-tta/experiments/"
               "codecfake_compat/results/compat_20260927a/feature_cache/compat_full512")
FORBIDDEN = {"label", "raw_label", "canonical_label", "attack_id", "correct_before",
             "correct_after", "helpful_update", "harmful_update"}


def write_once(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(domain, run_id, smoke=False):
    if domain not in ("codecfake", "asv2019_la_dev") or not str(sys.executable).endswith(
            "/envs/tta/bin/python"):
        raise ValueError("fixed domain and tta environment required")
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for name in ("manifests", "logs", "scores", "diagnostics", "analysis", "derived_audio"):
        (out / name).mkdir()
    try:
        start = time.perf_counter()
        source = HERE / "manifests" / (domain + "_confirmation_select.json")
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if (manifest.get("role") != "confirmation_select" or manifest.get("dataset_id") != domain or
                manifest.get("count") != 5000 or FORBIDDEN.intersection(manifest) or
                any(FORBIDDEN.intersection(row) for row in manifest["records"])):
            raise ValueError("large confirmation select manifest invalid")
        rows = manifest["records"][:32] if smoke else manifest["records"]
        expected_count = len(rows)
        ids = [row["sample_id"] for row in rows]
        if (ids != sorted(ids) or len(set(ids)) != expected_count or
                len({row["sample_index"] for row in rows}) != expected_count):
            raise ValueError("large assignment order/indices invalid")
        shutil.copyfile(source, out / "manifests" / source.name)
        bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
        reference = json.loads((BASE / "cache-target-in_the_wild/index.json")
                               .read_text(encoding="utf-8"))["identity"]
        if (reference["source_run_id"] != bundle["source_run_id"] or
                reference["checkpoint_ref"] != bundle["checkpoint_ref"] or
                reference["preprocess"] != bundle["preprocess"] or
                reference["views"]["num_views"] != 3 or bundle["embedding_dim"] != 160):
            raise ValueError("frozen/reference cache provenance mismatch")
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                         text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        write_once(out / "run_config.json", {
            "run_id": run_id, "branch": branch, "commit": commit, "python": sys.version,
            "python_executable": sys.executable, "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dataset": domain, "count": expected_count, "selection_seed": 2026,
            "role": "engineering_smoke_no_labels" if smoke else "large_dev_cache_no_labels",
            "method": "unchanged production Frozen SSL-AASIST extraction",
            "parameters": {"views": reference["views"], "numerical_mode": reference["numerical_mode"],
                           "codecfake_non16k": "scipy.signal.resample_poly Kaiser beta5 constant pad float32",
                           "resampler_version": scipy.__version__},
            "command": sys.argv, "target_labels_read": False})
        write_once(out / "provenance.json", {
            "source_bundle": str((BASE / "frozen/bundle.json").resolve()),
            "baseline_id": bundle["baseline_id"], "source_run_id": bundle["source_run_id"],
            "checkpoint_ref": bundle["checkpoint_ref"], "select_manifest": str(source.resolve()),
            "production_worker": str(ROOT / "workers/baseline_bridge.py"),
            "resampler": "scipy.signal.resample_poly" if domain == "codecfake" else "none",
            "resampler_version": scipy.__version__ if domain == "codecfake" else "none",
            "target_labels_read": False})
        worker_manifest = out / "manifests/worker_select.jsonl"
        resampled, native = 0, 0
        with worker_manifest.open("x", encoding="utf-8") as stream:
            for row in rows:
                sample_id = row["sample_id"]
                relative = row["audio_relpath"]
                root_key = "source"
                source_path = Path(row["root_ref"]) / relative
                if domain == "codecfake":
                    info = inspect_audio(source_path)
                    if info.samplerate != 16000:
                        derived_name = Path(relative).name
                        materialize_float_wav(source_path, out / "derived_audio" / derived_name)
                        relative, root_key = derived_name, "derived"
                        resampled += 1
                    else:
                        native += 1
                else:
                    info = sf.info(source_path)
                    if info.samplerate != 16000 or info.frames < 1:
                        raise ValueError("ASV2019 LA dev production rate incompatibility")
                    native += 1
                worker_row = {"schema_version": "0.3.0", "sample_id": sample_id,
                              "sample_index": row["sample_index"], "root_key": root_key,
                              "audio_relpath": relative, "split_role": "select"}
                stream.write(json.dumps(worker_row, allow_nan=False) + "\n")
        identity = CacheIdentity(
            cache_id="large-confirmation-%s-%s" % (domain, run_id),
            source_run_id=bundle["source_run_id"], checkpoint_ref=bundle["checkpoint_ref"],
            dataset_id=domain, split_role="select", manifest_ref=str(worker_manifest.resolve()),
            preprocess=bundle["preprocess"], views=reference["views"],
            seed=reference["seed"], dtype=reference["dtype"],
            numerical_mode=reference["numerical_mode"]).as_dict()
        cache_ref = out / "feature_cache"
        job = {"schema_version": "0.3.0", "job_type": "inference", "purpose": "select",
               "input_role": "select", "bundle_ref": str((BASE / "frozen/bundle.json").resolve()),
               "manifest_ref": str(worker_manifest.resolve()),
               "data_roots": {"source": rows[0]["root_ref"],
                              "derived": str((out / "derived_audio").resolve())},
               "probe": reference["views"], "numerical_mode": reference["numerical_mode"],
               "worker_slot": 0, "worker_count": 1, "expected_ids": ids,
               "cache_identity": identity, "output_dir": str(cache_ref.resolve())}
        job_path = out / "diagnostics/extraction_job.json"
        write_once(job_path, job)
        command = [sys.executable, str(ROOT / "workers/baseline_bridge.py"),
                   "extract", "--job", str(job_path)]
        with (out / "logs/extraction.log").open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError("production extractor failed with exit %d" % completed.returncode)
        cache = FeatureCache(cache_ref)
        if (cache.index.get("identity") != identity or cache.index["sample_count"] != expected_count or
                cache.index["feature_dim"] != 160 or cache.index["num_views"] != 3 or
                cache.index["allow_pickle"] is not False or
                tuple(cache.verify_expected_ids(ids)) != tuple(ids)):
            raise ValueError("large production cache coverage/schema mismatch")
        features = cache.load_by_id()
        if set(features) != set(ids) or any(z.shape != (3, 160) or z.dtype != np.float32 or
                                           not np.isfinite(z).all() for z in features.values()):
            raise ValueError("large feature cache finite/shape failure")
        old_max = None
        if domain == "codecfake" and not smoke:
            prior = FeatureCache(PRIOR_CODEC).load_by_id()
            old_ids = [row["sample_id"] for row in rows if row["old_fixed512_member"]]
            if len(old_ids) != 512 or set(old_ids) != set(prior):
                raise ValueError("old fixed512 nested coverage mismatch")
            old_max = max(float(np.max(np.abs(features[sid] - prior[sid]))) for sid in old_ids)
            if old_max > 1e-5:
                raise ValueError("old fixed512 changed features in larger cache")
        validation = {"status": "PASS", "dataset": domain, "sample_count": expected_count,
                      "unique_ids": expected_count, "exact_selected_id_coverage": True,
                      "feature_shape_per_sample": [3, 160], "dtype": "float32",
                      "all_finite": True, "native16k_count": native, "resampled_count": resampled,
                      "old_fixed512_max_feature_difference": old_max,
                      "cache_ref": str(cache_ref.resolve()), "elapsed_seconds": time.perf_counter() - start,
                      "target_labels_read": False}
        write_once(out / "cache_validation.json", validation)
        print(domain, validation, flush=True)
    except BaseException:
        write_once(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc(),
                                          "target_labels_read": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("codecfake", "asv2019_la_dev"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.domain, args.run_id, args.smoke)
