"""Materialize the fixed Codecfake dev selection with the production extractor."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from eptta.cache.keys import CacheIdentity
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECT = ROOT / "experiments/multidomain_mechanism/manifests/codecfake_mechanism_select.json"
PA_SELECT = HERE / "manifests/asv2019_pa_dev_mechanism_select.json"
BUNDLE = ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"
REFERENCE = ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild/index.json"


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(count, run_id, domain="codecfake"):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment is required")
    if domain not in ("codecfake", "asv2019_pa_dev"):
        raise ValueError("unsupported fixed development domain")
    selected_ref = SELECT if domain == "codecfake" else PA_SELECT
    expected_total = 512 if domain == "codecfake" else 270
    if count not in (32, expected_total):
        raise ValueError("only preregistered smoke or full counts are allowed")
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "scores", "diagnostics", "analysis"):
        (out / name).mkdir()
    try:
        doc = json.loads(selected_ref.read_text(encoding="utf-8"))
        rows = doc["records"][:count]
        if doc["dataset_id"] != domain or len(doc["records"]) != expected_total:
            raise ValueError("fixed development selection mismatch")
        if len({r["sample_id"] for r in rows}) != count:
            raise ValueError("duplicate selected ID")
        forbidden = {"label", "raw_label", "canonical_label", "attack_id"}
        if forbidden.intersection(doc) or any(forbidden.intersection(r) for r in rows):
            raise ValueError("label field entered extraction selection")
        bundle, *_ = verify_frozen_export(BUNDLE)
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))["identity"]
        if ref["preprocess"] != bundle["preprocess"] or ref["source_run_id"] != bundle["source_run_id"]:
            raise ValueError("reference feature path differs from Frozen bundle")
        metadata = []
        for row in rows:
            path = Path(row["root_ref"]) / row["audio_relpath"]
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
                raise ValueError("selected waveform violates production preprocessing: " + row["sample_id"])
            metadata.append({"sample_id": row["sample_id"], "sample_rate": info.samplerate,
                             "channels": info.channels, "frames": info.frames})
        write_new(out / "diagnostics/waveform_preflight.json", metadata)
        manifest = out / "diagnostics/extraction_select.jsonl"
        with manifest.open("x", encoding="utf-8") as stream:
            for i, row in enumerate(rows):
                worker_row = {"schema_version": "0.3.0", "sample_id": row["sample_id"],
                              "sample_index": i, "root_key": row["root_key"],
                              "audio_relpath": row["audio_relpath"], "split_role": "select"}
                stream.write(json.dumps(worker_row, allow_nan=False) + "\n")
        cache_id = "task-objective-%s-%s" % (domain, run_id)
        numerical_mode = dict(ref["numerical_mode"])
        identity = CacheIdentity(cache_id=cache_id, source_run_id=bundle["source_run_id"],
                                 checkpoint_ref=bundle["checkpoint_ref"], dataset_id=domain,
                                 split_role="select", manifest_ref=str(manifest.resolve()),
                                 preprocess=bundle["preprocess"], views=ref["views"],
                                 seed=ref["seed"], dtype=ref["dtype"],
                                 numerical_mode=numerical_mode)
        job = {"schema_version": "0.3.0", "job_type": "inference", "purpose": "select",
               "input_role": "select", "bundle_ref": str(BUNDLE.resolve()),
               "manifest_ref": str(manifest.resolve()),
               "data_roots": {rows[0]["root_key"]: rows[0]["root_ref"]},
               "probe": ref["views"], "numerical_mode": numerical_mode,
               "worker_slot": 0, "worker_count": 1,
               "expected_ids": [r["sample_id"] for r in rows],
               "cache_identity": identity.as_dict(),
               "output_dir": str((out / "diagnostics/feature_cache").resolve())}
        write_new(out / "diagnostics/extraction_job.json", job)
        command = [sys.executable, str(ROOT / "workers/baseline_bridge.py"),
                   "extract", "--job", str(out / "diagnostics/extraction_job.json")]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        write_new(out / "run_config.json", {"run_id": run_id, "branch": branch, "commit": commit,
                  "python": sys.version, "pytorch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu_available": torch.cuda.is_available(), "seed": 2026,
                  "dataset": domain, "sample_count": count,
                  "method": "production Frozen SSL-AASIST extraction",
                  "parameters": {"sample_rate": 16000, "waveform_length": 64600,
                                 "embedding_dim": bundle["embedding_dim"],
                                 "num_views": ref["views"]["num_views"],
                                 "probe": ref["views"], "numerical_mode": numerical_mode},
                  "command": command})
        write_new(out / "provenance.json", {"baseline_id": bundle["baseline_id"],
                  "source_run_id": bundle["source_run_id"],
                  "checkpoint_ref": bundle["checkpoint_ref"],
                  "bundle_ref": str(BUNDLE), "select_manifest": str(selected_ref),
                  "target_labels_read": False})
        with (out / "logs/extraction.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       check=False)
        if completed.returncode:
            raise RuntimeError("production extractor failed: exit %d" % completed.returncode)
        cache = FeatureCache(out / "diagnostics/feature_cache")
        index = cache.index
        if (index["identity"] != identity.as_dict() or index["sample_count"] != count or
                index["feature_dim"] != bundle["embedding_dim"] or index["num_views"] != 3):
            raise ValueError("cache schema/provenance mismatch")
        found = set()
        for ids, block in cache.iter_chunks():
            if block.shape[1:] != (3, bundle["embedding_dim"]) or not np.isfinite(block).all():
                raise ValueError("nonfinite or malformed feature block")
            found.update(ids)
        if found != {r["sample_id"] for r in rows}:
            raise ValueError("exact selected sample coverage failed")
        write_new(out / "analysis/summary.json", {"status": "PASS", "sample_count": count,
                  "cache_ref": str((out / "diagnostics/feature_cache").resolve()),
                  "sample_rate": 16000, "num_views": 3,
                  "embedding_dim": bundle["embedding_dim"],
                  "exact_coverage": True, "finite": True,
                  "target_labels_read": False})
        print(out, flush=True)
    except BaseException:
        write_new(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(32, 270, 512), required=True)
    parser.add_argument("--domain", choices=("codecfake", "asv2019_pa_dev"), default="codecfake")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run(args.count, args.run_id or "%s_cache_%d_%s" %
        (args.domain, args.count, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")), args.domain)
