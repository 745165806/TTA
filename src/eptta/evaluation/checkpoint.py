"""Compile and launch architecture-compatible checkpoint evaluation jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys

from eptta.data.io import sha256_file, write_json_new
from eptta.errors import ContractError, DataError, ResourceError
from eptta.models.author import inspect_author_repository


SUPPORTED_MODELS = {"aasist_source": "aasist", "ssl_aasist_source": "ssl_aasist"}


def _protocol_count(protocol_ref):
    seen = set()
    count = 0
    with Path(protocol_ref).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.strip().split()
            if len(fields) != 5 or fields[4] not in {"bonafide", "spoof"}:
                raise DataError(f"invalid ASVspoof protocol row {number}")
            sample_id = fields[1]
            if sample_id in seen:
                raise DataError(f"duplicate ASVspoof protocol sample ID: {sample_id}")
            seen.add(sample_id)
            count += 1
    if not count:
        raise DataError("evaluation protocol is empty")
    return count


def compile_checkpoint_evaluation(model_id, checkpoint_ref, protocol_ref, audio_dir,
                                  output_dir, cfg, batch_size=48, num_workers=4,
                                  threshold=0.0, physical_gpu_id=None,
                                  evaluation_tag="unspecified", asv_scores_ref=None):
    if model_id not in SUPPORTED_MODELS:
        raise ContractError("unsupported checkpoint evaluation model: " + model_id)
    if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 0:
        raise ContractError("evaluation batch size/workers are invalid")
    if physical_gpu_id is not None and (type(physical_gpu_id) is not int or physical_gpu_id < 0):
        raise ContractError("physical GPU ID must be a non-negative integer")
    checkpoint = Path(checkpoint_ref)
    protocol = Path(protocol_ref)
    audio = Path(audio_dir)
    if not checkpoint.is_file() or not protocol.is_file() or not audio.is_dir():
        raise ResourceError("checkpoint, protocol, or audio directory is missing")
    repo_key = SUPPORTED_MODELS[model_id]
    repository = cfg["paths"]["source_repos"].get(repo_key)
    if not repository:
        raise ResourceError("source repository binding is unresolved: " + repo_key)
    architecture = inspect_author_repository(model_id, repository)
    asv_scores = None
    tdcf_implementation = None
    if asv_scores_ref:
        asv_path = Path(asv_scores_ref)
        tdcf_path = Path(cfg["paths"]["source_repos"].get("aasist") or "") / "evaluation.py"
        if not asv_path.is_file() or not tdcf_path.is_file():
            raise ResourceError("ASV scores or pinned t-DCF implementation is missing")
        asv_scores = {"artifact_ref": str(asv_path.resolve()), "sha256": sha256_file(asv_path)}
        tdcf_implementation = {"artifact_ref": str(tdcf_path.resolve()), "sha256": sha256_file(tdcf_path)}
    initialization = None
    if model_id == "ssl_aasist_source":
        init_ref = cfg["paths"].get("generic_ssl_initialization")
        if not init_ref or not Path(init_ref).is_file():
            raise ResourceError("SSL-AASIST model construction requires the configured generic SSL initialization")
        initialization = {"artifact_ref": str(Path(init_ref).resolve()),
                          "sha256": sha256_file(init_ref),
                          "scope": "generic_ssl_frontend_only",
                          "pretraining_provenance": "configured_generic_ssl_frontend"}
    worker = Path(__file__).parents[3] / "workers/checkpoint_eval_bridge.py"
    if not worker.is_file():
        raise ResourceError("checkpoint evaluation worker is missing: " + str(worker))
    return {"schema_version": "0.1.0", "job_type": "checkpoint_evaluation",
            "model_id": model_id, "architecture": architecture,
            "initialization": initialization,
            "checkpoint_ref": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "protocol_ref": str(protocol.resolve()),
            "protocol_sha256": sha256_file(protocol),
            "expected_sample_count": _protocol_count(protocol),
            "audio_dir": str(audio.resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "batch_size": batch_size, "num_workers": num_workers,
            "threshold": float(threshold), "physical_gpu_id": physical_gpu_id,
            "evaluation_tag": str(evaluation_tag),
            "asv_scores": asv_scores, "tdcf_implementation": tdcf_implementation,
            "worker_ref": str(worker.resolve()), "worker_sha256": sha256_file(worker)}


def launch_checkpoint_evaluation(job, python_executable=None):
    output = Path(job["output_dir"])
    if output.exists():
        raise ContractError("evaluation output exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = output.parent / (output.name + ".job.json")
    write_json_new(job_path, job)
    worker = Path(job["worker_ref"])
    if sha256_file(worker) != job["worker_sha256"]:
        raise ContractError("checkpoint evaluation worker changed after job compilation")
    environment = os.environ.copy()
    if job["physical_gpu_id"] is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(job["physical_gpu_id"])
    completed = subprocess.run([python_executable or sys.executable, str(worker), "--job", str(job_path)],
                               check=False, env=environment)
    if completed.returncode:
        raise ResourceError("checkpoint evaluation worker failed with exit code %d" % completed.returncode)
    run_path = output / "run.json"
    metrics_path = output / "metrics.json"
    if not run_path.is_file() or not metrics_path.is_file():
        raise ResourceError("checkpoint evaluation returned without run/metrics artifacts")
    return {"schema_version": "0.1.0", "status": "EVALUATED",
            "output_dir": str(output), "run_ref": str(run_path),
            "metrics_ref": str(metrics_path),
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]}
