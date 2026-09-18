"""Compile source jobs and dispatch through this environment's interpreter."""
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

from eptta.data.io import write_json_new
from eptta.errors import ContractError, ResourceError
from eptta.training.contracts import InitializationRef, SourceManifestRef, SourceTrainJob
from eptta.training.recipe import load_training_recipe


def _run_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "source-%s-%s" % (stamp, secrets.token_hex(3))


def compile_source_job(recipe_ref, phase, output_dir):
    recipe = load_training_recipe(recipe_ref)
    payload, bindings = recipe["payload"], recipe["payload"]["bindings"]
    source, init = bindings["source"], bindings.get("initialization")
    initialization = None if init is None else InitializationRef(
        artifact_ref=init["artifact_ref"], scope=init["scope"],
        pretraining_provenance=init["pretraining_provenance"])
    spec = SourceTrainJob(
        schema_version="0.3.0", job_type="source_train", run_id=_run_id(),
        model_id=payload["model_id"], recipe_ref=str(Path(recipe_ref).resolve()),
        fit=SourceManifestRef(source["roles"]["fit"]["manifest_ref"],
                              source["roles"]["fit"]["dataset_id"], "fit"),
        source_val=SourceManifestRef(source["roles"]["source_val"]["manifest_ref"],
                                     source["roles"]["source_val"]["dataset_id"], "source_val"),
        initialization=initialization, output_dir=str(Path(output_dir).resolve()),
        training_seed=payload["runtime"].get("training_seed", 13), phase=phase)
    training_keys = ("optimizer", "lr", "max_epochs", "scheduler", "loss",
                     "class_weights_by_name", "sampler_policy", "augmentation_recipe_ref",
                     "trainable_scope", "weight_decay")
    training = {key: payload[key] for key in training_keys}
    if "scheduler_horizon_epochs" in payload:
        training["scheduler_horizon_epochs"] = payload["scheduler_horizon_epochs"]
    return {"schema_version": "0.3.0", "job_type": "source_train", "source_job": asdict(spec),
            "execution": {"architecture": bindings["architecture"],
                          "data_roots": bindings["data_roots"],
                          "preprocess": bindings["preprocess"], "training": training,
                          "runtime": payload["runtime"], "class_index_map": payload["class_index_map"],
                          "task_weight_origin": "trained_in_project",
                          "random_rule": "explicit_sample_index_prng_v1"}}


def _command(job, worker, tail):
    runtime = job["execution"]["runtime"]
    if runtime["strategy"] == "ddp":
        world_size = runtime.get("world_size")
        if type(world_size) is not int or world_size < 2:
            raise ContractError("DDP runtime requires world_size >= 2")
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=%d" % world_size, str(worker)] + tail
    return [sys.executable, str(worker)] + tail


def _environment(job):
    environment = os.environ.copy()
    gpu_ids = job["execution"]["runtime"].get("physical_gpu_ids")
    if gpu_ids:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    return environment


def _save_run_provenance(output, job):
    """Save a small self-contained project-code snapshot, never data or weights."""
    root = Path(__file__).resolve().parents[3]
    source = job["source_job"]
    versions = {}
    for distribution in ("torch", "torchaudio", "numpy", "scipy", "PyYAML", "fairseq",
                         "librosa", "soundfile", "tensorboardX"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    def git(*args):
        completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    status = git("status", "--porcelain") or ""
    author_root = Path(job["execution"]["architecture"]["repository_ref"])
    def author_git(*args):
        completed = subprocess.run(["git", "-C", str(author_root), *args], text=True,
                                   capture_output=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    author_status = author_git("status", "--porcelain") or ""
    meta = {"schema_version": "0.3.0", "run_id": source["run_id"],
            "model_id": source["model_id"], "seed": source["training_seed"],
            "recipe_ref": source["recipe_ref"], "fit": source["fit"],
            "source_val": source["source_val"], "random_rule": job["execution"]["random_rule"],
            "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status},
            "author_source": {"repository_ref": str(author_root.resolve()),
                              "commit": author_git("rev-parse", "HEAD"),
                              "dirty": bool(author_status), "status": author_status},
            "environment": {"python": platform.python_version(),
                            "platform": platform.platform(), "packages": versions},
            "code_snapshot_ref": "code_snapshot"}
    write_json_new(output / "config.yaml", job)
    write_json_new(output / "meta.json", meta)
    snapshot = output / "code_snapshot"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(root / "src" / "eptta", snapshot / "src" / "eptta", ignore=ignore)
    shutil.copytree(root / "workers", snapshot / "workers", ignore=ignore)
    for name in ("pyproject.toml", "environment.yml"):
        if (root / name).is_file():
            shutil.copy2(root / name, snapshot / name)
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if (root / name).is_file():
            shutil.copy2(root / name, snapshot / name)
    if author_status:
        completed = subprocess.run(["git", "-C", str(author_root), "diff", "--binary"],
                                   capture_output=True, check=False)
        if completed.returncode == 0:
            (output / "author_code.diff").write_bytes(completed.stdout)


def launch_source_job(job):
    output = Path(job["source_job"]["output_dir"])
    if output.exists():
        raise ContractError("new training output already exists; use --resume for this run")
    output.mkdir(parents=True)
    job_path = output / "source_train_job.json"
    write_json_new(job_path, job)
    _save_run_provenance(output, job)
    worker = Path(__file__).parents[3] / "workers/source_train_bridge.py"
    if not worker.is_file():
        raise ResourceError("source training worker is missing")
    completed = subprocess.run(_command(job, worker, ["train", "--job", str(job_path)]),
                               check=False, env=_environment(job))
    if completed.returncode:
        raise ResourceError("source training worker failed with exit code %d" % completed.returncode)
    result = output / "run.json"
    if not result.is_file():
        raise ResourceError("source training worker returned without run.json")
    return json.loads(result.read_text(encoding="utf-8"))


def launch_resume_job(run_ref, checkpoint_ref):
    run = Path(run_ref)
    worker = Path(__file__).parents[3] / "workers/source_train_bridge.py"
    job_path = run / "source_train_job.json"
    if not worker.is_file() or not job_path.is_file() or not Path(checkpoint_ref).is_file():
        raise ResourceError("resume worker, job, or checkpoint is missing")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    completed = subprocess.run(_command(job, worker, ["resume", "--run", str(run),
                                                       "--checkpoint", str(Path(checkpoint_ref))]),
                               check=False, env=_environment(job))
    if completed.returncode:
        raise ResourceError("resume worker failed with exit code %d" % completed.returncode)
    return json.loads((run / "run.json").read_text(encoding="utf-8"))
