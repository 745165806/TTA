"""Compile permission-minimal jobs and dispatch the source training bridge."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import os

from eptta.data.io import write_json_new
from eptta.errors import ContractError, ResourceError
from eptta.training.contracts import InitializationRef, ResumeRef, SourceManifestRef, SourceTrainJob
from eptta.training.recipe import load_locked_training_recipe, recipe_hash


def compile_source_job(recipe_ref, phase, output_dir, resume_ref=None):
    recipe = load_locked_training_recipe(recipe_ref)
    payload = recipe["payload"]
    bindings = payload["bindings"]
    source = bindings["source"]
    init = bindings.get("initialization")
    initialization = InitializationRef(**init) if init else None
    resume = ResumeRef(**resume_ref) if resume_ref else None
    spec = SourceTrainJob(
        schema_version="0.1.0", job_type="source_train", model_id=payload["model_id"],
        recipe_lock_ref=str(Path(recipe_ref).resolve()), recipe_hash=recipe_hash(recipe),
        fit=SourceManifestRef(source["roles"]["fit"]["manifest_ref"], source["snapshot_hash"], "fit"),
        source_val=SourceManifestRef(source["roles"]["source_val"]["manifest_ref"],
                                     source["snapshot_hash"], "source_val"),
        initialization=initialization, output_dir=str(Path(output_dir).resolve()),
        training_seed=payload["runtime"].get("training_seed", 13), phase=phase, resume=resume)
    envelope = {"schema_version": "0.1.0", "job_type": "source_train",
                "source_job": asdict(spec),
                "execution": {
                    "architecture": bindings["architecture"],
                    "manifest_hashes": {role: source["roles"][role]["manifest_sha256"]
                                        for role in ("fit", "source_val")},
                    "sample_counts": {role: source["roles"][role]["sample_count"]
                                      for role in ("fit", "source_val")},
                    "data_roots": bindings["data_roots"],
                    "preprocess_ref": bindings["preprocess_ref"],
                    "preprocess_hash": bindings["preprocess_hash"],
                    "training": {key: payload[key] for key in
                                 ("optimizer", "lr", "max_epochs", "scheduler", "loss",
                                  "class_weights_by_name", "sampler_policy",
                                  "augmentation_recipe_ref", "trainable_scope", "weight_decay")},
                    "runtime": payload["runtime"],
                    "class_index_map": payload["class_index_map"],
                    "task_weight_origin": "trained_in_project",
                }}
    return envelope


def launch_source_job(job, worker_ref=None, python_executable=None):
    output = Path(job["source_job"]["output_dir"])
    if output.exists():
        raise ContractError("new training output already exists; use resume-source")
    output.mkdir(parents=True)
    job_path = output / "source_train_job.json"
    write_json_new(job_path, job)
    worker = Path(worker_ref) if worker_ref else Path(__file__).parents[3] / "workers/source_train_bridge.py"
    if not worker.is_file():
        raise ResourceError("source training worker is missing: %s" % worker)
    python = python_executable or sys.executable
    runtime = job["execution"]["runtime"]
    if runtime["strategy"] == "ddp":
        world_size = runtime.get("world_size")
        if type(world_size) is not int or world_size < 2:
            raise ContractError("DDP runtime requires world_size >= 2")
        command = [python, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=%d" % world_size, str(worker), "train", "--job", str(job_path)]
    else:
        command = [python, str(worker), "train", "--job", str(job_path)]
    environment = os.environ.copy()
    gpu_ids = runtime.get("physical_gpu_ids")
    if gpu_ids:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise ResourceError("source training worker failed with exit code %d" % completed.returncode)
    result = output / "run.json"
    if not result.is_file():
        raise ResourceError("source training worker returned without run.json")
    return json.loads(result.read_text(encoding="utf-8"))


def launch_resume_job(run_ref, checkpoint_ref, worker_ref=None, python_executable=None):
    run = Path(run_ref)
    completed_run = run / "run.json"
    if completed_run.is_file() and json.loads(completed_run.read_text(encoding="utf-8")).get("status") == "TRAINED":
        raise ContractError("completed training runs cannot be resumed in place")
    worker = Path(worker_ref) if worker_ref else Path(__file__).parents[3] / "workers/source_train_bridge.py"
    job = json.loads((run / "source_train_job.json").read_text(encoding="utf-8"))
    runtime = job["execution"]["runtime"]
    python = python_executable or sys.executable
    tail = [str(worker), "resume", "--run", str(run), "--checkpoint", str(Path(checkpoint_ref))]
    if runtime["strategy"] == "ddp":
        world_size = runtime.get("world_size")
        if type(world_size) is not int or world_size < 2:
            raise ContractError("DDP runtime requires world_size >= 2")
        command = [python, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=%d" % world_size] + tail
    else:
        command = [python] + tail
    environment = os.environ.copy()
    if runtime.get("physical_gpu_ids"):
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in runtime["physical_gpu_ids"])
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode:
        raise ResourceError("resume worker failed with exit code %d" % completed.returncode)
    return json.loads((run / "run.json").read_text(encoding="utf-8"))
