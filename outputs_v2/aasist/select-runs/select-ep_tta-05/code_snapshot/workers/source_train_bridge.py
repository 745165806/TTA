#!/usr/bin/env python3
"""Permission-minimal source training worker in the active tta environment."""
from __future__ import absolute_import, print_function

import argparse
import contextlib
import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

# Required by torch deterministic CUDA GEMM.  This is set before this worker
# imports torch or initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "compat"))
from author_training import (ManifestDataset, EpochShardSampler, build_author_model,
                             canonical_to_native_tensor)

ACTIVE_OUTPUT = None


def atomic_json(path, value):
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path), dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_history(path, value):
    fields = ["epoch", "global_step", "train_weighted_loss_numerator", "train_sample_count",
              "source_val_loss", "source_val_eer", "source_val_count", "lr", "improved",
              "best_epoch", "epoch_checkpoint_ref"]
    exists = os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: value[key] for key in fields})
        stream.flush()
        os.fsync(stream.fileno())


def validate_job(job):
    if set(job) != {"schema_version", "job_type", "source_job", "execution"}:
        raise ValueError("source worker envelope has unknown fields")
    if job["schema_version"] != "0.3.0" or job["job_type"] != "source_train":
        raise ValueError("source worker job version/type mismatch")
    source = job["source_job"]
    allowed = {"schema_version", "job_type", "run_id", "model_id", "recipe_ref",
               "fit", "source_val", "initialization", "output_dir", "training_seed", "phase"}
    if set(source) != allowed:
        raise ValueError("source job has forbidden fields")
    if source["fit"]["role"] != "fit" or source["source_val"]["role"] != "source_val":
        raise ValueError("worker accepts fit/source_val only")
    serialized = json.dumps({"source_job": source, "data_roots": job["execution"].get("data_roots")},
                            sort_keys=True).lower()
    for forbidden in ("target_test", "control_test", '"select"', '"audit"', '"cal0"', '"cal1"'):
        if forbidden in serialized:
            raise ValueError("target/evaluation role leaked into source worker")
    if job["execution"].get("task_weight_origin") != "trained_in_project":
        raise ValueError("external task weight injection is forbidden")
    training = job["execution"]["training"]
    horizon = training.get("scheduler_horizon_epochs", training.get("max_epochs"))
    if type(horizon) is not int or horizon < int(training["max_epochs"]):
        raise ValueError("scheduler horizon must be an integer >= max_epochs")
    expected_augmentation = ({"none_author_freq_aug_false"} if source["model_id"] == "aasist_source"
                             else {"none_project_freq_aug_false",
                                   "none_project_deviation_requires_review"})
    if training.get("augmentation_recipe_ref") not in expected_augmentation:
        raise ValueError("unimplemented/unreviewed source augmentation recipe")
    if training.get("loss") != "weighted_categorical_cross_entropy":
        raise ValueError("worker only implements the audited semantic weighted CE")


def verify_preprocess(job):
    payload = job["execution"]["preprocess"]
    if not isinstance(payload, dict):
        raise ValueError("source preprocess config must be a mapping")
    expected = {"decode": "soundfile_float32_mono_mean_require_16khz",
                "train_unit": ("repeat_or_random_crop_64600_seed_epoch_sample",
                               "repeat_or_random_crop_64600_seed_epoch_sample_index"),
                "eval_unit": "repeat_or_crop_first_64600"}
    if (payload.get("decode") != expected["decode"] or payload.get("train_unit") not in expected["train_unit"] or
            payload.get("eval_unit") != expected["eval_unit"]):
        raise ValueError("worker does not implement the configured decode/unit profile")


def setup_distributed(runtime):
    import torch
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    strategy = runtime["strategy"]
    if world > 1:
        if strategy != "ddp":
            raise ValueError("torchrun world size requires runtime.strategy=ddp")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    elif strategy not in ("single_gpu", "single_process"):
        raise ValueError("runtime strategy requests DDP but WORLD_SIZE is one")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if runtime.get("allow_cpu", False) is not True:
            raise RuntimeError("CUDA is unavailable; real source training does not fall back to CPU")
        device = torch.device("cpu")
    return rank, world, local_rank, device


def seed_all(seed, rank):
    import torch
    random.seed(seed + rank)
    try:
        import numpy
        numpy.random.seed(seed + rank)
    except ImportError:
        pass
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, sampler, batch_size, workers):
    import torch
    indices = sampler.indices() if sampler is not None else list(range(len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False,
                                       drop_last=False, num_workers=workers, pin_memory=True)


def class_weights(job, device):
    import torch
    weights = job["execution"]["training"]["class_weights_by_name"]
    mapping = job["execution"]["class_index_map"]
    ordered = [None, None]
    for name in ("bonafide", "spoof"):
        ordered[mapping[name]] = float(weights[name])
    if any(value <= 0 for value in ordered):
        raise ValueError("class weights must be positive")
    return torch.tensor(ordered, dtype=torch.float32, device=device)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def train_epoch(model, adapter, loader, optimizer, scheduler, scaler, device, job, rank, world, accum):
    import torch
    import torch.nn.functional as functional
    model.train()
    weights = class_weights(job, device)
    optimizer.zero_grad(set_to_none=True)
    loss_numerator = 0.0
    weight_denominator = 0.0
    sample_count = 0
    optimizer_steps = 0
    pending = 0
    for batch_index, (waveform, canonical, _ids) in enumerate(loader):
        waveform = waveform.to(device, non_blocking=True)
        canonical = canonical.to(device, non_blocking=True)
        native = canonical_to_native_tensor(canonical, job["execution"]["class_index_map"])
        pending += 1
        is_last = batch_index + 1 == len(loader)
        synchronize = pending == accum or is_last
        sync_context = contextlib.nullcontext()
        if world > 1 and not synchronize:
            sync_context = model.no_sync()
        precision = job["execution"]["runtime"]["precision"]
        amp_enabled = precision in ("float16", "bfloat16") and device.type == "cuda"
        amp_dtype = torch.float16 if precision == "float16" else torch.bfloat16
        with sync_context:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _embedding, logits = adapter.forward(waveform, freq_aug=False)
                losses = functional.cross_entropy(logits, native, weight=weights, reduction="none")
                numerator = losses.sum()
            scaler.scale(numerator).backward()
        batch_weight = weights[native].sum().detach()
        loss_numerator += float(numerator.detach().cpu())
        weight_denominator += float(batch_weight.cpu())
        sample_count += waveform.shape[0]
        if synchronize:
            denominator = torch.tensor(weight_denominator, dtype=torch.float64, device=device)
            if world > 1:
                torch.distributed.all_reduce(denominator)
            scaler.unscale_(optimizer)
            scale = float(world) / float(denominator.item())
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                         for parameter in model.parameters())
            if not finite:
                raise FloatingPointError("non-finite source training gradient")
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending = 0
            weight_denominator = 0.0
    totals = torch.tensor([loss_numerator, sample_count, optimizer_steps], dtype=torch.float64, device=device)
    if world > 1:
        torch.distributed.all_reduce(totals)
    return {"weighted_loss_numerator": float(totals[0]), "sample_count": int(totals[1]),
            "optimizer_steps": int(totals[2] / world)}


def eer(scores, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("source_val EER undefined: one class is absent")
    pairs = sorted(zip(scores, labels), reverse=True)
    fa = 0
    ta = 0
    points = [(0.0, 1.0)]
    index = 0
    while index < len(pairs):
        value = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == value:
            if pairs[index][1] == 1:
                ta += 1
            else:
                fa += 1
            index += 1
        points.append((float(fa) / negatives, 1.0 - float(ta) / positives))
    for first, second in zip(points, points[1:]):
        d0, d1 = first[0] - first[1], second[0] - second[1]
        if d0 == 0:
            return first[0]
        if d0 * d1 <= 0:
            fraction = abs(d0) / (abs(d0) + abs(d1))
            far = first[0] + fraction * (second[0] - first[0])
            frr = first[1] + fraction * (second[1] - first[1])
            return (far + frr) / 2.0
    best = min(points, key=lambda pair: abs(pair[0] - pair[1]))
    return sum(best) / 2.0


def validate(model, adapter, loader, device, job):
    import torch
    import torch.nn.functional as functional
    before_modes = {name: module.training for name, module in model.named_modules()}
    before_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    model.eval()
    scores, labels, ids = [], [], []
    numerator, denominator = 0.0, 0.0
    weights = class_weights(job, device)
    with torch.no_grad():
        for waveform, canonical, sample_ids in loader:
            waveform = waveform.to(device)
            canonical = canonical.to(device)
            native = canonical_to_native_tensor(canonical, job["execution"]["class_index_map"])
            _embedding, logits = adapter.forward(waveform, freq_aug=False)
            losses = functional.cross_entropy(logits, native, weight=weights, reduction="none")
            numerator += float(losses.sum().cpu())
            denominator += float(weights[native].sum().cpu())
            mapping = job["execution"]["class_index_map"]
            score = logits[:, mapping["spoof"]] - logits[:, mapping["bonafide"]]
            scores.extend(float(value) for value in score.cpu())
            labels.extend(int(value) for value in canonical.cpu())
            ids.extend(sample_ids)
    if len(ids) != len(set(ids)) or len(ids) != len(loader.dataset):
        raise ValueError("source_val must contain every ID exactly once")
    after_buffers = dict(model.named_buffers())
    for name, value in before_buffers.items():
        if not torch.equal(value, after_buffers[name].detach().cpu()):
            raise ValueError("source_val changed model buffer: %s" % name)
    # The caller restores train/eval state explicitly for the next epoch.
    return {"source_val_loss": numerator / denominator, "source_val_eer": eer(scores, labels),
            "source_val_count": len(ids),
            "source_val_ids": sorted(ids),
            "module_modes_before_validation": before_modes}


def rng_state(rank):
    import torch
    state = {"rank": rank, "python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy
        state["numpy"] = numpy.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        import numpy
        numpy.random.set_state(state["numpy"])


def epoch_checkpoint_path(checkpoints, epoch):
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    return os.path.join(checkpoints, "epoch_%04d.pt" % epoch)


def _atomic_link_or_copy(source, destination):
    if not os.path.isfile(source):
        raise ValueError("checkpoint alias source is missing: %s" % source)
    directory = os.path.dirname(destination)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(destination), dir=directory)
    os.close(fd)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def publish_checkpoint_alias(source, destination):
    """Atomically advance a mutable convenience alias without changing epoch history."""
    _atomic_link_or_copy(source, destination)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, job,
                    model_patch, rank, world, sampler, train_result, validation,
                    best_eer, best_epoch):
    import torch
    local_rng = rng_state(rank)
    all_rng = [None for _ in range(world)]
    if world > 1:
        torch.distributed.all_gather_object(all_rng, local_rng)
    else:
        all_rng[0] = local_rng
    if rank != 0:
        return None
    if os.path.exists(path):
        raise ValueError("immutable epoch checkpoint already exists: %s" % path)
    source = job["source_job"]
    value = {"schema_version": "0.3.0", "model_state": unwrap(model).state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict() if scheduler else None,
             "scaler_state": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
             "best_metric": float(best_eer), "best_epoch": int(best_epoch),
             "train_metrics": train_result, "source_val_metrics": validation,
             "rng_states": all_rng, "sampler_state": {"epoch": sampler.epoch, "policy": sampler.policy,
                                                        "world_size": world},
             "run_id": source["run_id"], "recipe_ref": source["recipe_ref"],
             "fit": source["fit"], "source_val": source["source_val"],
             "architecture": job["execution"]["architecture"], "model_patch": model_patch,
             "class_index_map": job["execution"]["class_index_map"],
             "initialization": source.get("initialization"), "training_seed": source["training_seed"],
             "task_weight_origin": "trained_in_project",
             "random_rule": job["execution"]["random_rule"],
             "training": job["execution"]["training"],
             "runtime": job["execution"]["runtime"],
             "preprocess": job["execution"]["preprocess"],
             "resume_kind": "complete"}
    temporary = path + ".partial"
    torch.save(value, temporary)
    os.replace(temporary, path)
    return path


def build_optimizer(model, job, steps_per_epoch):
    import torch
    training = job["execution"]["training"]
    if training["optimizer"] != "adam":
        raise ValueError("only audited Adam recipe is currently implemented")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]),
                                 weight_decay=float(training.get("weight_decay") or 0.0))
    scheduler_name = training["scheduler"]
    if scheduler_name in ("none", None):
        scheduler = None
    elif scheduler_name in ("cosine", "cosine_author_audited"):
        total_steps = int(training.get("scheduler_horizon_epochs", training["max_epochs"])) * int(steps_per_epoch)
        minimum_factor = 5e-6 / float(training["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: minimum_factor + (1.0 - minimum_factor) *
            0.5 * (1.0 + math.cos(float(step) / total_steps * math.pi)))
    else:
        raise ValueError("unsupported/unreviewed scheduler")
    return optimizer, scheduler


def execute(job, resume_checkpoint=None, stop_after_epoch=None):
    import torch
    validate_job(job)
    verify_preprocess(job)
    source = job["source_job"]
    execution = job["execution"]
    runtime = execution["runtime"]
    rank, world, local_rank, device = setup_distributed(runtime)
    seed_all(source["training_seed"], rank)
    roots = execution["data_roots"]
    fit = ManifestDataset(source["fit"]["manifest_ref"], roots, "fit", source["training_seed"])
    val = ManifestDataset(source["source_val"]["manifest_ref"], roots, "source_val")
    if fit.random_rule != execution["random_rule"]:
        raise ValueError("new training requires explicit sample_index in the fit manifest")
    adapter, model_patch = build_author_model(job, device)
    model = adapter.model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        adapter.model = model
    preview_sampler = EpochShardSampler(len(fit), source["training_seed"], rank, world,
                                        int(runtime["per_gpu_batch_size"]), execution["training"]["sampler_policy"])
    local_batches = int(math.ceil(float(len(preview_sampler.indices())) / int(runtime["per_gpu_batch_size"])))
    steps_per_epoch = int(math.ceil(float(local_batches) / int(runtime["grad_accum_steps"])))
    optimizer, scheduler = build_optimizer(model, job, steps_per_epoch)
    precision = runtime["precision"]
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "float16" and device.type == "cuda")
    start_epoch, global_step, best_eer, best_epoch = 0, 0, float("inf"), -1
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        required_resume = {"model_state", "optimizer_state", "scheduler_state", "scaler_state",
                           "epoch", "global_step", "best_metric", "best_epoch", "rng_states",
                           "sampler_state", "random_rule", "run_id", "training", "runtime", "preprocess"}
        if checkpoint.get("schema_version") != "0.3.0" or required_resume - set(checkpoint):
            raise ValueError("checkpoint is evaluation/warm-start only; full resume state or new random rule is missing")
        if checkpoint.get("random_rule") != execution["random_rule"]:
            raise ValueError("legacy digest-derived crop sequence cannot be exactly resumed; start a new run")
        for key, expected in (("run_id", source["run_id"]), ("recipe_ref", source["recipe_ref"]),
                              ("fit", source["fit"]), ("source_val", source["source_val"])):
            if checkpoint.get(key) != expected:
                raise ValueError("resume rejected: %s changed" % key)
        for key, expected in (("architecture", execution["architecture"]),
                              ("class_index_map", execution["class_index_map"]),
                              ("initialization", source.get("initialization")),
                              ("training_seed", source["training_seed"]),
                              ("training", execution["training"]),
                              ("runtime", execution["runtime"]),
                              ("preprocess", execution["preprocess"]),
                              ("task_weight_origin", "trained_in_project")):
            if checkpoint.get(key) != expected:
                raise ValueError("resume rejected: %s changed" % key)
        if checkpoint["sampler_state"]["world_size"] != world:
            raise ValueError("exact resume rejected: DDP world size changed")
        if checkpoint.get("model_patch") != model_patch:
            raise ValueError("exact resume rejected: model construction patch changed")
        unwrap(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_states"][rank])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_eer = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
    output = source["output_dir"]
    checkpoints = os.path.join(output, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)
    sampler = EpochShardSampler(len(fit), source["training_seed"], rank, world,
                                int(runtime["per_gpu_batch_size"]), execution["training"]["sampler_policy"])
    val_loader = None
    if rank == 0:
        val_loader = make_loader(val, None, int(runtime["validation_batch_size"]), int(runtime.get("num_workers", 0)))
    phase = source["phase"]
    max_epochs = int(execution["training"]["max_epochs"])
    if phase == "smoke":
        max_epochs = min(max_epochs, int(runtime.get("smoke_epochs", 1)))
    if stop_after_epoch is not None:
        if type(stop_after_epoch) is not int or not start_epoch <= stop_after_epoch < max_epochs:
            raise ValueError("stop-after epoch must be within the epochs this process would run")
    log_path = os.path.join(output, "train_log.jsonl")
    metrics_path = os.path.join(output, "metrics.jsonl")
    history_path = os.path.join(output, "history.csv")
    started = time.time()
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        sampler.set_epoch(epoch)
        fit.set_epoch(epoch)
        loader = make_loader(fit, sampler, int(runtime["per_gpu_batch_size"]), int(runtime.get("num_workers", 0)))
        train_result = train_epoch(model, adapter, loader, optimizer, scheduler, scaler, device, job, rank, world,
                                   int(runtime["grad_accum_steps"]))
        global_step += train_result["optimizer_steps"]
        if world > 1:
            torch.distributed.barrier()
        validation = None
        if rank == 0:
            training_model = adapter.model
            adapter.model = unwrap(model)
            try:
                validation = validate(unwrap(model), adapter, val_loader, device, job)
            finally:
                adapter.model = training_model
        if world > 1:
            values = [validation]
            torch.distributed.broadcast_object_list(values, src=0)
            validation = values[0]
        improved = validation["source_val_eer"] < best_eer
        if improved:
            best_eer, best_epoch = validation["source_val_eer"], epoch
        epoch_path = epoch_checkpoint_path(checkpoints, epoch)
        save_checkpoint(epoch_path, model, optimizer, scheduler, scaler, epoch, global_step,
                        job, model_patch, rank, world, sampler, train_result, validation,
                        best_eer, best_epoch)
        if rank == 0:
            publish_checkpoint_alias(epoch_path, os.path.join(checkpoints, "last.pt"))
            if improved:
                publish_checkpoint_alias(epoch_path, os.path.join(checkpoints, "best.pt"))
            epoch_ref = "checkpoints/" + os.path.basename(epoch_path)
            record = {"epoch": epoch, "global_step": global_step, "train": train_result,
                      "source_val_loss": validation["source_val_loss"],
                      "source_val_eer": validation["source_val_eer"],
                      "source_val_count": validation["source_val_count"],
                      "source_val_ids": validation["source_val_ids"],
                      "lr": optimizer.param_groups[0]["lr"], "improved": improved,
                      "best_epoch": best_epoch, "epoch_checkpoint_ref": epoch_ref}
            record["sampler_coverage"] = sampler.disclosure()
            append_jsonl(log_path, record)
            append_jsonl(metrics_path, {"epoch": epoch, "source_val_eer": validation["source_val_eer"],
                                        "checkpoint_ref": epoch_ref})
            append_history(history_path, {
                "epoch": epoch, "global_step": global_step,
                "train_weighted_loss_numerator": train_result["weighted_loss_numerator"],
                "train_sample_count": train_result["sample_count"],
                "source_val_loss": validation["source_val_loss"],
                "source_val_eer": validation["source_val_eer"],
                "source_val_count": validation["source_val_count"],
                "lr": optimizer.param_groups[0]["lr"], "improved": improved,
                "best_epoch": best_epoch, "epoch_checkpoint_ref": epoch_ref})
        if world > 1:
            torch.distributed.barrier()
        completed_epoch = epoch
        if stop_after_epoch is not None and epoch == stop_after_epoch:
            break
    if rank == 0:
        status = "TRAINED" if completed_epoch + 1 == max_epochs else "INTERRUPTED"
        run = {"schema_version": "0.3.0", "status": status, "phase": phase,
               "training_run_id": source["run_id"], "model_id": source["model_id"],
               "recipe_ref": source["recipe_ref"], "fit": source["fit"],
               "source_val": source["source_val"],
               "architecture": execution["architecture"], "class_index_map": execution["class_index_map"],
               "model_patch": model_patch,
               "initialization": source.get("initialization"), "training_seed": source["training_seed"],
               "task_weight_origin": "trained_in_project", "world_size": world,
               "sampler_coverage": sampler.disclosure(),
               "random_rule": execution["random_rule"], "exact_resume_supported": True,
               "best_epoch": best_epoch, "best_metric": best_eer,
               "metrics_ref": "metrics.jsonl", "history_ref": "history.csv",
               "train_log_ref": "train_log.jsonl",
               "elapsed_seconds": time.time() - started}
        atomic_json(os.path.join(output, "run.json"), run)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main(argv=None):
    global ACTIVE_OUTPUT
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--job", required=True)
    train.add_argument("--stop-after-epoch", type=int)
    resume = sub.add_parser("resume")
    resume.add_argument("--run", required=True)
    resume.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        with open(args.job, encoding="utf-8") as stream:
            job = json.load(stream)
        ACTIVE_OUTPUT = job.get("source_job", {}).get("output_dir")
        execute(job, stop_after_epoch=args.stop_after_epoch)
    else:
        job_path = os.path.join(args.run, "source_train_job.json")
        with open(job_path, encoding="utf-8") as stream:
            job = json.load(stream)
        ACTIVE_OUTPUT = job.get("source_job", {}).get("output_dir")
        if os.path.abspath(job["source_job"]["output_dir"]) != os.path.abspath(args.run):
            raise ValueError("resume run path disagrees with original job")
        execute(job, args.checkpoint)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        failure = {"schema_version": "0.1.0", "status": "INTERRUPTED",
                   "error_type": "KeyboardInterrupt", "message": "source training interrupted"}
        if ACTIVE_OUTPUT and os.path.isdir(ACTIVE_OUTPUT):
            atomic_json(os.path.join(ACTIVE_OUTPUT, "failure.json"), failure)
        print(json.dumps(failure), file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        failure = {"schema_version": "0.1.0", "status": "FAILED",
                   "error_type": type(exc).__name__, "message": str(exc)}
        if ACTIVE_OUTPUT and os.path.isdir(ACTIVE_OUTPUT):
            atomic_json(os.path.join(ACTIVE_OUTPUT, "failure.json"), failure)
        print(json.dumps(failure), file=sys.stderr)
        exit_code = 2
    sys.exit(exit_code)
