"""Fixed, source-only real-model diagnostic; never trains or selects hyperparameters."""
import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import numpy as np
import torch

from baselines.probe import three_view_probe
from core import fast_update, score
from eptta.evaluation.metrics import binary_metrics
from stage1_source import _dataset
from experiments.meta_audio_tta.meta_helpers import load_checkpoint as _load_checkpoint
from experiments.meta_audio_tta.meta_helpers import pair_flips as _pair_flips


def fixed_indices(rows, per_class, offset):
    indices = []
    for label in (0, 1):
        matches = sorted((row["sample_id"], index) for index, row in enumerate(rows)
                         if row["canonical_label"] == label)
        if len(matches) < offset + per_class:
            raise ValueError("fixed source subset is incomplete")
        indices.extend(index for _sample_id, index in matches[offset:offset + per_class])
    if len(indices) != 2 * per_class or len(set(indices)) != len(indices):
        raise ValueError("source subset coverage error")
    return indices


def validate_resource_snapshot(snapshot, limits):
    if not all(math.isfinite(value) for value in snapshot.values()):
        raise RuntimeError("source diagnostic resource snapshot is nonfinite")
    if snapshot["free_gpu_gib"] < limits["min_free_gpu_gib"]:
        raise RuntimeError("source diagnostic GPU free memory is below fixed gate")
    if snapshot["gpu_utilization_fraction"] > limits["max_gpu_utilization_fraction"]:
        raise RuntimeError("source diagnostic GPU is already active")
    if snapshot["available_ram_gib"] < limits["min_available_ram_gib"]:
        raise RuntimeError("source diagnostic available RAM is below fixed gate")
    if snapshot["disk_free_gib"] < limits["min_disk_free_gib"]:
        raise RuntimeError("source diagnostic disk headroom is below fixed gate")
    if snapshot["io_some_avg10_fraction"] > limits["max_io_some_avg10_fraction"]:
        raise RuntimeError("source diagnostic disk I/O pressure is above fixed gate")
    if snapshot["cpu_load_fraction"] > limits["max_cpu_load_fraction"]:
        raise RuntimeError("source diagnostic CPU load is above fixed gate")


def resource_snapshot(device, output):
    free_gpu, _total_gpu = torch.cuda.mem_get_info(device)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    utilization_by_gpu = {}
    for line in query.stdout.splitlines():
        index_text, utilization_text = line.split(",", maxsplit=1)
        utilization_by_gpu[int(index_text.strip())] = int(utilization_text.strip()) / 100
    gpu_index = device.index
    if gpu_index is None or gpu_index not in utilization_by_gpu:
        raise RuntimeError("source diagnostic GPU utilization could not be verified")
    gpu_utilization = utilization_by_gpu[gpu_index]
    memory_info = Path("/proc/meminfo").read_text(encoding="utf-8")
    available_kib = int(next(line.split()[1] for line in memory_info.splitlines()
                             if line.startswith("MemAvailable:")))
    disk_free = shutil.disk_usage(output.parent).free
    io_pressure = Path("/proc/pressure/io").read_text(encoding="utf-8")
    some_line = next(line for line in io_pressure.splitlines() if line.startswith("some "))
    io_some_avg10 = float(next(field.split("=", maxsplit=1)[1]
                               for field in some_line.split() if field.startswith("avg10="))) / 100
    return {"free_gpu_gib": free_gpu / 2**30,
            "gpu_utilization_fraction": gpu_utilization,
            "available_ram_gib": available_kib / 2**20,
            "disk_free_gib": disk_free / 2**30,
            "io_some_avg10_fraction": io_some_avg10,
            "cpu_load_fraction": os.getloadavg()[0] / max(1, os.cpu_count())}


def score_episode(detector, system, waveform, sample_index, device, inner_lr):
    """No label argument: both K=0 and K=1 use this exact waveform."""
    views = three_view_probe(waveform, sample_index)[[0, 2]].to(device)
    if system is None:
        with torch.no_grad():
            logits = detector(views[:1])
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise FloatingPointError("invalid CE Frozen logits")
            k0 = logits[0, 0] - logits[0, 1]
        return {"k0": float(k0), "k1": None, "byol_loss": None,
                "backbone_gradient_l1": 0.0, "aux_gradient_l1": 0.0,
                "backbone_update_l1": 0.0, "aux_update_l1": 0.0,
                "projection_norm": None, "target_projection_norm": None,
                "projection_cosine": None}, None
    with torch.no_grad():
        k0 = score(system, views[:1])
    fast, byol_loss, gradients = fast_update(system, views, inner_lr, create_graph=False)
    with torch.no_grad():
        k1 = score(system, views[:1], fast)
        feature = system._online_feature
        target_feature = system._target_feature
        if feature is None or target_feature is None:
            raise AssertionError("BYOL feature capture is missing")
        online_projection = system.projector(feature.detach())[0]
        target_projection = system.target_projector(target_feature.detach())[0]
        projection_norm = float(online_projection.norm())
        target_norm = float(target_projection.norm())
        projection_cosine = float(torch.nn.functional.cosine_similarity(
            online_projection[None], target_projection[None])[0])
        representation = torch.nn.functional.normalize(online_projection, dim=0).cpu().numpy()
    backbone_gradient_l1 = sum(float(gradient.detach().abs().sum())
                               for name, gradient in gradients.items()
                               if name.startswith("online_model."))
    aux_gradient_l1 = sum(float(gradient.detach().abs().sum())
                          for name, gradient in gradients.items()
                          if not name.startswith("online_model."))
    base_params = dict(system.named_parameters())
    backbone_l1 = sum(float((fast[name].detach() - base_params[name].detach()).abs().sum())
                      for name in gradients if name.startswith("online_model."))
    aux_l1 = sum(float((fast[name].detach() - base_params[name].detach()).abs().sum())
                 for name in gradients if not name.startswith("online_model."))
    values = (float(k0), float(k1), float(byol_loss), backbone_gradient_l1,
              aux_gradient_l1, backbone_l1, aux_l1,
              projection_norm, target_norm, projection_cosine)
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("nonfinite source diagnostic")
    return {"k0": float(k0), "k1": float(k1), "byol_loss": float(byol_loss),
            "backbone_gradient_l1": backbone_gradient_l1,
            "aux_gradient_l1": aux_gradient_l1,
            "backbone_update_l1": backbone_l1, "aux_update_l1": aux_l1,
            "projection_norm": projection_norm,
            "target_projection_norm": target_norm,
            "projection_cosine": projection_cosine}, representation


def summarize_scores(rows, threshold=0.0):
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate diagnostic sample ID")
    labels = [row["canonical_label"] for row in rows]
    k0 = [row["k0"] for row in rows]
    result = {"count": len(rows), "k0": binary_metrics(k0, labels, threshold)}
    k1 = [row["k1"] for row in rows]
    if all(value is None for value in k1):
        result["k1"] = "NOT_RUN: CE-only has no trained BYOL head"
        return result
    if any(value is None or not math.isfinite(value) for value in k1):
        raise ValueError("partial/nonfinite K=1 diagnostic coverage")
    result["k1"] = binary_metrics(k1, labels, threshold, k0)
    result["pair_order_flips"] = _pair_flips(k0, k1, labels)
    before = np.asarray(k0, dtype=np.float64)
    after = np.asarray(k1, dtype=np.float64)
    delta = after - before
    slope, intercept = np.linalg.lstsq(
        np.column_stack([before, np.ones_like(before)]), after, rcond=None)[0]
    residual = after - (slope * before + intercept)
    result["score_change"] = {
        "nonzero_count": int(np.count_nonzero(np.abs(delta) > 1e-8)),
        "positive_count": int(np.count_nonzero(delta > 1e-8)),
        "negative_count": int(np.count_nonzero(delta < -1e-8)),
        "mean": float(delta.mean()), "median": float(np.median(delta)),
        "mean_abs": float(np.abs(delta).mean()),
        "affine_slope": float(slope), "affine_intercept": float(intercept),
        "affine_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "before_std": float(before.std()), "after_std": float(after.std())}
    result["update"] = {
        key: {"mean": statistics.fmean(row[key] for row in rows),
              "nonzero_count": sum(row[key] > 0 for row in rows)}
        for key in ("backbone_gradient_l1", "aux_gradient_l1",
                    "backbone_update_l1", "aux_update_l1")}
    return result


def summarize_acoustic_pairs(paired):
    original, perturbed = paired["original"], paired["deterministic_fir"]
    if set(original) != set(perturbed):
        raise AssertionError("fit acoustic pair coverage mismatch")
    return {
        "count": len(original),
        "k0_changed_direction": int(sum(
            (perturbed[key]["k0"] - original[key]["k0"]) *
            (perturbed[key]["k1"] - original[key]["k1"]) < 0
            for key in original)),
        "adaptation_delta_sign_changed": int(sum(
            np.sign(original[key]["k1"] - original[key]["k0"]) !=
            np.sign(perturbed[key]["k1"] - perturbed[key]["k0"])
            for key in original)),
    }


def _record(dataset, index, condition, detector, system, device, inner_lr):
    waveform, label, sample_id = dataset[index]
    sample_index = dataset.rows[index]["sample_index"]
    if condition == "deterministic_fir":
        waveform = three_view_probe(waveform, sample_index)[2]
    values, representation = score_episode(detector, system, waveform,
                                            sample_index, device, inner_lr)
    return {"sample_id": sample_id, "sample_index": sample_index,
            "canonical_label": label, "condition": condition, **values}, representation


def _checkpoint(args):
    gate = json.loads(args.stage1_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise ValueError("diagnostics require stage-1 real-model gate PASS")
    root = args.stage1_root if args.variant in ("ce", "joint") else args.stage2_root
    selection_path = root / args.variant / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = Path(selection["selected_checkpoint"])
    if not checkpoint.is_file() or checkpoint.parent.parent != selection_path.parent:
        raise ValueError("selected checkpoint path is invalid")
    if args.variant in ("ce", "joint"):
        if gate[args.variant]["checkpoint"] != str(checkpoint):
            raise ValueError("selected stage-1 checkpoint differs from gate")
    else:
        history_path = selection_path.parent / "history.jsonl"
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
        if len(rows) != 2 or [row["epoch"] for row in rows] != [1, 2]:
            raise ValueError("stage-2 history is incomplete")
        best = min(rows, key=lambda row: (row["source_val"]["k1_eer"], row["epoch"]))
        if best["checkpoint"] != str(checkpoint) or best["epoch"] != selection["selected_epoch"]:
            raise ValueError("stage-2 selection differs from history")
        if best["source_val"]["nonzero_score_changes"] <= 0:
            raise ValueError("selected meta checkpoint has no observed detector score update")
        run_config = json.loads((selection_path.parent / "run_config.json").read_text())
        if run_config["variant"] != args.variant or run_config["source_checkpoint"] != gate["joint"]["checkpoint"]:
            raise ValueError("stage-2 checkpoint lineage is invalid")
    return checkpoint, selection_path


def run(args):
    if args.output.exists():
        raise FileExistsError("diagnostic output path already exists")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["source_only"] or args.variant not in config["allowed_variants"]:
        raise ValueError("source-only diagnostic config mismatch")
    if args.device not in config["allowed_devices"]:
        raise ValueError("source diagnostic may only use GPU 2 or 3")
    checkpoint, selection_path = _checkpoint(args)
    snapshot = resource_snapshot(torch.device(args.device), args.output)
    validate_resource_snapshot(snapshot, config["resource_gate"])
    source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
    if abs(config["inner_lr"] - 0.0003) > 1e-12 or config["decision_threshold"] != 0.0:
        raise ValueError("fixed diagnostic learning rate or threshold changed")
    device = torch.device(args.device)
    detector, system = _load_checkpoint(source_config, checkpoint, device)
    if (system is None) != (args.variant == "ce"):
        raise ValueError("CE/BYOL checkpoint type mismatch")
    source_val = _dataset(source_config, "source_val")
    val_indices = fixed_indices(source_val.rows, config["source_val_count_per_class"],
                                config["source_val_offset_per_class"])
    if args.variant in ("cross", "same"):
        prior_ids = set(json.loads((selection_path.parent / "run_config.json").read_text())["validation_ids"])
        if any(source_val.rows[index]["sample_id"] in prior_ids for index in val_indices):
            raise ValueError("diagnostic source_val overlaps stage-2 validation subset")
    fit = _dataset(source_config, "fit")
    fit.set_epoch(0)
    fit_indices = fixed_indices(fit.rows, config["fit_count_per_class"], 0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run_config.json").write_text(json.dumps({
        "diagnostic_config": config, "source_config": str(args.source_config),
        "variant": args.variant, "checkpoint": str(checkpoint),
        "resource_snapshot": snapshot,
        "checkpoint_selection": str(selection_path), "stage1_gate": str(args.stage1_gate),
        "device": args.device, "command": sys.argv,
        "source_val_ids": [source_val.rows[index]["sample_id"] for index in val_indices],
        "fit_ids": [fit.rows[index]["sample_id"] for index in fit_indices]
    }, indent=2) + "\n", encoding="utf-8")
    original_params = ({name: param.detach().clone() for name, param in system.named_parameters()
                        if param.requires_grad} if system is not None else {})
    original_buffers = ({name: buffer.detach().clone() for name, buffer in system.named_buffers()}
                        if system is not None else {})
    val_rows, representations = [], []
    with (args.output / "source_val.scores.jsonl").open("x", encoding="utf-8") as stream:
        for index in val_indices:
            row, representation = _record(source_val, index, "original",
                                           detector, system, device, config["inner_lr"])
            val_rows.append(row)
            if representation is not None:
                representations.append(representation)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    fit_rows = []
    if system is not None:
        with (args.output / "fit_acoustic.scores.jsonl").open("x", encoding="utf-8") as stream:
            for condition in config["acoustic_conditions"]:
                for index in fit_indices:
                    row, _representation = _record(fit, index, condition,
                                                   detector, system, device, config["inner_lr"])
                    fit_rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    if system is not None:
        if any(not torch.equal(value, dict(system.named_parameters())[name])
               for name, value in original_params.items()):
            raise AssertionError("source diagnostic changed checkpoint parameters")
        if any(not torch.equal(value, dict(system.named_buffers())[name])
               for name, value in original_buffers.items()):
            raise AssertionError("source diagnostic changed checkpoint buffers")
    summary = {"status": "PASS", "variant": args.variant,
               "checkpoint": str(checkpoint), "source_val": summarize_scores(
                   val_rows, config["decision_threshold"]),
               "fit_acoustic": {}, "representation": "NOT_RUN: CE-only has no BYOL head",
               "checkpoint_state_unchanged": True}
    if system is not None:
        summary["fit_acoustic"] = {
            condition: summarize_scores([row for row in fit_rows if row["condition"] == condition],
                                        config["decision_threshold"])
            for condition in config["acoustic_conditions"]}
        paired = {condition: {row["sample_id"]: row for row in fit_rows
                              if row["condition"] == condition}
                  for condition in config["acoustic_conditions"]}
        summary["fit_acoustic"]["perturbation_pair"] = summarize_acoustic_pairs(paired)
        matrix = np.stack(representations)
        summary["representation"] = {
            "count": len(matrix),
            "normalized_dimension_std_mean": float(matrix.std(axis=0).mean()),
            "normalized_dimension_std_max": float(matrix.std(axis=0).max()),
            "projection_norm_mean": statistics.fmean(row["projection_norm"] for row in val_rows),
            "target_projection_norm_mean": statistics.fmean(
                row["target_projection_norm"] for row in val_rows),
            "projection_cosine_mean": statistics.fmean(
                row["projection_cosine"] for row in val_rows)}
        if not all(math.isfinite(value) for value in summary["representation"].values()):
            raise FloatingPointError("nonfinite representation summary")
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "variant": args.variant,
                      "source_val_count": len(val_rows), "fit_count": len(fit_rows)},
                     allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--stage1-gate", type=Path, required=True)
    parser.add_argument("--variant", choices=("ce", "joint", "cross", "same"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("real-model source diagnostics require a declared GPU")
    run(args)


if __name__ == "__main__":
    main()
