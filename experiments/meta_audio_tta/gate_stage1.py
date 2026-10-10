"""Stage-1 completion and selected joint-checkpoint real-model gate."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import torch

from core import episodic_scores, meta_objective
from smoke_real import _read_fit_pair
from stage2_meta import _build_from_joint
from baselines.probe import three_view_probe
from eptta.baselines.ports.audio_native import SCOPE_A, preregistered_parameter_names


def _run_record(folder, mode, config):
    with (folder / "history.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    run_config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    steps = math.ceil(25380 / config["batch_size"])
    if len(rows) != config["epochs"] or [row["epoch"] for row in rows] != list(
            range(1, config["epochs"] + 1)):
        raise ValueError(mode + " incomplete epoch history")
    if any(row["source_train_steps"] != steps or row["fit_count"] != 25380 or
           row["source_val_count"] != 5654 or not math.isfinite(row["source_val_eer"])
           for row in rows):
        raise ValueError(mode + " invalid training/validation accounting")
    best = min(rows, key=lambda row: (row["source_val_eer"], row["epoch"]))
    if (best["epoch"] != selection["selected_epoch"] or
            best["checkpoint"] != selection["selected_checkpoint"]):
        raise ValueError(mode + " source_val selection mismatch")
    if run_config["mode"] != mode or run_config["run_id"] != config["run_id"]:
        raise ValueError(mode + " run config mismatch")
    checkpoint = Path(selection["selected_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(mode + " selected checkpoint missing")
    return {"mode": mode, "selected_epoch": best["epoch"],
            "source_val_eer": best["source_val_eer"], "checkpoint": str(checkpoint),
            "initial_state": run_config["initial_state"],
            "epochs": len(rows), "steps_per_epoch": steps}


def run(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite stage-1 gate")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ce = _run_record(args.root / "ce", "ce", config)
    joint = _run_record(args.root / "joint", "joint", config)
    if joint["source_val_eer"] >= 0.5:
        raise ValueError("selected joint source_val EER is not better than chance; stop meta stage")
    if ce["initial_state"] != joint["initial_state"] or not Path(ce["initial_state"]).is_file():
        raise ValueError("CE and joint do not share the same initial model state")
    device = torch.device(args.device)
    system, names = _build_from_joint(config, Path(joint["checkpoint"]), device,
                                      config["ema_decay"])
    torch.cuda.reset_peak_memory_stats(device)
    excluded = {f"encoder.{i}.0.bn1.{part}" for i in range(1, 6)
                for part in ("weight", "bias")}
    expected = {"online_model." + name for name in
                set(preregistered_parameter_names(system.online_model, SCOPE_A)) - excluded}
    backbone_names = {name for name in names if name.startswith("online_model.")}
    if backbone_names != expected or len(names) != 34:
        raise AssertionError("selected joint BN affine whitelist changed")
    fit_pair = _read_fit_pair(Path(config["roles"]["fit"]["manifest"]),
                              Path(config["data_roots"]["asvspoof2019_la"]))
    views = [three_view_probe(row["waveform"], row["sample_index"])[[0, 2]].to(device)
             for row in fit_pair]
    before_buffers = {name: buffer.detach().clone() for name, buffer in system.named_buffers()}
    before_params = {name: param.detach().clone() for name, param in system.named_parameters()}
    k0, k1, _loss, grads = episodic_scores(system, views[0][:1], views[0], 3e-4)
    backbone_grad = sum(float(grads[name].detach().abs().sum()) for name in names
                        if name.startswith("online_model."))
    if backbone_grad <= 0 or abs(float(k1 - k0)) <= 1e-8:
        raise AssertionError("selected joint backbone BN update does not affect spoof score")
    params = dict(system.named_parameters())
    meta_results = {}
    for variant, query, label in (("cross", views[1], fit_pair[1]["canonical_label"]),
                                  ("same", views[0], fit_pair[0]["canonical_label"])):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        outer, _detail = meta_objective(system, views[0], query, label, 3e-4)
        gradients = torch.autograd.grad(outer, [params[name] for name in names],
                                        allow_unused=False)
        norm = sum(float(gradient.detach().abs().sum()) for gradient in gradients)
        if norm <= 0 or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("selected joint " + variant + " meta gradient invalid")
        torch.cuda.synchronize(device)
        meta_results[variant] = {"outer_loss": float(outer.detach()), "gradient_l1": norm,
                                 "forward_and_meta_gradient_seconds": time.perf_counter() - started}
    if any(not torch.equal(value, dict(system.named_buffers())[name])
           for name, value in before_buffers.items()):
        raise AssertionError("selected joint episode changed BN/target buffers")
    if any(not torch.equal(value, dict(system.named_parameters())[name])
           for name, value in before_params.items()):
        raise AssertionError("selected joint episode changed its initial parameters")
    return {"status": "PASS", "ce": ce, "joint": joint,
            "selected_backbone_bn_tensors": sum(name.startswith("online_model.") for name in names),
            "selected_fast_tensors": len(names), "k0": float(k0), "k1": float(k1),
            "score_change": float(k1 - k0), "backbone_gradient_l1": backbone_grad,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "meta": meta_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("stage-1 selected checkpoint gate requires real CUDA model")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
