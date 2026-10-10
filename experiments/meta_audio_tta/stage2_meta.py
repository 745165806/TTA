"""Fixed-budget real SSL-AASIST second-order source meta comparison."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import torch

from author_training import build_author_model
from baselines.probe import three_view_probe
from core import BYOLSystem, configure_meta, episodic_scores, meta_objective
from eptta.evaluation.metrics import binary_metrics
from stage1_source import _construction, _dataset, _exclusive_save


def _views(dataset, index, device):
    audio, label, sample_id = dataset[index]
    sample_index = dataset.rows[index]["sample_index"]
    views = three_view_probe(audio, sample_index)[[0, 2]].to(device)
    return views, label, sample_id


def _build_from_joint(source_config, checkpoint, device, ema_decay):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("mode") != "joint" or not state.get("byol_state"):
        raise ValueError("stage 2 requires a valid joint CE+BYOL checkpoint")
    online, _ = build_author_model(_construction(source_config), device)
    online.model.load_state_dict(state["model_state"], strict=True)
    target, _ = build_author_model(_construction(source_config), device)
    system = BYOLSystem(online.model, target.model,
                        source_config["architecture"]["class_index_map"],
                        ema_decay=ema_decay).to(device)
    saved = state["byol_state"]
    system.projector.load_state_dict(saved["projector"], strict=True)
    system.predictor.load_state_dict(saved["predictor"], strict=True)
    system.target_model.load_state_dict(saved["target_model"], strict=True)
    system.target_projector.load_state_dict(saved["target_projector"], strict=True)
    names = configure_meta(system)
    return system, names


def _validation_indices(dataset, per_class):
    indices = []
    for label in (0, 1):
        matching = sorted((row["sample_id"], i) for i, row in enumerate(dataset.rows)
                          if row["canonical_label"] == label)
        if len(matching) < per_class:
            raise ValueError("source_val has too few items in one class")
        indices.extend(i for _id, i in matching[:per_class])
    return indices


def _evaluate(system, dataset, indices, inner_lr, device):
    frozen, adapted, labels, changed = [], [], [], 0
    for index in indices:
        views, label, _id = _views(dataset, index, device)
        k0, k1, _loss, _grads = episodic_scores(system, views[:1], views, inner_lr)
        frozen.append(float(k0))
        adapted.append(float(k1))
        labels.append(label)
        changed += abs(float(k1 - k0)) > 1e-8
    k0_metrics = binary_metrics(frozen, labels, 0.0)
    k1_metrics = binary_metrics(adapted, labels, 0.0)
    return {"count": len(indices), "k0_eer": float(k0_metrics["eer"]),
            "k1_eer": float(k1_metrics["eer"]),
            "k0_auc": float(k0_metrics["auroc"]),
            "k1_auc": float(k1_metrics["auroc"]),
            "nonzero_score_changes": changed}


def run(config, source_config, source_selection, stage1_gate, variant, output, device):
    if output.exists():
        raise FileExistsError("refusing to overwrite stage-2 run")
    if config["meta_batch_tasks"] != 1:
        raise ValueError("this episodic audio comparison fixes one task per outer step")
    if variant not in config["variants"]:
        raise ValueError("variant is not in fixed config")
    selected = json.loads(source_selection.read_text(encoding="utf-8"))
    checkpoint = Path(selected["selected_checkpoint"])
    gate = json.loads(stage1_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("joint", {}).get("checkpoint") != str(checkpoint):
        raise ValueError("selected joint checkpoint lacks a matching real-model stage-1 gate")
    if not checkpoint.is_file() or checkpoint.parent.parent != source_selection.parent:
        raise ValueError("stage-2 checkpoint is not selected from supplied joint run")
    if source_selection.parent.name != "joint":
        raise ValueError("stage 2 cannot start from CE-only training")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    system, names = _build_from_joint(source_config, checkpoint, device,
                                      config["ema_after_outer_step"])
    optimizer = torch.optim.Adam([dict(system.named_parameters())[name] for name in names],
                                 lr=config["outer_lr"],
                                 weight_decay=config["outer_weight_decay"])
    fit = _dataset(source_config, "fit")
    val = _dataset(source_config, "source_val")
    if {row["sample_id"] for row in fit.rows} & {row["sample_id"] for row in val.rows}:
        raise ValueError("fit/source_val overlap")
    if 2 * config["tasks_per_epoch"] > len(fit):
        raise ValueError("not enough fit samples for independent cross queries")
    val_indices = _validation_indices(val, config["source_val_per_class"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps({"variant": variant, "meta": config,
        "source_config": str(source_selection), "source_checkpoint": str(checkpoint),
        "validation_ids": [val.rows[i]["sample_id"] for i in val_indices],
        "selected_fast_parameters": names}, indent=2) + "\n", encoding="utf-8")
    best = (math.inf, None)
    for epoch in range(1, config["epochs"] + 1):
        fit.set_epoch(epoch)
        task_rng = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(len(fit), generator=task_rng).tolist()
        supports = order[:config["tasks_per_epoch"]]
        queries = order[config["tasks_per_epoch"]:2 * config["tasks_per_epoch"]]
        start = time.perf_counter()
        ce_sum = byol_sum = loss_sum = 0.0
        system.adaptation_mode()
        for support_i, independent_query_i in zip(supports, queries):
            support, support_label, _support_id = _views(fit, support_i, device)
            if variant == "cross":
                query, query_label, _query_id = _views(fit, independent_query_i, device)
            else:
                query, query_label = support, support_label
            optimizer.zero_grad(set_to_none=True)
            outer, detail = meta_objective(system, support, query, query_label,
                                            config["inner_lr"], config["auxiliary_weight"])
            if not torch.isfinite(outer):
                raise FloatingPointError("nonfinite meta objective")
            outer.backward()
            if any(not torch.isfinite(dict(system.named_parameters())[name].grad).all()
                   for name in names):
                raise FloatingPointError("nonfinite meta gradient")
            if any(parameter.grad is not None for parameter in system.target_model.parameters()) or \
                    any(parameter.grad is not None for parameter in system.target_projector.parameters()):
                raise AssertionError("EMA target received an outer gradient")
            optimizer.step()
            system.update_ema()
            ce_sum += float(detail["query_ce"].detach())
            byol_sum += float(detail["query_byol"].detach())
            loss_sum += float(outer.detach())
        validation = _evaluate(system, val, val_indices, config["inner_lr"], device)
        checkpoint_out = output / "checkpoints" / f"epoch_{epoch:04d}.pt"
        _exclusive_save({"epoch": epoch, "variant": variant,
                         "source_checkpoint": str(checkpoint),
                         "model_state": system.online_model.state_dict(),
                         "byol_state": {"projector": system.projector.state_dict(),
                                        "predictor": system.predictor.state_dict(),
                                        "target_model": system.target_model.state_dict(),
                                        "target_projector": system.target_projector.state_dict()},
                         "optimizer_state": optimizer.state_dict(),
                         "source_val": validation}, checkpoint_out)
        candidate = (validation["k1_eer"], epoch)
        if candidate < best:
            best = candidate
        record = {"epoch": epoch, "variant": variant, "fit_support_tasks": len(supports),
                  "unique_fit_audio_exposures": len(supports) * (2 if variant == "cross" else 1),
                  "outer_optimizer_steps": len(supports),
                  "mean_outer_loss": loss_sum / len(supports),
                  "mean_query_ce": ce_sum / len(supports),
                  "mean_query_byol": byol_sum / len(supports),
                  "source_val": validation, "checkpoint": str(checkpoint_out),
                  "elapsed_seconds": time.perf_counter() - start}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    (output / "selection.json").write_text(json.dumps({"selected_epoch": best[1],
        "source_val_k1_eer": best[0],
        "selected_checkpoint": str(output / "checkpoints" / f"epoch_{best[1]:04d}.pt")},
        indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--stage1-gate", type=Path, required=True)
    parser.add_argument("--variant", choices=("cross", "same"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("real second-order meta run requires declared CUDA device")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
    run(config, source_config, args.source_selection, args.stage1_gate,
        args.variant, args.output,
        torch.device(args.device))


if __name__ == "__main__":
    main()
