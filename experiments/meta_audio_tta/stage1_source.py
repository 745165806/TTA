"""Matched CE / CE+BYOL source training with one shared random initialization."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import torch
from torch.nn import functional as F

from author_training import ManifestDataset, build_author_model
from baselines.probe import three_view_probe
from core import BYOLSystem
from eptta.training.selection import equal_error_rate


def _exclusive_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite " + str(path))
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("partial checkpoint already exists")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _construction(config):
    return {"source_job": {"model_id": "ssl_aasist_source",
                           "initialization": config["generic_frontend_initialization"]},
            "execution": {"architecture": config["architecture"]}}


def _dataset(config, role):
    ref = Path(config["roles"][role]["manifest"])
    return ManifestDataset(ref, config["data_roots"], role,
                           training_seed=int(config["seed"]))


def _loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       generator=generator, num_workers=0, pin_memory=True)


def _views(waveform, ids, rows_by_id, device):
    tensors = []
    for audio, sample_id in zip(waveform, ids):
        sample_index = rows_by_id[sample_id]["sample_index"]
        tensors.append(three_view_probe(audio, sample_index)[[0, 2]])
    return torch.stack(tensors).to(device, non_blocking=True)


def _losses(mode, detector, system, views, canonical, byol_weight):
    batch = views.shape[0]
    flat = views.reshape(batch * 2, -1)
    if system is None:
        logits = detector(flat)
    else:
        logits, features = system._online(flat)
    native = 1 - canonical
    class_weights = torch.tensor([0.1, 0.9], device=flat.device)
    ce = F.cross_entropy(logits, native.repeat_interleave(2), weight=class_weights)
    if mode == "ce":
        return ce, ce, None
    prediction = F.normalize(system.predictor(system.projector(features)), dim=-1)
    target = F.normalize(system._target(flat), dim=-1).detach()
    a, b = prediction.reshape(batch, 2, -1).unbind(1)
    ta, tb = target.reshape(batch, 2, -1).unbind(1)
    byol = ((2 - 2 * (a * tb).sum(-1)) + (2 - 2 * (b * ta).sum(-1))).mean() / 2
    return ce + byol_weight * byol, ce, byol


def _validate(detector, dataset, batch_size, device):
    detector.eval()
    loader = _loader(dataset, batch_size, False, 0)
    scores, labels, seen = [], [], set()
    with torch.no_grad():
        for audio, canonical, ids in loader:
            if seen.intersection(ids):
                raise ValueError("duplicate source_val ID")
            seen.update(ids)
            logits = detector(audio.to(device))
            if logits.shape != (len(ids), 2) or not torch.isfinite(logits).all():
                raise FloatingPointError("source_val logits invalid")
            scores.extend((logits[:, 0] - logits[:, 1]).cpu().tolist())
            labels.extend(canonical.tolist())
    if len(seen) != len(dataset):
        raise ValueError("source_val coverage mismatch")
    return float(equal_error_rate(scores, labels))


def prepare_initial(config, output, device):
    if output.exists():
        raise FileExistsError("initial state already exists")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    adapter, patch = build_author_model(_construction(config), device)
    _exclusive_save({"model_state": adapter.model.state_dict(),
                     "architecture": config["architecture"],
                     "initialization": config["generic_frontend_initialization"],
                     "seed": config["seed"], "model_patch": patch}, output)
    print(json.dumps({"status": "PASS", "initial_state": str(output),
                      "parameter_count": sum(p.numel() for p in adapter.model.parameters())}))


def run(config, mode, initial, output, device):
    if output.exists():
        raise FileExistsError("run directory already exists")
    initial_state = torch.load(initial, map_location="cpu", weights_only=True)
    if (initial_state["architecture"] != config["architecture"] or
            initial_state["initialization"] != config["generic_frontend_initialization"] or
            initial_state["seed"] != config["seed"]):
        raise ValueError("shared initial state conflicts with training config")
    fit = _dataset(config, "fit")
    val = _dataset(config, "source_val")
    if {r["sample_id"] for r in fit.rows} & {r["sample_id"] for r in val.rows}:
        raise ValueError("fit/source_val overlap")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    adapter, _patch = build_author_model(_construction(config), device)
    detector = adapter.model
    detector.load_state_dict(initial_state["model_state"], strict=True)
    if mode == "joint":
        target, _patch = build_author_model(_construction(config), device)
        system = BYOLSystem(detector, target.model, {"spoof": 0, "bonafide": 1},
                            ema_decay=config["ema_decay"]).to(device)
        trainable = [p for p in system.parameters() if p.requires_grad]
    else:
        system = None
        trainable = list(detector.parameters())
    optimizer = torch.optim.Adam(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps({**config, "mode": mode,
        "initial_state": str(initial)}, indent=2) + "\n", encoding="utf-8")
    history = output / "history.jsonl"
    rows_by_id = {row["sample_id"]: row for row in fit.rows}
    best_eer, best_epoch = math.inf, None
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        fit.set_epoch(epoch)
        if system is None:
            detector.train()
        else:
            system.source_train_mode()
        loader = _loader(fit, config["batch_size"], True, config["seed"] + epoch)
        total_loss, total_ce, total_byol, steps = 0.0, 0.0, 0.0, 0
        for audio, canonical, ids in loader:
            views = _views(audio, ids, rows_by_id, device)
            canonical = canonical.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, byol = _losses(mode, detector, system, views, canonical,
                                     config["byol_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite source loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
                raise FloatingPointError("non-finite source gradient")
            optimizer.step()
            if system is not None:
                system.update_ema()
            total_loss += float(loss.detach())
            total_ce += float(ce.detach())
            total_byol += 0 if byol is None else float(byol.detach())
            steps += 1
        eer = _validate(detector, val, config["validation_batch_size"], device)
        checkpoint = output / "checkpoints" / f"epoch_{epoch:04d}.pt"
        _exclusive_save({"epoch": epoch, "mode": mode,
                         "model_state": detector.state_dict(),
                         "byol_state": None if system is None else {
                             "projector": system.projector.state_dict(),
                             "predictor": system.predictor.state_dict(),
                             "target_model": system.target_model.state_dict(),
                             "target_projector": system.target_projector.state_dict()},
                         "optimizer_state": optimizer.state_dict(),
                         "source_val_eer": eer, "source_train_steps": steps}, checkpoint)
        if (eer, epoch) < (best_eer, best_epoch or math.inf):
            best_eer, best_epoch = eer, epoch
        row = {"epoch": epoch, "source_val_eer": eer, "best_epoch": best_epoch,
               "source_train_steps": steps, "fit_count": len(fit),
               "source_val_count": len(val), "mean_loss": total_loss / steps,
               "mean_ce": total_ce / steps, "mean_byol": total_byol / steps if system else None,
               "checkpoint": str(checkpoint), "elapsed_seconds": time.perf_counter() - epoch_start}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    (output / "selection.json").write_text(json.dumps({"selected_epoch": best_epoch,
        "source_val_eer": best_eer,
        "selected_checkpoint": str(output / "checkpoints" / f"epoch_{best_epoch:04d}.pt")},
        indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("ce", "joint", "prepare"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if torch.cuda.is_available() is False or not args.device.startswith("cuda:"):
        raise RuntimeError("full source training requires declared CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if args.mode == "prepare":
        prepare_initial(config, args.initial, device)
    else:
        run(config, args.mode, args.initial, args.output, device)


if __name__ == "__main__":
    main()
