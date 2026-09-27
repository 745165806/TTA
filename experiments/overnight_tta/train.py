"""Full cached-source training for residual adapter and budget-matched static ERM."""
import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "distribution_conditioned_head"))
from experiments.distribution_conditioned_head.train import load_source, source_head
from eptta.evaluation.metrics import binary_metrics
from residual import ResidualDetector, supervised_contrastive


def _before_deadline(config):
    deadline = datetime.fromisoformat(config["budget_deadline_utc"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("nightly budget expired; do not start a new experiment")


def _balanced_batches(labels, seed, epoch, batch_size):
    half = batch_size // 2
    bona = [i for i, label in enumerate(labels) if label == 0]
    spoof = [i for i, label in enumerate(labels) if label == 1]
    rng = random.Random(seed * 100003 + epoch)
    rng.shuffle(spoof)
    steps = math.ceil(len(spoof) / half)
    if len(spoof) < steps * half:
        spoof += spoof[:steps * half - len(spoof)]
    for step in range(steps):
        indices = spoof[step * half:(step + 1) * half] + rng.sample(bona, half)
        rng.shuffle(indices)
        yield indices


@torch.no_grad()
def _validate(model, static_w, static_b, data, device):
    x, labels = data["x"].to(device), [int(v) for v in data["y"].tolist()]
    result = {}
    model.eval()
    for name, view in (("clean", 0), ("noise", 1), ("fir", 2)):
        static = (x[:, view] @ static_w + static_b).cpu().tolist()
        adapted = model(x[:, view]).cpu().tolist()
        for arm, scores in (("erm", static), ("residual", adapted)):
            metric = binary_metrics(scores, labels, 0.0)
            result[f"{arm}_{name}_auc"] = metric["auroc"]
            result[f"{arm}_{name}_eer"] = metric["eer"]
    return result


def run(config_ref, run_id, seed):
    config = json.loads(Path(config_ref).read_text())
    _before_deadline(config)
    output = Path(__file__).parent / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({**config, "seed": seed, "run_id": run_id}, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle, source_w, source_b = source_head(Path(config["bundle_ref"]))
    fit, val = load_source(config, "fit", bundle), load_source(config, "select", bundle)
    if set(fit["audio_ids"]) & set(val["audio_ids"]):
        raise ValueError("source fit/select original ID overlap")
    x, y = fit["x"].to(device), fit["y"].to(device)
    source_w, source_b = source_w.to(device), source_b.to(device)
    residual = ResidualDetector(source_w, source_b).to(device)
    static_w = source_w.detach().clone().requires_grad_(True)
    static_b = source_b.detach().clone().requires_grad_(True)
    residual_opt = torch.optim.Adam(residual.parameters(), lr=config["source_lr"])
    static_opt = torch.optim.Adam((static_w, static_b), lr=config["source_lr"])
    history, seen = [], set()
    steps = presentations = numerical_failures = 0
    started = time.monotonic()
    best = {"erm": (float("inf"), float("inf"), float("inf")),
            "residual": (float("inf"), float("inf"), float("inf"))}
    for epoch in range(1, config["source_epochs"] + 1):
        residual.train()
        static_losses, residual_losses, contrast_losses = [], [], []
        for ids in _balanced_batches(fit["y"].tolist(), seed, epoch, config["source_batch_size"]):
            seen.update(ids)
            presentations += len(ids)
            steps += 1
            z0, z1, labels = x[ids, 0], x[ids, 1], y[ids]
            static_opt.zero_grad(set_to_none=True)
            static_logits = torch.cat((z0 @ static_w + static_b, z1 @ static_w + static_b))
            static_bce = F.binary_cross_entropy_with_logits(static_logits, labels.repeat(2))
            static_bce.backward()
            static_opt.step()
            residual_opt.zero_grad(set_to_none=True)
            h0, h1 = residual.embed(z0), residual.embed(z1)
            logits = torch.cat((residual.head(h0).squeeze(-1), residual.head(h1).squeeze(-1)))
            bce = F.binary_cross_entropy_with_logits(logits, labels.repeat(2))
            contrast = supervised_contrastive(h0, h1, labels, config["supcon_temperature"])
            loss = bce + config["supcon_weight"] * contrast
            if not all(torch.isfinite(t).item() for t in (static_bce, bce, contrast, loss)):
                numerical_failures += 1
                raise ValueError("nonfinite source training loss")
            loss.backward()
            residual_opt.step()
            static_losses.append(float(static_bce.detach()))
            residual_losses.append(float(loss.detach()))
            contrast_losses.append(float(contrast.detach()))
        metric = _validate(residual, static_w.detach(), static_b.detach(), val, device)
        record = {"epoch": epoch, "optimizer_steps": steps, "presentations": presentations,
            "unique_fit_ids_seen": len(seen), "static_bce": float(np.mean(static_losses)),
            "residual_total": float(np.mean(residual_losses)),
            "supcon": float(np.mean(contrast_losses)), "elapsed_seconds": time.monotonic() - started,
            **metric}
        history.append(record)
        for arm in ("erm", "residual"):
            key = (record[f"{arm}_clean_eer"],
                   (record[f"{arm}_noise_eer"] + record[f"{arm}_fir_eer"]) / 2,
                   -record[f"{arm}_clean_auc"])
            checkpoint = output / f"{arm}_epoch_{epoch:04d}.pt"
            if arm == "erm":
                torch.save({"w": static_w.detach().cpu(), "b": static_b.detach().cpu(),
                            "epoch": epoch, "seed": seed}, checkpoint)
            else:
                torch.save({"model": {k: v.detach().cpu() for k, v in residual.state_dict().items()},
                            "epoch": epoch, "seed": seed}, checkpoint)
            if key < best[arm]:
                best[arm] = key
                (output / f"{arm}_selection.json").write_text(json.dumps({
                    "epoch": epoch, "checkpoint_ref": str(checkpoint.resolve()),
                    "source_select_key": list(key), "optimizer_steps_at_epoch": steps}, indent=2))
        print(json.dumps(record), flush=True)
    with (output / "training_curve.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    summary = {"status": "PASS", "run": str(output.resolve()), "seed": seed, "device": str(device),
        "optimizer_steps_per_arm": steps, "presentations_per_arm": presentations,
        "unique_fit_ids_seen": len(seen), "fit_count": len(fit["ids"]),
        "numerical_failures": numerical_failures, "elapsed_seconds": time.monotonic() - started,
        "fit_cache_id": fit["cache_id"], "select_cache_id": val["cache_id"]}
    (output / "source_training.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run(args.config, args.run_id, args.seed)
