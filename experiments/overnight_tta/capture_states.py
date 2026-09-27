"""Replay saved score runs to retain final target states and arm timings."""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from adapt_eval import _load_selected, selected_features, source_prototypes, t3a_batch_scores, target_adapt
from experiments.distribution_conditioned_head.train import load_source


def median_seconds(call, repeats=20):
    call()
    times = []
    for _ in range(repeats):
        begin = time.perf_counter()
        call()
        times.append(time.perf_counter() - begin)
    return statistics.median(times)


def run(config_ref, run_id, train_run_id, seed):
    config = json.loads(Path(config_ref).read_text())
    torch.set_num_threads(2)
    bundle, w, b, model, erm, _, _ = _load_selected(train_run_id, config["bundle_ref"])
    output = Path(__file__).parent / "results" / run_id
    if not (output / "metrics.csv").exists() or (output / "state_capture.json").exists():
        raise ValueError("scored run missing or state capture already exists")
    fit = load_source(config, "fit", bundle)
    prototypes = source_prototypes(model, fit)
    lr = json.loads((output / "target_lr_selection.json").read_text())["lr"]
    detector = torch.load(Path(config["bundle_ref"]).parent / bundle["detector_state_ref"],
                          map_location="cpu", weights_only=True)
    native_w = detector["model_state"]["out_layer.weight"].float()
    native_b = detector["model_state"]["out_layer.bias"].float()
    summary = {}
    for domain in ("itw_target10", "wavefake_dev"):
        selection = json.loads(Path(config["itw_select" if domain == "itw_target10" else
                                           "wavefake_select"]).read_text())
        if any("label" in row for row in selection["records"]):
            raise ValueError("label in target selection")
        ids = [r["sample_id"] for r in selection["records"]]
        cache = config["itw_cache" if domain == "itw_target10" else "wavefake_cache"]
        views, _ = selected_features(cache, ids, bundle)
        adapted, info = target_adapt(model, views, prototypes, config, lr, seed)
        z = views[:, 0]
        with torch.inference_mode():
            replay = adapted(z).numpy()
        with (output / f"{domain}_scores.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        expected = np.array([float(row["residual_on"]) for row in saved])
        if [row["sample_id"] for row in saved] != ids or not np.array_equal(replay, expected):
            raise AssertionError("final target state does not reproduce durable scores")
        state_ref = output / f"{domain}_target_adapter.pt"
        with state_ref.open("xb") as stream:
            torch.save({"model": adapted.state_dict(), "seed": seed, "target_lr": lr,
                        "source_run": train_run_id, "domain": domain}, stream)
        with torch.inference_mode():
            arm_times = {
                "frozen": median_seconds(lambda: z @ w + b),
                "erm": median_seconds(lambda: z @ erm["w"] + erm["b"]),
                "residual_off": median_seconds(lambda: model(z)),
                "residual_on": median_seconds(lambda: adapted(z)),
                "t3a_batch_template_and_score": median_seconds(
                    lambda: t3a_batch_scores(views, native_w, native_b, config["t3a_k"]))}
        summary[domain] = {"score_parity": "exact", "count": len(ids),
                           "state_ref": str(state_ref), "replay_updates": info["updates"],
                           "median_inference_seconds_20_repeats": arm_times}
    (output / "state_capture.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "run": str(output), "domains": summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.run_id, args.train_run_id, args.seed)
