"""Label-free H-UA1 score worker. Never import target development labels here."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
import torch

from experiments.head_capacity_geometry.resources import ROOT, load_domain
from experiments.head_tta.mixture import ALPHAS, adapt_buffer, make_source_geometry


HERE = Path(__file__).resolve().parent
BATCH = 256


def write_new(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(run_id: str, smoke: bool):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "scores", "diagnostics", "analysis", "manifests"):
        (out / part).mkdir()
    try:
        command = [sys.executable, *sys.argv]
        write_new(out / "run_config.json", {
            "run_id": run_id, "role": "LABEL_FREE_DEVELOPMENT_SCORE_WORKER",
            "smoke": smoke, "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
            "seed": 2026, "domains": ["itw", "wavefake"],
            "buffer_size": BATCH, "order": "fixed_manifest_order",
            "alphas": ALPHAS, "geometry_contract": "HEAD_ADAPTATION_CONTRACT.md",
            "command": command, "target_labels_loaded": False,
            "target90_labels_accessed": False, "target90_metrics_accessed": False,
            "final_heldout_metrics_accessed": False})
        provenance = {}
        completion = {}
        for domain in ("itw", "wavefake"):
            ids, views, _, resources, info = load_domain(domain)
            if smoke:
                ids, views = ids[:32], views[:32]
            provenance[domain] = info
            source = make_source_geometry(resources)
            write_new(out / "manifests" / f"{domain}_order.json", {
                "domain": domain, "sample_ids": ids, "count": len(ids),
                "selection": info["selection_ref"], "smoke": smoke})
            observed = []
            score_path = out / "scores" / f"{domain}.jsonl"
            diag_path = out / "diagnostics" / f"{domain}_buffers.jsonl"
            with score_path.open("x", encoding="utf-8") as scores_file, \
                    diag_path.open("x", encoding="utf-8") as diag_file:
                for begin in range(0, len(ids), BATCH):
                    batch_ids = ids[begin:begin+BATCH]
                    start = time.perf_counter()
                    adapted = adapt_buffer(views[begin:begin+BATCH], source)
                    elapsed = time.perf_counter()-start
                    frozen = adapted["frozen"]
                    candidate = adapted["scores"]
                    for i, sid in enumerate(batch_ids):
                        row = {"sample_id": sid, "domain": domain,
                               "buffer_index": begin//BATCH, "score_frozen": float(frozen[i]),
                               "q_spoof": float(adapted["q_spoof"][i]),
                               "reliability": float(adapted["reliability"][i]),
                               "numeric_status": "ok"}
                        for alpha in ALPHAS:
                            row[f"score_alpha_{alpha}"] = float(candidate[alpha][i])
                        scores_file.write(json.dumps(row, allow_nan=False)+"\n")
                        observed.append(sid)
                    delta = candidate[.5]-frozen
                    diag = adapted["diagnostic"] | {
                        "domain": domain, "buffer_index": begin//BATCH,
                        "sample_ids": batch_ids, "runtime_seconds": elapsed,
                        "mean_abs_score_delta_alpha_0.5": float(np.mean(np.abs(delta))),
                        "max_abs_score_delta_alpha_0.5": float(np.max(np.abs(delta)))}
                    diag_file.write(json.dumps(diag, allow_nan=False)+"\n")
            if observed != ids or len(set(observed)) != len(ids):
                raise ValueError(f"{domain}: score ID coverage/order failure")
            source_frozen = views[:, 0, :].astype(np.float64) @ source.w + source.b
            max_parity = float(np.max(np.abs(source_frozen-np.array([
                json.loads(line)["score_frozen"] for line in score_path.open()]))))
            if max_parity > 1e-6:
                raise ValueError(f"{domain}: source frozen score parity failure")
            completion[domain] = {"count": len(ids), "unique_ids": len(set(ids)),
                                  "buffer_count": (len(ids)+BATCH-1)//BATCH,
                                  "frozen_parity_max_difference": max_parity,
                                  "numeric_failures": 0}
            print(f"{domain}: {len(ids)} scores, {completion[domain]['buffer_count']} buffers, "
                  f"Frozen parity {max_parity:.3g}", flush=True)
        write_new(out / "provenance.json", provenance)
        write_new(out / "score_completion.json", {
            "complete": True, "smoke": smoke, "domains": completion,
            "target_labels_loaded_in_worker": False})
    except BaseException as exc:
        write_new(out / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run(args.run_id, args.smoke))
