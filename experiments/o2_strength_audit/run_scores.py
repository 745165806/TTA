"""One-step, label-free O2 head-update score worker for a fixed mechanism test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from experiments.head_capacity_geometry.resources import BASE, ROOT, load_domain, select_rows
from experiments.task_objective_discovery.objectives import source_geometry


HERE = Path(__file__).resolve().parent
ITW_CACHE = ROOT / "experiments/gradient_alignment_audit/target10_only_cache"
ARMS = ("Frozen", "O2-raw", "O2-normalized-small",
        "O2-normalized-medium", "O2-normalized-large")
DELTAS = {"O2-normalized-small": .01, "O2-normalized-medium": .03,
          "O2-normalized-large": .10}
BATCH = 128
EPS = 1e-12


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_itw(resources):
    ids, _, _ = select_rows("itw")
    bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
    cache = FeatureCache(ITW_CACHE)
    identity = cache.index["identity"]
    if (cache.index["sample_count"] != 3178 or cache.index["num_views"] != 3 or
            cache.index["feature_dim"] != 160 or identity["dataset_id"] != "in_the_wild" or
            identity["source_run_id"] != bundle["source_run_id"] or
            identity["checkpoint_ref"] != bundle["checkpoint_ref"] or
            identity["split_role"] != "select"):
        raise ValueError("isolated target10 cache/source identity mismatch")
    values = cache.load_by_id()
    if set(values) != set(ids) or len(values) != len(ids):
        raise ValueError("ITW exact selected-only feature ID coverage mismatch")
    views = np.stack([values[sid] for sid in ids])
    if views.shape != (3178, 3, 160) or views.dtype != np.float32 or \
            not np.isfinite(views).all():
        raise ValueError("ITW isolated feature shape/dtype/finite failure")
    return ids, views, {"cache_ref": str(ITW_CACHE.resolve()),
                         "cache_id": cache.cache_id, "selected_only_count": len(ids)}


def one_buffer(views, resources, sigma_s):
    """All five scores from the same one-step O2 gradient at the source head."""
    z = torch.from_numpy(np.asarray(views, dtype=np.float32))
    w = resources.w.detach().clone().requires_grad_(True)
    b = float(resources.b)
    logits = z @ w + b
    o2 = (((logits - logits.mean(dim=1, keepdim=True))/sigma_s).square()).mean()
    gradient, = torch.autograd.grad(o2, w)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("nonfinite O2 head gradient")
    source_norm = float(torch.linalg.vector_norm(resources.w))
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    if source_norm <= 0 or not math.isfinite(gradient_norm):
        raise FloatingPointError("invalid source head/O2 gradient norm")
    raw_eta = .01*source_norm
    weights = {"Frozen": resources.w.detach(), "O2-raw":
               resources.w.detach()-raw_eta*gradient.detach()}
    for name, delta in DELTAS.items():
        weights[name] = (resources.w.detach() -
                         (delta*source_norm/(gradient_norm+EPS))*gradient.detach())
    original = z[:, 0, :].double()
    source_score = original @ resources.w.double() + b
    scores = {}
    diagnostics = {}
    for arm in ARMS:
        head = weights[arm]
        score = original @ head.double() + b
        step = head-resources.w
        step_norm = float(torch.linalg.vector_norm(step))
        cosine = float(torch.dot(head, resources.w)/(
            torch.linalg.vector_norm(head)*torch.linalg.vector_norm(resources.w)))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        difference = score-source_score
        if not bool(torch.isfinite(score).all()) or not math.isfinite(angle):
            raise FloatingPointError("nonfinite O2 candidate head score/angle")
        scores[arm] = score.numpy()
        diagnostics[arm] = {"objective_before": float(o2.detach()),
            "gradient_norm": gradient_norm, "source_head_norm": source_norm,
            "raw_eta": raw_eta if arm == "O2-raw" else None,
            "normalized_delta": DELTAS.get(arm), "effective_step_norm": step_norm,
            "head_angle_degrees": angle,
            "mean_abs_score_delta": float(difference.abs().mean()),
            "mean_signed_score_delta": float(difference.mean()),
            "max_abs_score_delta": float(difference.abs().max()),
            "numeric_status": "ok"}
    return scores, diagnostics


def run(run_id):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    torch.set_num_threads(1)
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    try:
        wave_ids, wave_values, _, resources, wave_info = load_domain("wavefake")
        itw_ids, itw_values, itw_info = load_itw(resources)
        geometry = source_geometry(resources)
        data = {"itw": (itw_ids, itw_values), "wavefake": (wave_ids, wave_values)}
        diag_rows = []
        coverage = {}
        for domain, (ids, values) in data.items():
            observed = []
            score_path = out / "scores" / f"{domain}.jsonl"
            with score_path.open("x", encoding="utf-8") as stream:
                for begin in range(0, len(ids), BATCH):
                    start = time.perf_counter()
                    part_ids = ids[begin:begin+BATCH]
                    scores, diagnostics = one_buffer(values[begin:begin+BATCH],
                                                      resources, geometry["sigma_s"])
                    elapsed = time.perf_counter()-start
                    for position, sid in enumerate(part_ids):
                        row = {"sample_id": sid, "domain": domain,
                               "buffer_index": begin//BATCH,
                               "scores": {arm: float(scores[arm][position]) for arm in ARMS}}
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                        observed.append(sid)
                    for arm in ARMS:
                        diag_rows.append({"domain": domain, "buffer_index": begin//BATCH,
                            "buffer_size": len(part_ids), "arm": arm,
                            "runtime_seconds_for_buffer": elapsed, **diagnostics[arm]})
            if observed != ids or len(set(observed)) != len(ids):
                raise ValueError(f"{domain}: score ID/order/unique coverage failure")
            coverage[domain] = {"count": len(ids), "unique_ids": len(set(ids)),
                                "buffer_count": (len(ids)+BATCH-1)//BATCH,
                                "numeric_failures": 0}
            print(f"{domain}: {len(ids)} scores, {coverage[domain]['buffer_count']} reset buffers", flush=True)
        with (out / "diagnostics.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diag_rows[0]),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(diag_rows)
        write_new(out / "score_completion.json", {
            "complete": True, "role": "LABEL_FREE_ONE_STEP_O2_HEAD_SCORE_WORKER",
            "branch": subprocess.check_output(["git", "branch", "--show-current"],
                      cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=ROOT, text=True).strip(),
            "command": [sys.executable, *sys.argv], "buffer_size": BATCH,
            "order": "fixed_manifest", "arms": ARMS, "normalized_deltas": DELTAS,
            "raw_eta_rule": "0.01*source_head_norm", "bias_update": False,
            "domains": coverage, "itw_cache": itw_info, "wavefake_cache": wave_info,
            "target_labels_read": False, "target90_labels_or_metrics_read": False})
        return coverage
    except BaseException:
        write_new(out / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    print(json.dumps(run(parser.parse_args().run_id), indent=2))
