#!/usr/bin/env python
"""P0 label-free mechanism diagnosis worker.

Runs the production ``run_method("ep_tta_guarded", ...)`` path for every
(candidate, sample) assigned to this group and writes one JSONL record per
sample.  This worker MUST NOT read target labels: it only consumes the
label-free select manifest, the feature cache, the frozen resources and the
candidate parameters.
"""
import argparse
import json
import math
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

import torch

from eptta.adaptation.math import apply_adapter, view_loss
from scripts.common import (ep_config, load_candidates, load_context, load_select_sample_ids,
                            make_target)

torch.set_num_threads(1)


def _clip_relative(value):
    return max(-1.0, min(1.0, value))


def score_sample(cand, sample_id, features, resources, cache_id):
    target = make_target(sample_id, features, cache_id)
    result = run_method_guarded(target, resources, ep_config(cand))

    score_before = float(result["score_before"])
    score_after = float(result["score"])
    status = result.get("status")
    numeric_fallback = status != "ok"

    z = target.features
    w, b = resources.w, resources.b
    before_views = (z @ w + b).detach().tolist()

    R = result.get("R")
    if R is None:
        R = torch.zeros((resources.U.shape[1], resources.U.shape[1]),
                        dtype=z.dtype, device=z.device)
    adapted = apply_adapter(z, resources.U, R)
    after_views = (adapted @ w + b).detach().tolist()

    loss_before = float(view_loss(z))
    loss_after = float(view_loss(adapted))
    relative = (loss_before - loss_after) / max(loss_before, 1e-12)

    trace = result.get("trace") or []
    guard_activation = sum(bool(row.get("margin_guard_applied")) for row in trace)
    guard_backtracks = sum(int(row.get("margin_guard_backtracks", 0)) for row in trace)
    guard_reverts = sum(bool(row.get("margin_guard_reverted")) for row in trace)
    guard_steps = len(trace)

    return {
        "sample_id": sample_id,
        "K": cand["K"],
        "lr": cand["lr"],
        "steps": cand["steps"],
        "score_before": score_before,
        "score_after": score_after,
        "delta_score": score_after - score_before,
        "view0_score_before": before_views[0],
        "view1_score_before": before_views[1],
        "view2_score_before": before_views[2],
        "view0_score_after": after_views[0],
        "view1_score_after": after_views[1],
        "view2_score_after": after_views[2],
        "view_loss_before": loss_before,
        "view_loss_after": loss_after,
        "view_reduction": relative,
        "view_reduction_clipped": _clip_relative(relative),
        "R_norm": float(torch.linalg.vector_norm(R)),
        "guard_activation_count": guard_activation,
        "guard_backtracks": guard_backtracks,
        "guard_reverts": guard_reverts,
        "guard_steps": guard_steps,
        "numeric_fallback": numeric_fallback,
    }


def run_method_guarded(target, resources, cfg):
    from eptta.baselines.dispatch import run_method
    return run_method("ep_tta_guarded", target, resources, cfg, {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--num-groups", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.group < 0 or args.group >= args.num_groups:
        raise SystemExit("group must be in [0, num_groups)")

    candidates = load_candidates()
    sample_ids = load_select_sample_ids()
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    _bundle, resources, _meta, cache, features, _threshold = load_context()

    assigned = [c for i, c in enumerate(candidates) if i % args.num_groups == args.group]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / ("p0_group_%d.jsonl" % args.group)
    if out_path.exists():
        raise SystemExit("refusing to overwrite existing output: %s" % out_path)

    total = len(assigned) * len(sample_ids)
    written = 0
    with out_path.open("w", encoding="utf-8") as stream:
        for cand in assigned:
            for sample_id in sample_ids:
                record = score_sample(cand, sample_id, features, resources, cache.cache_id)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                written += 1
            print("group %d: done candidate K=%s lr=%s (%d/%d records)" %
                  (args.group, cand["K"], cand["lr"], written, total), flush=True)

    if written != total:
        raise SystemExit("incomplete output: wrote %d of %d records" % (written, total))
    print("group %d: wrote %s (%d records)" % (args.group, out_path, written))


if __name__ == "__main__":
    main()
