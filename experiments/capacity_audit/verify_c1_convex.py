"""Check whether the original supervised bounded-R C1 was optimization limited."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from experiments.capacity_audit.run_ladder import HERE, load_itw, load_wavefake, metric, write_new


def solve_fold(X, y, train, heldout, resources):
    U = resources.U.numpy().astype(np.float64)
    w = resources.w.numpy().astype(np.float64)
    v = U.T @ w
    vnorm = np.linalg.norm(v)
    if not np.isfinite(vnorm) or vnorm <= 0:
        raise ValueError("source classifier has no projected direction")
    A = X.astype(np.float64) @ U
    offset = X.astype(np.float64) @ w + resources.b
    radius = .1 * vnorm
    train_A, train_offset, train_y = A[train], offset[train], y[train].astype(np.float64)

    def loss_grad(q):
        scores = train_offset + train_A @ q
        loss = np.logaddexp(0, scores).mean() - (train_y * scores).mean()
        loss += 1e-4 * np.dot(q, q) / (vnorm * vnorm)
        residual = expit(scores) - train_y
        gradient = train_A.T @ residual / len(train) + 2e-4 * q / (vnorm * vnorm)
        return float(loss), gradient

    constraint = {"type": "ineq", "fun": lambda q: radius*radius - np.dot(q, q),
                  "jac": lambda q: -2*q}
    result = minimize(loss_grad, np.zeros(8), method="SLSQP", jac=True,
                      constraints=[constraint], options={"maxiter": 500, "ftol": 1e-10})
    if (not result.success or not np.isfinite(result.x).all() or
            np.linalg.norm(result.x) > radius + 1e-7):
        raise RuntimeError(f"bounded convex C1 solver failed: {result.message}, norm={np.linalg.norm(result.x)}")
    predictions = offset[heldout] + A[heldout] @ result.x
    if not np.isfinite(predictions).all():
        raise FloatingPointError("nonfinite optimized C1 held-out score")
    return predictions, {"success": bool(result.success), "iterations": int(result.nit),
                         "training_objective": float(result.fun), "q_norm": float(np.linalg.norm(result.x)),
                         "radius": float(radius), "source_v_norm": float(vnorm)}


def run(args):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment required")
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "predictions", "analysis"):
        (out / part).mkdir()
    try:
        ids, X, y, resources, provenance, _ = load_itw() if args.domain == "itw" else load_wavefake(
            args.cache, args.assignment, args.labels)
        original = HERE / "results" / (args.domain + "_capacity_20260927a")
        fold_doc = json.loads((original / "folds.json").read_text())
        id_index = {sample_id: i for i, sample_id in enumerate(ids)}
        if len(id_index) != len(ids) or len(fold_doc["folds"]) != 5:
            raise ValueError("original held-out fold assignment invalid")
        write_new(out / "run_config.json", {"role": "SUPERVISED_C1_OPTIMIZATION_CHECK_NOT_TTA",
            "domain": args.domain, "command": sys.argv, "scipy": scipy.__version__,
            "optimizer": "SLSQP", "maxiter": 500, "ftol": 1e-10, "R_radius": .1,
            "loss": "BCE_plus_1e-4_minimum_R_norm_squared", "outer_folds_ref": str(original / "folds.json"),
            "target90_accessed": False, "final_holdout_accessed": False})
        write_new(out / "provenance.json", provenance)
        scores = np.full(len(ids), np.nan, np.float64)
        details = []
        for fold in fold_doc["folds"]:
            heldout = np.array([id_index[sid] for sid in fold["held_out_sample_ids"]], dtype=np.int64)
            train = np.setdiff1d(np.arange(len(ids)), heldout, assume_unique=True)
            prediction, detail = solve_fold(X, y, train, heldout, resources)
            scores[heldout] = prediction
            m = metric(prediction, y[heldout])
            details.append({"fold": fold["fold"], "count": len(heldout),
                "auc": m["auroc"], "eer": m["eer"], **detail})
        if not np.isfinite(scores).all():
            raise ValueError("optimized C1 fold coverage incomplete")
        frozen = (X @ resources.w.numpy() + resources.b).astype(np.float64)
        c0 = metric(frozen, y)
        c1 = metric(scores, y)
        table = {"domain": args.domain, "count": len(ids), "frozen_auc": c0["auroc"],
            "frozen_eer": c0["eer"], "optimized_C1_auc": c1["auroc"],
            "optimized_C1_eer": c1["eer"], "delta_auc": c1["auroc"]-c0["auroc"],
            "eer_improvement": c0["eer"]-c1["eer"]}
        with (out / "analysis/fold_metrics.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader(); writer.writerows(details)
        with (out / "predictions/held_out.jsonl").open("x") as stream:
            for sample_id, label, score in zip(ids, y, scores):
                stream.write(json.dumps({"sample_id": sample_id, "label": int(label),
                                         "optimized_C1_score": float(score)}) + "\n")
        write_new(out / "analysis/summary.json", {"status": "PASS", "table": table,
            "folds": details, "role": "SUPERVISED_C1_OPTIMIZATION_CHECK_NOT_TTA"})
        print(table, flush=True)
    except BaseException:
        write_new(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("itw", "wavefake"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--labels", type=Path)
    run(parser.parse_args())
