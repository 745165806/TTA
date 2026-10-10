"""Five-fold supervised head geometry; never a TTA worker."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from eptta.evaluation.metrics import binary_metrics
from experiments.capacity_audit.run_ladder import inner_group_split, inner_stratified_split
from experiments.head_capacity_geometry.resources import ROOT, load_domain
from experiments.head_capacity_geometry.supervised_labels import load_labels


HERE = Path(__file__).resolve().parent
ARMS = ("Frozen", "H0_bias", "H1_scale_bias", "H2_direction", "H3_full_linear",
        "H4_nonlinear_upper_bound")


def write_new(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metrics(scores, labels):
    return binary_metrics(scores.tolist(), labels.astype(int).tolist(), 0.)


def load_saved_folds_and_upper(domain, ids, labels):
    prior = ROOT / "experiments/capacity_audit/results" / (domain + "_capacity_20260927a")
    folds = json.loads((prior / "folds.json").read_text())
    index = {sid: i for i, sid in enumerate(ids)}
    heldout = []
    for fold in folds["folds"]:
        positions = [index[sid] for sid in fold["held_out_sample_ids"]]
        if len(positions) != len(set(positions)) or np.bincount(labels[positions], minlength=2).min() == 0:
            raise ValueError("invalid outer fold class coverage")
        heldout.append(np.asarray(positions, dtype=np.int64))
    if sorted(np.concatenate(heldout).tolist()) != list(range(len(ids))):
        raise ValueError("outer fold coverage mismatch")
    saved = {}
    with (prior / "predictions/held_out.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in index or row["label"] != int(labels[index[sid]]) or sid in saved:
                raise ValueError("prior capacity prediction coverage/label mismatch")
            saved[sid] = row["scores"]
    if set(saved) != set(ids):
        raise ValueError("prior capacity scores incomplete")
    c2 = np.array([saved[sid]["C2_linear"] for sid in ids], dtype=np.float64)
    c3 = np.array([saved[sid]["C3_nonlinear"] for sid in ids], dtype=np.float64)
    return folds, heldout, c2, c3


def fit_h0_h1(source_score, y, train):
    base = source_score[train]
    truth = y[train].astype(np.float64)

    def objective_h0(x):
        score = base + x[0]
        loss = np.logaddexp(0, score).mean() - np.mean(truth * score) + 1e-4 * x[0]**2
        grad = np.mean(expit(score) - truth) + 2e-4*x[0]
        return float(loss), np.array([grad])

    h0 = minimize(objective_h0, np.array([0.]), jac=True, method="L-BFGS-B",
                  options={"maxiter": 500, "ftol": 1e-10})
    if not h0.success:
        raise RuntimeError(f"H0 convex fit failed: {h0.message}")

    def objective_h1(x):
        a, offset = x
        score = a*base + offset
        residual = expit(score) - truth
        loss = np.logaddexp(0, score).mean() - np.mean(truth*score)
        loss += 1e-4*((a-1)**2 + offset**2)
        grad = np.array([np.mean(residual*base)+2e-4*(a-1),
                         np.mean(residual)+2e-4*offset])
        return float(loss), grad

    h1 = minimize(objective_h1, np.array([1., 0.]), jac=True, method="L-BFGS-B",
                  bounds=[(1e-6, None), (None, None)],
                  options={"maxiter": 500, "ftol": 1e-10})
    if not h1.success or h1.x[0] <= 0:
        raise RuntimeError(f"H1 positive-affine fit failed: {h1.message}")
    return float(h0.x[0]), float(h1.x[0]), float(h1.x[1])


def fit_h2_h3(arm, X, y, train, validation, groups, source_w, source_b, fold):
    inner_train, inner_val = (inner_stratified_split(train, y, 2026 + fold)
                              if groups is None else inner_group_split(train, y, groups, 2026 + fold))
    torch.manual_seed(2026 + fold)
    torch.set_num_threads(1)
    labels = torch.from_numpy(y.astype(np.float32))
    if arm == "H3_full_linear":
        mean = X[inner_train].mean(axis=0)
        scale = X[inner_train].std(axis=0)
        scale[scale < 1e-6] = 1.
        features = torch.from_numpy(((X - mean)/scale).astype(np.float32))
        model = torch.nn.Linear(160, 1)
        parameters = list(model.parameters())

        def score(index):
            return model(features[index]).squeeze(-1)

    elif arm == "H2_direction":
        features = torch.from_numpy(X.astype(np.float32))
        direction = torch.nn.Parameter(torch.from_numpy(source_w.copy()))
        parameters = [direction]
        source_norm = float(np.linalg.norm(source_w))

        def score(index):
            w = source_norm * direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
            return features[index] @ w + source_b

    else:
        raise ValueError(arm)
    optimizer = torch.optim.Adam(parameters, lr=.01)
    best, best_epoch, best_state, stale = math.inf, 0, None, 0
    for epoch in range(1, 101):
        perm = np.random.default_rng(2026 + 1000*fold + epoch).permutation(inner_train)
        for begin in range(0, len(perm), 256):
            indices = perm[begin:begin+256]
            optimizer.zero_grad(set_to_none=True)
            logits = score(indices)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
            if arm == "H3_full_linear":
                loss = loss + 1e-4 * sum(parameter.square().sum() for parameter in parameters)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite head training objective")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
                score(inner_val), labels[inner_val]))
        if not math.isfinite(val_loss):
            raise FloatingPointError("nonfinite inner head validation")
        if val_loss < best - 1e-4:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = [parameter.detach().clone() for parameter in parameters]
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("head training found no valid epoch")
    with torch.no_grad():
        for parameter, value in zip(parameters, best_state):
            parameter.copy_(value)
        predictions = score(validation).numpy().astype(np.float64)
        if arm == "H3_full_linear":
            trained_weight = model.weight.detach().numpy()[0]
            effective_w = (trained_weight / scale).astype(np.float64)
            effective_b = float(model.bias.detach().numpy()[0] - np.dot(effective_w, mean))
        else:
            effective_w = (source_norm * direction.detach().numpy() /
                           np.linalg.norm(direction.detach().numpy())).astype(np.float64)
            effective_b = float(source_b)
    if not np.isfinite(predictions).all() or not np.isfinite(effective_w).all():
        raise FloatingPointError("nonfinite held-out head predictions")
    return predictions, effective_w, effective_b, {"best_epoch": best_epoch,
        "inner_validation_bce": best, "inner_train": len(inner_train), "inner_val": len(inner_val)}


def run(domain, run_id):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "scores", "diagnostics", "analysis"):
        (out / part).mkdir()
    try:
        ids, views, groups, resources, provenance = load_domain(domain)
        y = load_labels(domain, ids)
        X = views[:, 0, :]
        source_w = resources.w.numpy().astype(np.float32)
        source_b = float(resources.b)
        source_score = (X @ source_w + source_b).astype(np.float64)
        folds, heldout, c2_prior, c3_prior = load_saved_folds_and_upper(domain, ids, y)
        write_new(out / "run_config.json", {"run_id":run_id, "role":"SUPERVISED_DEVELOPMENT_HEAD_GEOMETRY_NOT_TTA",
            "domain":domain, "count":len(ids), "branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "python":sys.version, "torch":torch.__version__, "seed":2026, "outer_folds":5,
            "arms":ARMS, "command":sys.argv, "target90_labels_accessed":False,
            "target90_metrics_accessed":False, "final_heldout_metrics_accessed":False})
        write_new(out / "provenance.json", provenance)
        write_new(out / "folds.json", folds)
        scores = {arm: np.full(len(ids), np.nan, np.float64) for arm in ARMS}
        scores["Frozen"] = source_score
        scores["H4_nonlinear_upper_bound"] = c3_prior
        head_weights = []
        for index, val in enumerate(heldout):
            train = np.setdiff1d(np.arange(len(ids)), val, assume_unique=True)
            offset, a, b = fit_h0_h1(source_score, y, train)
            scores["H0_bias"][val] = source_score[val] + offset
            scores["H1_scale_bias"][val] = a*source_score[val] + b
            fold_record = {"fold":index, "H0_bias_delta":offset,
                           "H1_positive_scale":a, "H1_offset":b}
            for arm in ("H2_direction", "H3_full_linear"):
                prediction, w, bias, details = fit_h2_h3(arm, X, y, train, val, groups,
                                                          source_w, source_b, index)
                scores[arm][val] = prediction
                fold_record[arm] = {"w":w.tolist(), "b":bias, **details}
            head_weights.append(fold_record)
        if any(not np.isfinite(score).all() for score in scores.values()):
            raise ValueError("incomplete/nonfinite held-out head scores")
        for val in heldout:
            reference = metrics(source_score[val], y[val])
            for arm in ("H0_bias", "H1_scale_bias"):
                current = metrics(scores[arm][val], y[val])
                if (abs(current["auroc"] - reference["auroc"]) > 1e-12 or
                        abs(current["eer"] - reference["eer"]) > 1e-12):
                    raise ValueError("within-fold positive affine ranking invariance failure")
        parity = float(np.max(np.abs(scores["H3_full_linear"] - c2_prior)))
        if parity > 1e-5:
            raise ValueError(f"H3 prior C2 held-out score parity failure: {parity}")
        write_new(out / "diagnostics/head_weights.json", {"source_w":source_w.tolist(),
            "source_b":source_b, "folds":head_weights, "H3_prior_C2_max_score_difference":parity})
        with (out / "scores" / (domain + ".jsonl")).open("x") as stream:
            fold_for = {int(position):index for index, val in enumerate(heldout) for position in val}
            for i, sid in enumerate(ids):
                stream.write(json.dumps({"sample_id":sid, "domain":domain,
                    "outer_fold":fold_for[i], "scores":{arm:float(values[i]) for arm,values in scores.items()}})+"\n")
        rows, fold_rows = [], []
        frozen_auc = metrics(source_score, y)["auroc"]
        for arm in ARMS:
            m = metrics(scores[arm], y)
            rows.append({"domain":domain, "arm":arm, "count":len(ids), "auc":m["auroc"],
                         "eer":m["eer"], "delta_auc":m["auroc"]-frozen_auc})
            for fold, val in enumerate(heldout):
                fm = metrics(scores[arm][val], y[val])
                fold_rows.append({"domain":domain, "arm":arm, "fold":fold,
                                  "count":len(val), "auc":fm["auroc"], "eer":fm["eer"]})
        for path, table in ((out/"analysis/metrics.csv", rows),
                            (out/"analysis/fold_metrics.csv", fold_rows)):
            with path.open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader(); writer.writerows(table)
        write_new(out / "analysis/summary.json", {"status":"PASS", "role":"SUPERVISED_DEVELOPMENT_ONLY",
            "domain":domain, "count":len(ids), "metrics":rows,
            "H3_prior_C2_max_score_difference":parity,
            "H0_H1_within_fold_ranking_invariant":True,
            "target90_labels_accessed":False,"target90_metrics_accessed":False})
        (out / "analysis/report.md").write_text("# Supervised head geometry: "+domain+"\n\n"
            "Five-fold held-out development diagnosis, not TTA.\n\n"
            +"| Head | AUC | EER | ΔAUC |\n|---|---:|---:|---:|\n"
            +"\n".join(f"| {r['arm']} | {r['auc']:.6f} | {r['eer']:.6f} | {r['delta_auc']:+.6f} |" for r in rows)+"\n")
        print(domain, rows, "H3 parity", parity, flush=True)
    except BaseException:
        write_new(out / "failure.json", {"status":"FAIL", "traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("itw", "wavefake"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.domain, args.run_id)
