"""Supervised development-only five-fold capacity ladder on frozen embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_
from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "outputs_v2/ssl_aasist"
ITW_LABELS = ROOT / "experiments/target10_selection/manifests/inwild_target10.json"
SEED = 2026
FOLDS = 5
LR = 0.01
EPOCHS = 100
BATCH = 256
L2 = 1e-4
PATIENCE = 10


def write_new(path: Path, doc: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_itw():
    from experiments.multidomain_mechanism.guard_worker import load_context
    select = json.loads((ROOT / "experiments/large_scale_confirmation/manifests/"
                         "in_the_wild_confirmation_select.json").read_text())
    ids = [row["sample_id"] for row in select["records"]]
    if select["count"] != 3178 or len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("fixed ITW target10 selection invalid")
    labels = json.loads(ITW_LABELS.read_text())
    ymap = {row["sample_id"]: row["label"] for row in labels["records"]}
    if labels["count"] != 3178 or set(ymap) != set(ids) or set(ymap.values()) != {0, 1}:
        raise ValueError("target10 selected-only label coverage invalid")
    resources, features, _, provenance = load_context("in_the_wild", set(ids))
    X = np.stack([features[sid][0] for sid in ids]).astype(np.float32)
    y = np.array([ymap[sid] for sid in ids], dtype=np.int64)
    return ids, X, y, resources, provenance, None


def load_wavefake(cache_ref: Path, assignment_ref: Path, labels_ref: Path):
    assignment = json.loads(assignment_ref.read_text())
    label_doc = json.loads(labels_ref.read_text())
    ids = [row["sample_id"] for row in assignment["records"]]
    groups = [row["audio_id"] for row in assignment["records"]]
    ymap = {row["sample_id"]: row["label"] for row in label_doc["records"]}
    if (assignment["dataset_id"] != "wavefake" or len(ids) != len(set(ids)) or
            set(ids) != set(ymap) or set(ymap.values()) != {0, 1}):
        raise ValueError("WaveFake fixed assignment/label coverage invalid")
    cache = FeatureCache(cache_ref)
    if cache.index["num_views"] != 3 or cache.index["feature_dim"] != 160:
        raise ValueError("WaveFake frozen feature shape invalid")
    features = cache.load_by_id()
    if set(features) != set(ids):
        raise ValueError("WaveFake exact cache coverage invalid")
    bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
    resources, _, _ = load_frozen_resources(BASE / "resources", bundle)
    if cache.index["identity"]["source_run_id"] != bundle["source_run_id"] or cache.index["identity"]["checkpoint_ref"] != bundle["checkpoint_ref"]:
        raise ValueError("WaveFake cache uses a different Frozen model")
    X = np.stack([features[sid][0] for sid in ids]).astype(np.float32)
    y = np.array([ymap[sid] for sid in ids], dtype=np.int64)
    return ids, X, y, resources, {"cache_identity": cache.index["identity"]}, groups


def group_stratified_folds(y: np.ndarray, groups: list[str]):
    # Every matched content group contains exactly one real and one generated row.
    by_group = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    if any(len(rows) != 2 or sorted(y[rows].tolist()) != [0, 1] for rows in by_group.values()):
        raise ValueError("WaveFake pairing is not exactly one real/one fake per content group")
    rng = np.random.default_rng(SEED)
    names = sorted(by_group)
    rng.shuffle(names)
    fold_for_group = {name: index % FOLDS for index, name in enumerate(names)}
    for fold in range(FOLDS):
        val = np.array([index for index, group in enumerate(groups) if fold_for_group[group] == fold])
        train = np.array([index for index, group in enumerate(groups) if fold_for_group[group] != fold])
        yield train, val


def stratified_folds(y: np.ndarray):
    """Fixed five-fold allocation without an additional runtime dependency."""
    if set(y.tolist()) != {0, 1}:
        raise ValueError("two classes required")
    buckets = [[] for _ in range(FOLDS)]
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        if len(indices) < FOLDS:
            raise ValueError("fewer than five members in a class")
        np.random.default_rng(SEED + label).shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % FOLDS].append(int(index))
    for bucket in buckets:
        heldout = np.array(sorted(bucket), dtype=np.int64)
        train = np.setdiff1d(np.arange(len(y)), heldout, assume_unique=True)
        yield train, heldout


def inner_stratified_split(indices: np.ndarray, y: np.ndarray, seed: int):
    inner_train, inner_val = [], []
    for label in (0, 1):
        class_indices = np.array([i for i in indices if y[i] == label], dtype=np.int64)
        if len(class_indices) < 2:
            raise ValueError("inner split lacks a class")
        np.random.default_rng(seed + label).shuffle(class_indices)
        n_val = max(1, int(round(.1 * len(class_indices))))
        n_val = min(n_val, len(class_indices) - 1)
        inner_val.extend(class_indices[:n_val].tolist())
        inner_train.extend(class_indices[n_val:].tolist())
    return np.array(sorted(inner_train), dtype=np.int64), np.array(sorted(inner_val), dtype=np.int64)


def inner_group_split(indices: np.ndarray, y: np.ndarray, groups: list[str], seed: int):
    """Keep real/generated content pairs together even during epoch selection."""
    names = sorted({groups[index] for index in indices})
    if len(names) < 10:
        raise ValueError("insufficient paired content groups for inner validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_val = max(1, int(round(.1 * len(names))))
    validation_groups = set(names[:n_val])
    inner_val = np.array([i for i in indices if groups[i] in validation_groups], dtype=np.int64)
    inner_train = np.array([i for i in indices if groups[i] not in validation_groups], dtype=np.int64)
    if set(y[inner_val].tolist()) != {0, 1} or set(y[inner_train].tolist()) != {0, 1}:
        raise ValueError("inner grouped split lost a class")
    return inner_train, inner_val


def fit_arm(arm: str, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
            val_idx: np.ndarray, resources, fold: int, groups: list[str] | None = None):
    inner_train, inner_val = (inner_stratified_split(train_idx, y, SEED + fold)
                              if groups is None else inner_group_split(train_idx, y, groups, SEED + fold))
    torch.manual_seed(SEED + fold)
    torch.set_num_threads(1)
    mean = np.zeros(X.shape[1], np.float32)
    scale = np.ones(X.shape[1], np.float32)
    if arm != "C1_supervised_R":
        mean = X[inner_train].mean(axis=0)
        scale = X[inner_train].std(axis=0)
        scale[scale < 1e-6] = 1.
    transformed = torch.from_numpy(((X - mean) / scale).astype(np.float32))
    labels = torch.from_numpy(y.astype(np.float32))
    if arm == "C1_supervised_R":
        R = torch.nn.Parameter(torch.zeros((8, 8), dtype=torch.float32))
        params = [R]

        def logits(index):
            return apply_adapter(transformed[index], resources.U, R) @ resources.w + resources.b

    elif arm == "C2_linear":
        model = torch.nn.Linear(160, 1)
        params = list(model.parameters())

        def logits(index):
            return model(transformed[index]).squeeze(-1)

    elif arm == "C3_nonlinear":
        model = torch.nn.Sequential(torch.nn.Linear(160, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 1))
        params = list(model.parameters())

        def logits(index):
            return model(transformed[index]).squeeze(-1)

    else:
        raise ValueError(arm)
    optimizer = torch.optim.Adam(params, lr=LR)
    best = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        perm = np.random.default_rng(SEED + 1000 * fold + epoch).permutation(inner_train)
        for begin in range(0, len(perm), BATCH):
            batch = perm[begin:begin + BATCH]
            optimizer.zero_grad(set_to_none=True)
            score = logits(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels[batch])
            loss = loss + L2 * sum(param.square().sum() for param in params)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite supervised training loss")
            loss.backward()
            optimizer.step()
            if arm == "C1_supervised_R":
                project_frobenius_(R, .1)
        with torch.no_grad():
            vscore = logits(inner_val)
            vloss = float(torch.nn.functional.binary_cross_entropy_with_logits(vscore, labels[inner_val]))
        if not math.isfinite(vloss):
            raise FloatingPointError("nonfinite inner-validation loss")
        if vloss < best - 1e-4:
            best, best_epoch, stale = vloss, epoch, 0
            best_state = [param.detach().clone() for param in params]
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("no inner-validation model state")
    with torch.no_grad():
        for param, value in zip(params, best_state):
            param.copy_(value)
        heldout = logits(val_idx).detach().numpy().astype(np.float64)
    if not np.isfinite(heldout).all():
        raise FloatingPointError("nonfinite held-out predictions")
    return heldout, {"best_epoch": best_epoch, "inner_validation_bce": best,
                     "train_count": len(inner_train), "inner_val_count": len(inner_val),
                     "parameter_count": sum(p.numel() for p in params),
                     "R_norm": float(torch.linalg.vector_norm(R)) if arm == "C1_supervised_R" else None}


def metric(scores, labels):
    return binary_metrics(scores.tolist(), labels.astype(int).tolist(), 0.0)


def run(args):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("logs", "predictions", "analysis"):
        (out / sub).mkdir()
    try:
        ids, X, y, resources, provenance, groups = (load_itw() if args.domain == "itw" else
            load_wavefake(args.cache, args.assignment, args.labels))
        if X.shape != (len(ids), 160) or not np.isfinite(X).all():
            raise ValueError("frozen feature shape/finite failure")
        frozen = (X @ resources.w.numpy() + resources.b).astype(np.float64)
        if not np.isfinite(frozen).all():
            raise FloatingPointError("nonfinite Frozen scores")
        folds = list(stratified_folds(y)) if groups is None else list(group_stratified_folds(y, groups))
        if sorted(np.concatenate([val for _, val in folds]).tolist()) != list(range(len(ids))):
            raise ValueError("held-out fold coverage invalid")
        write_new(out / "run_config.json", {"role": "SUPERVISED_DEVELOPMENT_DIAGNOSIS_NOT_TTA",
            "run_id": args.run_id, "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "domain": args.domain, "count": len(ids), "class_counts": np.bincount(y, minlength=2).tolist(),
            "python": sys.version, "torch": torch.__version__, "seed": SEED, "folds": FOLDS,
            "optimizer": "Adam", "lr": LR, "epochs_max": EPOCHS, "batch": BATCH, "l2": L2,
            "inner_val_fraction": .1, "patience": PATIENCE, "min_delta": 1e-4,
            "R_radius": .1, "command": sys.argv, "target90_accessed": False, "final_holdout_accessed": False})
        write_new(out / "provenance.json", provenance)
        write_new(out / "folds.json", {"seed": SEED, "folds": [{"fold": k,
            "held_out_sample_ids": [ids[i] for i in val], "train_count": len(train),
            "held_out_class_counts": np.bincount(y[val], minlength=2).tolist()} for k, (train, val) in enumerate(folds)]})
        scores = {"C0_Frozen": frozen}
        diagnostics = {}
        fold_rows = []
        for arm in ("C1_supervised_R", "C2_linear", "C3_nonlinear"):
            if arm == "C3_nonlinear":
                c0 = metric(scores["C0_Frozen"], y)
                c2 = metric(scores["C2_linear"], y)
                trigger = abs(c2["auroc"] - c0["auroc"]) >= .005 or (
                    1-c0["auroc"] >= .01 and 1-c2["auroc"] >= .01 and
                    abs(c2["auroc"] - c0["auroc"]) < .005)
                if not trigger:
                    diagnostics[arm] = {"status": "NOT_RUN", "reason": "preregistered C3 trigger false"}
                    continue
            pred = np.full(len(ids), np.nan, np.float64)
            train_details = []
            start = time.monotonic()
            for fold, (train, heldout) in enumerate(folds):
                prediction, detail = fit_arm(arm, X, y, train, heldout, resources, fold, groups)
                pred[heldout] = prediction
                detail["fold"] = fold
                train_details.append(detail)
                fold_metric = metric(prediction, y[heldout])
                fold_rows.append({"domain": args.domain, "arm": arm, "fold": fold,
                    "count": len(heldout), "bonafide": int((y[heldout] == 0).sum()),
                    "spoof": int((y[heldout] == 1).sum()), "auc": fold_metric["auroc"],
                    "eer": fold_metric["eer"], **detail})
                print(args.domain, arm, fold, fold_metric["auroc"], fold_metric["eer"], flush=True)
            if not np.isfinite(pred).all():
                raise ValueError("incomplete held-out prediction coverage")
            scores[arm] = pred
            diagnostics[arm] = {"training_folds": train_details, "runtime_seconds": time.monotonic()-start}
        for fold, (_, heldout) in enumerate(folds):
            m = metric(frozen[heldout], y[heldout])
            fold_rows.append({"domain": args.domain, "arm": "C0_Frozen", "fold": fold,
                "count": len(heldout), "bonafide": int((y[heldout] == 0).sum()),
                "spoof": int((y[heldout] == 1).sum()), "auc": m["auroc"], "eer": m["eer"]})
        table = []
        c0 = metric(frozen, y)
        for arm, pred in scores.items():
            m = metric(pred, y)
            table.append({"domain": args.domain, "arm": arm, "count": len(ids),
                "auc": m["auroc"], "eer": m["eer"], "delta_auc": m["auroc"]-c0["auroc"],
                "eer_improvement": c0["eer"]-m["eer"]})
        with (out / "analysis/capacity_table.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
        with (out / "analysis/fold_metrics.csv").open("x", newline="") as stream:
            keys = ["domain","arm","fold","count","bonafide","spoof","auc","eer",
                    "best_epoch","inner_validation_bce","train_count","inner_val_count","parameter_count","R_norm"]
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader(); writer.writerows(fold_rows)
        with (out / "predictions/held_out.jsonl").open("x") as stream:
            for i, sid in enumerate(ids):
                stream.write(json.dumps({"sample_id": sid, "label": int(y[i]),
                    "fold": next(k for k, (_, val) in enumerate(folds) if i in val),
                    "scores": {arm:float(pred[i]) for arm,pred in scores.items()}}, allow_nan=False)+"\n")
        summary = {"status": "PASS", "role": "SUPERVISED_DEVELOPMENT_DIAGNOSIS_NOT_TTA",
                   "domain": args.domain, "count": len(ids), "capacity_table": table,
                   "c3": diagnostics.get("C3_nonlinear"), "target90_accessed": False,
                   "final_holdout_accessed": False}
        write_new(out / "analysis/summary.json", summary)
        write_new(out / "analysis/diagnostics.json", diagnostics)
        (out / "analysis/report.md").write_text("# Supervised development capacity audit\n\n"
            "This is not TTA or final performance. All predictions are held out by fold.\n\n"
            + "| Arm | AUC | EER | ΔAUC | EER improvement |\n|---|---:|---:|---:|---:|\n"
            + "\n".join(f"| {r['arm']} | {r['auc']:.6f} | {r['eer']:.6f} | {r['delta_auc']:+.6f} | {r['eer_improvement']:+.6f} |" for r in table)+"\n")
    except BaseException:
        write_new(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--domain", choices=("itw", "wavefake"), required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--labels", type=Path)
    run(parser.parse_args())
