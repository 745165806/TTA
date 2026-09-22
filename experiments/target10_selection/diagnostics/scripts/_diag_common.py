"""Shared helpers for target10_selection diagnostics.

Imports the private ``_common`` module of the parent experiment (so diagnostics
reuse the *exact* production adaptation/evaluation path) and adds only
diagnostic-only instrumented helpers.  Nothing here mutates core ``src/`` code.
"""
import json
import math
import sys
from pathlib import Path

import torch

# The EP workloads are many small tensor ops on CPU; torch's default OpenMP
# intra-op parallelism causes massive oversubscription when several diagnostic
# workers run together.  Pin to a single thread (single small matmuls are far
# faster single-threaded) — this matches the per-sample production semantics.
torch.set_num_threads(1)

# Make the parent experiment's private ``scripts/`` importable so we reuse its
# production ``_common`` helpers verbatim.
_DIAG_DIR = Path(__file__).resolve().parents[1]          # .../diagnostics/
_EXP_SCRIPTS = _DIAG_DIR.parent / "scripts"              # .../target10_selection/scripts/
if str(_EXP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EXP_SCRIPTS))

from _common import (  # noqa: E402  (re-export the production helpers)
    ROOT, EXP_DIR, RESULTS_DIR, FROZEN_BUNDLE, RESOURCES, CACHE,
    TARGET10, TARGET10_SELECT, TARGET90, BEST_PARAM,
    RHO, GAMMA, LAMBDA_KEEP,
    load_context, load_select_sample_ids, read_target90_records, compute_metrics,
)

from eptta.adaptation.math import apply_adapter, project_frobenius_  # noqa: E402
from eptta.adaptation.objectives import target_objective  # noqa: E402
from eptta.adaptation.regularizers import regularizer  # noqa: E402
from eptta.adaptation.types import EPConfig, TargetViews  # noqa: E402
from eptta.baselines.dispatch import run_method  # noqa: E402
from eptta.evaluation.metrics import binary_metrics  # noqa: E402
from eptta.training.selection import equal_error_rate  # noqa: E402

DIAG_DIR = _DIAG_DIR
DIAG_SCRIPTS = DIAG_DIR / "scripts"
DIAG_CONFIGS = DIAG_DIR / "configs"
DIAG_RESULTS = DIAG_DIR / "results"
DIAG_LOGS = DIAG_DIR / "logs"
DIAG_REPORTS = DIAG_DIR / "reports"

DIAG_SEED = 2026
DIAG_SAMPLE_COUNT = 500

# Historical target10-v0 protocol.  Keep these definitions local: the active
# experiment now uses guarded-v2 with a different grid and selection metrics.
V0_KS = [0, 1, 3, 5, 10]
V0_LRS = [1e-5, 5e-5, 1e-4]


def candidates():
    for k in V0_KS:
        if k == 0:
            yield {"K": 0, "lr": None, "steps": 0}
        else:
            for lr in V0_LRS:
                yield {"K": k, "lr": lr, "steps": k}


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-float(value)))


def binary_entropy(probability):
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(probability * math.log(probability)
             + (1.0 - probability) * math.log(1.0 - probability))


def selection_score(row):
    return (row["entropy"] / math.log(2.0)
            + row["consistency"] + row["stability"]) / 3.0


def best_key(row):
    return (row["selection_score"], -row["K"],
            -(row["lr"] if row["lr"] is not None else 0.0))

# Diagnostic config specs for Diag 2/3 (adaptation effectiveness & consistency).
DIAG_CONFIG_SPECS = [
    {"key": "A_K0", "label": "K=0 (frozen)", "K": 0, "lr": None, "steps": 0},
    {"key": "B_K1_lr1e-5", "label": "K=1 lr=1e-5", "K": 1, "lr": 1e-5, "steps": 1},
    {"key": "C_K5_lr1e-4", "label": "K=5 lr=1e-4", "K": 5, "lr": 1e-4, "steps": 5},
    {"key": "D_K10_lr1e-4", "label": "K=10 lr=1e-4", "K": 10, "lr": 1e-4, "steps": 10},
    {"key": "E_K1_lr0.003", "label": "source-select K=1 lr=0.003", "K": 1, "lr": 0.003, "steps": 1},
]


def ensure_dirs():
    for d in (DIAG_RESULTS, DIAG_LOGS, DIAG_REPORTS, DIAG_CONFIGS):
        d.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path, obj):
    """Write JSON atomically to a unique temp path then rename (no partial files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(__import__("os").getpid()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_config(cand):
    """EPConfig for a candidate dict (K=0 -> lr sentinel, steps=0)."""
    lr = cand["lr"] if cand["lr"] is not None else 1e-5
    return EPConfig(steps=cand["K"], lr=lr, rho=RHO, gamma=GAMMA, lambda_keep=LAMBDA_KEEP)


def load_target10_labels():
    """POST-HOC ONLY: sample_id -> canonical 0/1 label from the raw audit manifest.

    These labels must NEVER feed the original label-free selection.  They are
    read here only to judge, after the fact, whether the unsupervised selection
    metric tracked real performance.
    """
    doc = json.loads(TARGET10.read_text(encoding="utf-8"))
    return {r["sample_id"]: int(r["label"]) for r in doc["records"]}


def score_one_ep(sample_id, z, resources, cfg):
    """Return the adapted original-view score for one sample (score_after logit)."""
    target = TargetViews(sample_id, z, cache_id_for(resources))
    result = run_method("ep_tta", target, resources, cfg, {})
    return float(result["score"]), float(result["score_before"])


def cache_id_for(resources):
    # The production pipeline passes the feature cache's cache_id as the
    # TargetViews.feature_artifact_id; recover it from the bundle-independent
    # constant in _common (CACHE.cache_id is read in the caller instead).
    from _common import CACHE
    return CACHE.cache_id


def spearman(xs, ys):
    """Spearman rank correlation; returns None when a series has no variance."""
    from scipy.stats import spearmanr as _spearmanr
    xs = list(xs)
    ys = list(ys)
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rho, _p = _spearmanr(xs, ys)
    if not math.isfinite(float(rho)):
        return None
    return float(rho)


def pearson(xs, ys):
    from scipy.stats import pearsonr as _pearsonr
    xs = list(xs)
    ys = list(ys)
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    r, _p = _pearsonr(xs, ys)
    if not math.isfinite(float(r)):
        return None
    return float(r)


def run_ep_final(z, resources, cfg):
    """Compact faithful EP loop returning only the final adapted state.

    Mirrors ``run_cache_method("ep_tta", ...)`` exactly (view_variance + margin
    keep, manual SGD, Frobenius projection) but keeps only R_final, the adapted
    views and their logits/probs.  Used by Diag 3 for soft view-consistency.
    """
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=z.dtype, device=z.device, requires_grad=True)
    with torch.enable_grad():
        for _step in range(cfg.steps):
            R = parameter
            objective = target_objective("view_variance", apply_adapter(z, resources.U, R),
                                         resources.w, resources.b)
            keep = regularizer("margin", R, resources, GAMMA)
            loss = objective + LAMBDA_KEEP * keep
            gradient, = torch.autograd.grad(loss, parameter, retain_graph=True)
            with torch.no_grad():
                parameter.add_(gradient, alpha=-cfg.lr)
                project_frobenius_(parameter, cfg.rho)
    R_final = parameter.detach().clone()
    adapted_views = apply_adapter(z, resources.U, R_final)
    logits = adapted_views @ resources.w + resources.b
    return R_final, adapted_views, [float(v) for v in logits.detach().cpu().tolist()]


def run_ep_diagnostic(sample_id, z, resources, cfg):
    """Faithful instrumented replica of ``run_cache_method("ep_tta", ...)``.

    This does NOT modify core code.  It reproduces the exact view_variance +
    margin keep loop (SGD, no momentum, Frobenius projection at rho) while
    additionally recording per-step gradient/update norms and before/after
    objectives so the diagnostics can quantify whether adaptation is effective.
    """
    before = float(z[0] @ resources.w + resources.b)          # frozen logit == score
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=z.dtype, device=z.device, requires_grad=True)

    # "before" objective (R=0 => adapted == Z).
    R0 = torch.zeros((rank, rank), dtype=z.dtype, device=z.device)
    obj0 = float(target_objective("view_variance", apply_adapter(z, resources.U, R0),
                                  resources.w, resources.b))
    keep0 = float(regularizer("margin", R0, resources, GAMMA))
    loss0 = obj0 + LAMBDA_KEEP * keep0

    step_trace = []
    grad_norms = []
    reg_grad_norms = []
    update_norms = []
    completed = 0
    with torch.enable_grad():
        for step in range(cfg.steps):
            R = parameter
            objective = target_objective("view_variance", apply_adapter(z, resources.U, R),
                                         resources.w, resources.b)
            keep = regularizer("margin", R, resources, GAMMA)
            loss = objective + LAMBDA_KEEP * keep
            gradient, = torch.autograd.grad(loss, parameter, retain_graph=True)
            grad_norm = float(torch.linalg.vector_norm(gradient))
            reg_gradient, = torch.autograd.grad(LAMBDA_KEEP * keep, parameter)
            reg_grad_norm = float(torch.linalg.vector_norm(reg_gradient))
            before_param = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(gradient, alpha=-cfg.lr)
                pre_norm = float(torch.linalg.vector_norm(parameter))
                projection_applied = pre_norm > cfg.rho
                project_frobenius_(parameter, cfg.rho)
                post_norm = float(torch.linalg.vector_norm(parameter))
            step_update_norm = float(torch.linalg.vector_norm(parameter.detach() - before_param))
            completed = step + 1
            grad_norms.append(grad_norm)
            reg_grad_norms.append(reg_grad_norm)
            update_norms.append(step_update_norm)
            step_trace.append({
                "step": completed,
                "objective_view_variance": float(objective.detach()),
                "keep_margin": float(keep.detach()),
                "total_loss": float(loss.detach()),
                "grad_norm_total": grad_norm,
                "regularizer_gradient_norm": reg_grad_norm,
                "pre_projection_norm": pre_norm,
                "projection_applied": projection_applied,
                "R_norm_after": post_norm,
                "step_update_norm": step_update_norm,
            })

    R_final = parameter.detach().clone()
    adapted_views = apply_adapter(z, resources.U, R_final)
    logits_after = adapted_views @ resources.w + resources.b          # [3]
    score_after = float(logits_after[0]) if cfg.steps > 0 else before
    adapted_view_logits = [float(v) for v in logits_after.detach().cpu().tolist()]
    adapted_view_probs = [sigmoid(v) for v in adapted_view_logits]
    obj_after = float(target_objective("view_variance", adapted_views, resources.w, resources.b))
    keep_after = float(regularizer("margin", R_final, resources, GAMMA))
    loss_after = obj_after + LAMBDA_KEEP * keep_after

    p_before = sigmoid(before)
    p_after = sigmoid(score_after)
    update_norm_total = float(torch.linalg.vector_norm(R_final))

    return {
        "sample_id": sample_id,
        "steps": completed,
        "score_before": before,
        "score_after": score_after,
        "logit_before": before,               # single-logit head: score == logit
        "logit_after": score_after,
        "prob_before": p_before,
        "prob_after": p_after,
        "score_delta": score_after - before,
        "abs_score_delta": abs(score_after - before),
        "abs_prob_delta": abs(p_after - p_before),
        "prediction_before": int(before > 0.0),
        "prediction_after": int(score_after > 0.0),
        "flip": int((before > 0.0) != (score_after > 0.0)),
        "entropy_before": binary_entropy(p_before),
        "entropy_after": binary_entropy(p_after),
        "entropy_delta": binary_entropy(p_after) - binary_entropy(p_before),
        "objective_before": obj0,
        "objective_after": obj_after,
        "keep_before": keep0,
        "keep_after": keep_after,
        "loss_before": loss0,
        "loss_after": loss_after,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
        "reg_grad_norm_mean": (sum(reg_grad_norms) / len(reg_grad_norms)) if reg_grad_norms else 0.0,
        "reg_grad_norm_max": max(reg_grad_norms) if reg_grad_norms else 0.0,
        "update_norm_total": update_norm_total,
        "update_norm_relative_rho": update_norm_total / RHO,
        "cumulative_update_norm": sum(update_norms),
        "R_norm_final": update_norm_total,
        "adapted_view_logits": adapted_view_logits,
        "adapted_view_probs": adapted_view_probs,
        "step_trace": step_trace,
    }
