"""Development-only gradient alignment of existing losses at the frozen linear head.

No optimizer step is taken. Target labels enter only supervised_reference().
O1/O2/O3 are the existing losses lifted from R to (w,b) at R=0, with their
source anchor geometry and frozen O3 reliability held fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from eptta.adaptation.math import view_loss
from eptta.adaptation.objectives import (calibrated_logits, calibrated_pseudo_bce,
                                         mean_view_entropy)
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from experiments.head_capacity_geometry.resources import BASE, ROOT, load_domain, select_rows
from experiments.head_capacity_geometry.supervised_labels import load_labels
from experiments.task_objective_discovery.objectives import (frozen_reliability,
    objective_terms, soft_affinity, source_geometry)


HERE = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments/task_objective_discovery/objective_config.json"
REFERENCE_SCORES = (ROOT.parent / "exp-head-tta/experiments/head_tta/results/"
                    "hua1_development_20260927a/scores/itw.jsonl")
ARMS = ("ENT", "PL", "O1", "O2", "O3")
CHUNK = 128
EPS = 1e-12
ZERO = 1e-8


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_itw_selected_rows(resources, selected_cache_ref):
    """Require an independently materialized target10-only cache."""
    ids, _, _ = select_rows("itw")
    if Path(selected_cache_ref).resolve() == (BASE / "cache-target-in_the_wild").resolve():
        raise ValueError("shared target_test cache is forbidden for this audit")
    bundle, *_ = verify_frozen_export(BASE / "frozen/bundle.json")
    cache = FeatureCache(selected_cache_ref)
    identity = cache.index["identity"]
    if (identity["dataset_id"] != "in_the_wild" or
            identity["source_run_id"] != bundle["source_run_id"] or
            identity["checkpoint_ref"] != bundle["checkpoint_ref"] or
            cache.index["feature_dim"] != 160 or cache.index["num_views"] != 3 or
            cache.index["sample_count"] != len(ids)):
        raise ValueError("ITW cache must be a standalone 3178-row target10 cache with source identity")
    rows = cache.load_by_id()
    if set(rows) != set(ids) or len(rows) != len(ids):
        raise ValueError("standalone ITW target10 cache exact ID coverage failure")
    views = np.stack([rows[sid] for sid in ids])
    if views.shape != (len(ids), 3, 160) or views.dtype != np.float32 or \
            not np.isfinite(views).all():
        raise ValueError("standalone ITW target10 cache shape/dtype/finite failure")
    reference = [json.loads(line) for line in REFERENCE_SCORES.open()]
    if [row["sample_id"] for row in reference] != ids or len(reference) != len(ids):
        raise ValueError("prior selected-only Frozen reference coverage mismatch")
    predicted = views[:, 0].astype(np.float64) @ resources.w.numpy().astype(np.float64) + float(resources.b)
    maximum = float(np.max(np.abs(predicted - np.array(
        [row["score_frozen"] for row in reference], dtype=np.float64))))
    if maximum > 1e-5:
        raise ValueError(f"selected cache sample_index/Frozen parity failed: {maximum}")
    return ids, views, {"cache_ref": str(cache.root), "selection_ref": str(
        ROOT / "experiments/large_scale_confirmation/manifests/in_the_wild_confirmation_select.json"),
        "selected_count": len(ids), "reader": "standalone_selected_only_cache",
        "cache_id_sidecars_read": True, "nonselected_feature_rows_read": False,
        "frozen_score_parity_max_difference": maximum,
        "frozen_reference": str(REFERENCE_SCORES)}


def objective_values(views, resources, geometry, config, w, b):
    """Exact batched O1/O2/O3 formulas; frozen-source geometry and O3 q."""
    head = SimpleNamespace(U=resources.U, w=w, b=b)
    count = views.shape[0]
    probabilities = soft_affinity(views.reshape(-1, 160), head, geometry,
                                  config["affinity_temperature"]).reshape(count, 3, 2)
    consensus = probabilities.mean(dim=1)
    detached = consensus.detach().clamp_min(1e-12)
    agreement = (detached[:, None, :] *
                 (detached[:, None, :].log() - probabilities.clamp_min(1e-12).log())
                 ).sum(dim=-1).mean(dim=1)
    affinity_entropy = -(consensus * consensus.clamp_min(1e-12).log()).sum(dim=1)
    o1 = agreement + config["affinity_entropy_weight"] * affinity_entropy
    scores = views @ w + b
    o2 = (((scores - scores.mean(dim=1, keepdim=True)) /
           geometry["sigma_s"]).square()).mean(dim=1)
    with torch.no_grad():
        frozen_prob = soft_affinity(views.reshape(-1, 160), resources, geometry,
                                    config["affinity_temperature"]).reshape(count, 3, 2)
        frozen_agreement = (1 - frozen_prob[:, :, 1].std(dim=1, unbiased=False)).clamp(0, 1)
        frozen_gap = (frozen_prob[:, :, 1] - frozen_prob[:, :, 0]).mean(dim=1).abs()
        sharpness = config["o3_reliability_sharpness"]
        weight = torch.sigmoid(sharpness * (frozen_agreement - .5)) * \
                 torch.sigmoid(sharpness * (frozen_gap - .5))
    o3 = config["o3_affinity_weight"] * weight * o1 + \
         config["decision_consistency_weight"] * o2
    return o1, o2, o3


def parity_check(views, resources, geometry, config):
    zero_R = torch.zeros((resources.U.shape[1], resources.U.shape[1]), dtype=resources.w.dtype)
    batched = objective_values(views[:3], resources, geometry, config,
                               resources.w, torch.tensor(float(resources.b)))
    largest = 0.0
    for index in range(3):
        target = SimpleNamespace(features=views[index])
        o1, o2, *_ = objective_terms(zero_R, target, resources, geometry, config)
        weight = frozen_reliability(target, resources, geometry, config)[2]
        expected = (o1, o2, config["o3_affinity_weight"]*weight*o1 +
                    config["decision_consistency_weight"]*o2)
        largest = max(largest, *(abs(float(batched[arm][index]-expected[arm]))
                                 for arm in range(3)))
    if largest > 1e-5:
        raise ValueError(f"O1/O2/O3 original-formula parity failure: {largest}")
    return largest


def unsupervised_gradients(views, resources, geometry, config):
    """The signature deliberately has no target-label argument."""
    w = resources.w.detach().clone().requires_grad_(True)
    b = torch.tensor(float(resources.b), dtype=w.dtype, requires_grad=True)
    scores = views @ w + b
    frozen_teacher = ((views[:, 0] @ resources.w + resources.b) > resources.tau0)
    teacher = frozen_teacher.to(scores.dtype).unsqueeze(1)
    entropy = mean_view_entropy(scores).mean()
    pseudo = calibrated_pseudo_bce(calibrated_logits(scores, resources.tau0, 1.0),
                                   teacher)
    o1, o2, o3 = objective_values(views, resources, geometry, config, w, b)
    losses = {"ENT": entropy, "PL": pseudo, "O1": o1.mean(),
              "O2": config["decision_consistency_weight"]*o2.mean(),
              "O3": o3.mean()}
    result = {}
    for arm in ARMS:
        gw, gb = torch.autograd.grad(losses[arm], (w, b), retain_graph=True)
        value = (gw.detach().numpy().astype(np.float64), float(gb.detach()))
        if not np.isfinite(value[0]).all() or not math.isfinite(value[1]):
            raise FloatingPointError(f"{arm}: nonfinite unlabeled head gradient")
        result[arm] = value
    # The actual production view-variance loss has no head dependency.
    if view_loss(views).requires_grad:
        raise ValueError("generic feature-view loss unexpectedly depends on head")
    return result


def supervised_reference(views, labels, resources):
    """The only function that receives target development labels."""
    w = resources.w.detach().clone().requires_grad_(True)
    b = torch.tensor(float(resources.b), dtype=w.dtype, requires_grad=True)
    logits = views[:, 0] @ w + b
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
    gw, gb = torch.autograd.grad(loss, (w, b))
    value = (gw.detach().numpy().astype(np.float64), float(gb.detach()))
    if not np.isfinite(value[0]).all() or not math.isfinite(value[1]):
        raise FloatingPointError("nonfinite supervised reference gradient")
    return value


def combine(weighted, counts):
    total = float(sum(counts))
    return (sum(count*pair[0] for count, pair in zip(counts, weighted))/total,
            sum(count*pair[1] for count, pair in zip(counts, weighted))/total)


def comparison(domain, arm, level, index, count, unsup, sup):
    sw, sb = sup
    sn = float(np.linalg.norm(sw))
    if sn <= EPS:
        raise ValueError(f"{domain}: supervised w-gradient is zero")
    if unsup is None:
        return {"domain": domain, "objective": arm, "level": level,
                "chunk_index": index, "sample_count": count, "head_gradient": "zero/N/A",
                "cosine": None, "supervised_gradient_norm": sn,
                "unsupervised_gradient_norm": 0.0, "gradient_norm_ratio": 0.0,
                "supervised_bias_gradient": sb, "unsupervised_bias_gradient": 0.0,
                "bias_sign_match": None}
    uw, ub = unsup
    un = float(np.linalg.norm(uw))
    cosine = float(np.dot(uw, sw)/(un*sn + EPS)) if un > ZERO else None
    sign_match = (int(np.sign(ub) == np.sign(sb))
                  if abs(ub) > ZERO and abs(sb) > ZERO else None)
    return {"domain": domain, "objective": arm, "level": level,
            "chunk_index": index, "sample_count": count,
            "head_gradient": "nonzero" if un > ZERO else "zero/N/A",
            "cosine": cosine, "supervised_gradient_norm": sn,
            "unsupervised_gradient_norm": un, "gradient_norm_ratio": un/sn,
            "supervised_bias_gradient": sb, "unsupervised_bias_gradient": ub,
            "bias_sign_match": sign_match}


def summarize(rows):
    result = {}
    for domain in ("itw", "wavefake"):
        result[domain] = {}
        for arm in (*ARMS, "Base_view_variance"):
            group = [row for row in rows if row["domain"] == domain and row["objective"] == arm]
            full = next(row for row in group if row["level"] == "full")
            chunks = [row for row in group if row["level"] == "chunk"]
            cosines = np.array([row["cosine"] for row in chunks if row["cosine"] is not None])
            ratios = np.array([row["gradient_norm_ratio"] for row in chunks])
            bias = [row["bias_sign_match"] for row in chunks if row["bias_sign_match"] is not None]
            result[domain][arm] = {
                "full_cosine": full["cosine"],
                "mean_cosine": float(cosines.mean()) if len(cosines) else None,
                "median_cosine": float(np.median(cosines)) if len(cosines) else None,
                "p25_cosine": float(np.percentile(cosines, 25)) if len(cosines) else None,
                "p75_cosine": float(np.percentile(cosines, 75)) if len(cosines) else None,
                "positive_chunk_fraction": float((cosines > 0).mean()) if len(cosines) else None,
                "valid_cosine_chunks": int(len(cosines)), "chunk_count": len(chunks),
                "full_gradient_norm_ratio": full["gradient_norm_ratio"],
                "mean_chunk_gradient_norm_ratio": float(ratios.mean()),
                "full_supervised_gradient_norm": full["supervised_gradient_norm"],
                "full_unsupervised_gradient_norm": full["unsupervised_gradient_norm"],
                "full_supervised_bias_gradient": full["supervised_bias_gradient"],
                "full_unsupervised_bias_gradient": full["unsupervised_bias_gradient"],
                "bias_sign_match_fraction": float(np.mean(bias)) if bias else None,
                "valid_bias_chunks": len(bias), "head_gradient": full["head_gradient"]}
    return result


def decide(summary):
    def strongly_aligned(item):
        return (item["full_cosine"] is not None and item["full_cosine"] >= .2 and
                item["median_cosine"] is not None and item["median_cosine"] >= .2 and
                item["positive_chunk_fraction"] >= .75)
    if any(strongly_aligned(summary["itw"][arm]) and
           strongly_aligned(summary["wavefake"][arm]) for arm in ARMS):
        return "DIRECTION_ALIGNED_BUT_EXECUTION_INEFFECTIVE"
    if any(summary["itw"][arm]["full_cosine"] is not None and
           summary["wavefake"][arm]["full_cosine"] is not None and
           ((summary["itw"][arm]["full_cosine"] >= .2 and
             summary["wavefake"][arm]["full_cosine"] <= -.1) or
            (summary["wavefake"][arm]["full_cosine"] >= .2 and
             summary["itw"][arm]["full_cosine"] <= -.1)) for arm in ARMS):
        return "DOMAIN_DEPENDENT_ALIGNMENT"
    weak = sum(all(item["median_cosine"] is None or item["median_cosine"] <= .1
                   for item in (summary["itw"][arm], summary["wavefake"][arm]))
               for arm in ARMS)
    if weak >= 3:
        return "OBJECTIVE_GRADIENT_MISMATCH"
    return "INCONCLUSIVE"


def run(run_id, itw_selected_cache):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    torch.set_num_threads(1)
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        config = json.loads(CONFIG.read_text())
        wave_ids, wave_values, _, resources, wave_info = load_domain("wavefake")
        itw_ids, itw_values, itw_info = load_itw_selected_rows(resources, itw_selected_cache)
        domains = {"itw": (itw_ids, itw_values, itw_info),
                   "wavefake": (wave_ids, wave_values, wave_info)}
        geometry = source_geometry(resources)
        unsupervised = {}
        parity = {}
        for domain, (ids, values, _) in domains.items():
            views = torch.from_numpy(values.copy())
            parity[domain] = parity_check(views, resources, geometry, config)
            chunks = []
            for begin in range(0, len(ids), CHUNK):
                part = views[begin:begin+CHUNK]
                chunks.append({"begin": begin, "count": len(part),
                               "gradients": unsupervised_gradients(part, resources,
                                                                    geometry, config)})
            if sum(chunk["count"] for chunk in chunks) != len(ids):
                raise ValueError("unsupervised exact sample coverage failed")
            unsupervised[domain] = chunks
            print(f"{domain}: {len(ids)} selected feature rows, {len(chunks)} label-free chunks", flush=True)
        # Label sidecars are opened only after all unlabeled gradients exist.
        rows = []
        for domain, (ids, values, _) in domains.items():
            labels = load_labels(domain, ids)
            views = torch.from_numpy(values.copy())
            chunks = unsupervised[domain]
            supervised = [supervised_reference(views[chunk["begin"]:chunk["begin"]+chunk["count"]],
                                                torch.from_numpy(labels[chunk["begin"]:chunk["begin"]+chunk["count"]]),
                                                resources) for chunk in chunks]
            counts = [chunk["count"] for chunk in chunks]
            full_sup = combine(supervised, counts)
            for arm in (*ARMS, "Base_view_variance"):
                for index, (chunk, sup) in enumerate(zip(chunks, supervised)):
                    rows.append(comparison(domain, arm, "chunk", index, chunk["count"],
                                           chunk["gradients"].get(arm), sup))
                full_u = (combine([chunk["gradients"][arm] for chunk in chunks], counts)
                          if arm in ARMS else None)
                rows.append(comparison(domain, arm, "full", None, len(ids), full_u,
                                       full_sup))
        if any(not math.isfinite(row[key]) for row in rows
               for key in ("supervised_gradient_norm", "unsupervised_gradient_norm",
                           "gradient_norm_ratio", "supervised_bias_gradient",
                           "unsupervised_bias_gradient") if row[key] is not None):
            raise FloatingPointError("nonfinite output diagnostic")
        if any(not math.isfinite(row["cosine"]) for row in rows if row["cosine"] is not None):
            raise FloatingPointError("nonfinite cosine")
        summary = summarize(rows)
        decision = decide(summary)
        fieldnames = list(rows[0])
        with (out / "alignment.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        payload = {"run_id": run_id, "role": "SUPERVISED_DEVELOPMENT_DIAGNOSTIC_NOT_TTA",
                   "branch": subprocess.check_output(["git", "branch", "--show-current"],
                      cwd=ROOT, text=True).strip(),
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=ROOT, text=True).strip(),
                   "command": [sys.executable, *sys.argv], "chunk_size": CHUNK,
                   "sample_counts": {name: len(domain[0]) for name, domain in domains.items()},
                   "source_geometry_parity_max": parity,
                   "itw_feature_io": itw_info,
                   "wavefake_feature_io": wave_info,
                   "objective_config_ref": str(CONFIG),
                   "definitions": {"ENT": "existing mean_view_entropy(raw three-view scores)",
                     "PL": "existing calibrated_pseudo_bce with frozen original-view teacher at tau0, T=1",
                     "O1_O2_O3": "existing formulas at R=0, source geometry and O3 reliability frozen; head w,b variable",
                     "Base_view_variance": "existing feature view_loss independent of head: zero/N/A",
                     "supervised": "BCEWithLogits(original-view raw score, selected development label)"},
                   "full_gradient": "sample-count-weighted mean of independent chunk gradients; exact for separable mean loss",
                   "decision_rule": "strong: full and median cosine >=0.2 plus >=0.75 positive chunks on both domains; opposite-domain full cosine >=0.2 vs <=-0.1; mismatch: >=3/5 arms median <=0.1 on both; otherwise inconclusive",
                   "decision": decision, "domains": summary,
                   "target90_labels_accessed": False, "target90_metrics_accessed": False,
                   "target90_feature_values_accessed": False, "new_tta_method_implemented": False}
        write_new(out / "summary.json", payload)
        with (out / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("# Gradient Alignment Audit\n\n")
            stream.write("Supervised gradients use selected development labels solely as a diagnostic reference; no head update or new TTA method was run. All five unlabeled gradients were computed before target labels were opened.\n\n")
            stream.write("| Domain | Objective | Full cosine | Mean cosine | Median cosine | p25 | p75 | Positive chunk fraction | Full grad norm ratio | Bias sign-match fraction |\n")
            stream.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for domain in ("itw", "wavefake"):
                for arm in (*ARMS, "Base_view_variance"):
                    item = summary[domain][arm]
                    show = lambda value: "N/A" if value is None else f"{value:.4f}"
                    stream.write(f"| {domain} | {arm} | {show(item['full_cosine'])} | "
                                 f"{show(item['mean_cosine'])} | {show(item['median_cosine'])} | "
                                 f"{show(item['p25_cosine'])} | {show(item['p75_cosine'])} | "
                                 f"{show(item['positive_chunk_fraction'])} | "
                                 f"{show(item['full_gradient_norm_ratio'])} | "
                                 f"{show(item['bias_sign_match_fraction'])} |\n")
            stream.write(f"\nDecision: **{decision}**. ITW/WaveFake selected counts: 3178/4096; "
                         f"chunk counts: {len(unsupervised['itw'])}/{len(unsupervised['wavefake'])}. "
                         "`Base_view_variance` has no head gradient by definition. "
                         "Full-domain gradients are exact sample-weighted aggregates of 128-row chunk gradients. "
                         "For O1/O2/O3, this is a parameter-scope lift of the existing R loss at R=0, "
                         "not a replay of its R-update gradient. No target90 feature values, labels or metrics were read.\n")
        return payload
    except BaseException:
        write_new(out / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--itw-selected-cache", required=True,
                        help="Existing standalone 3178-row target10 feature cache; mixed target_test cache is rejected")
    args = parser.parse_args()
    report = run(args.run_id, args.itw_selected_cache)
    print(report["decision"], flush=True)
