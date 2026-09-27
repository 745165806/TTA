"""Generate task-objective scores from label-free fixed select manifests."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.math import apply_adapter, view_loss
from eptta.adaptation.types import EPConfig, TargetViews
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources
from experiments.multidomain_mechanism.guard_worker import load_context, load_select
from experiments.task_objective_discovery.objectives import run_objective, source_geometry


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "objective_config.json"
SELECT_ROOT = ROOT / "experiments/multidomain_mechanism/manifests"
PA_SELECT_ROOT = HERE / "manifests"
ARMS = ("Frozen", "ep_no_keep", "O1", "O2", "O3")


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def dev_context(domain, cache_ref, wanted):
    bundle, *_ = verify_frozen_export(ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json")
    resources, _, meta = load_frozen_resources(ROOT / "outputs_v2/ssl_aasist/resources", bundle)
    cache = FeatureCache(cache_ref)
    identity = cache.index["identity"]
    expected = {"source_run_id": bundle["source_run_id"],
                "checkpoint_ref": bundle["checkpoint_ref"],
                "preprocess": bundle["preprocess"],
                "dataset_id": domain, "split_role": "select"}
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("development cache source provenance mismatch")
    if cache.index["feature_dim"] != bundle["embedding_dim"] or cache.index["num_views"] != 3:
        raise ValueError("development cache feature schema mismatch")
    features = {}
    for ids, block in cache.iter_chunks():
        for i, sample_id in enumerate(ids):
            if sample_id in wanted:
                features[sample_id] = block[i].copy()
    if set(features) != wanted or any(not np.isfinite(z).all() for z in features.values()):
        raise ValueError("development selected coverage/finite values mismatch")
    return resources, features, cache.cache_id, {
        "baseline_id": bundle["baseline_id"], "source_run_id": bundle["source_run_id"],
        "checkpoint_ref": bundle["checkpoint_ref"], "cache_identity": identity,
        "cache_ref": str(cache_ref), "tau0": float(meta["scalars"]["tau0"]),
        "device": "cpu"}


def diagnostic(target, resources, arm, result, elapsed):
    R = result.get("R")
    if R is None:
        rank = resources.U.shape[1]
        R = torch.zeros((rank, rank), dtype=target.features.dtype)
    with torch.no_grad():
        adapted = apply_adapter(target.features, resources.U, R)
        source_adapted = apply_adapter(resources.anchors_z, resources.U, R)
        signs = 2 * resources.anchors_y.to(source_adapted.dtype) - 1
        margins = signs * (source_adapted @ resources.w + resources.b - resources.tau0)
        damage = (torch.relu(0.9 * resources.anchors_m0 - margins) /
                  resources.anchors_m0).mean()
        score_after = result.get("score_after", result.get("score"))
        score_before = result["score_before"]
        row = {"sample_id": target.sample_id, "arm": arm,
               "score_frozen": score_before, "score_before": score_before,
               "score_after": score_after, "score_delta": score_after - score_before,
               "view_loss_before": float(view_loss(target.features)),
               "view_loss_after": float(view_loss(adapted)),
               "update_norm": float(torch.linalg.vector_norm(R)),
               "source_evidence_damage": float(damage),
               "runtime_seconds": elapsed, "numeric_status": result.get("status", "ok"),
               "objective_before": result.get("objective_before"),
               "objective_after": result.get("objective_after"),
               "gradient_norm_first": ((result.get("trace") or [{}])[0].get("gradient_norm")
                                       if result.get("trace") else None),
               "affinity_agreement": result.get("affinity_agreement"),
               "affinity_gap": result.get("affinity_gap"),
               "reliability_weight": result.get("reliability_weight")}
    if row["numeric_status"] != "ok" or any(type(v) is float and not math.isfinite(v)
                                             for v in row.values()):
        raise ValueError("numeric failure or nonfinite objective result")
    return row


def run_domain(domain, rows, cache_ref, out, cfg, objective_config):
    wanted = {row["sample_id"] for row in rows}
    if domain in ("codecfake", "asv2019_pa_dev"):
        resources, features, cache_id, provenance = dev_context(domain, cache_ref, wanted)
    else:
        resources, features, cache_id, provenance = load_context(domain, wanted)
    geometry = source_geometry(resources)
    write_new(out / "diagnostics" / (domain + "_provenance.json"), provenance)
    path = out / "scores" / (domain + ".jsonl")
    with path.open("x", encoding="utf-8") as stream:
        for record in rows:
            target = TargetViews(record["sample_id"], torch.from_numpy(features[record["sample_id"]]), cache_id)
            frozen = run_cache_method("frozen", target, resources, cfg)
            for arm in ARMS:
                start = time.perf_counter()
                if arm == "Frozen":
                    result = frozen
                elif arm == "ep_no_keep":
                    result = run_cache_method(arm, target, resources, cfg)
                else:
                    result = run_objective(arm, target, resources, cfg, objective_config, geometry)
                row = diagnostic(target, resources, arm, result, time.perf_counter() - start)
                row["domain"] = domain
                if row["score_frozen"] != frozen["score"]:
                    raise ValueError("Frozen reference mismatch")
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    return {"status": "SCORES_COMPLETE", "samples": len(rows), "rows": len(rows) * len(ARMS)}


def run(args):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment required")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    objective_config = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg = EPConfig(objective_config["steps"], objective_config["lr"],
                   objective_config["rho"], objective_config["gamma"],
                   objective_config["lambda_keep"])
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "scores", "diagnostics", "analysis", "manifests"):
        (out / name).mkdir()
    try:
        selected = {}
        for domain in args.domains:
            if domain == "asv2019_pa_dev":
                source = PA_SELECT_ROOT / (domain + "_mechanism_select.json")
                doc = json.loads(source.read_text(encoding="utf-8"))
                rows = doc["records"]
                forbidden = {"label", "raw_label", "canonical_label", "attack_id"}
                if doc["role"] != "mechanism_select" or doc["dataset_id"] != domain or \
                        len(rows) != 270 or forbidden.intersection(doc) or \
                        any(forbidden.intersection(row) for row in rows):
                    raise ValueError("unsafe PA development selection")
                selected[domain] = rows[:32] if args.smoke else rows
            else:
                selected[domain] = load_select(domain, 32 if args.smoke else None)
        for domain in selected:
            source = PA_SELECT_ROOT if domain == "asv2019_pa_dev" else SELECT_ROOT
            shutil.copyfile(source / (domain + "_mechanism_select.json"),
                            out / "manifests" / (domain + "_mechanism_select.json"))
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                         text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        write_new(out / "run_config.json", {"run_id": args.run_id,
                  "role": "engineering_smoke_no_labels" if args.smoke else "mechanism_dev_scores_no_labels",
                  "branch": branch, "commit": commit, "python": sys.version,
                  "pytorch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu_available": torch.cuda.is_available(), "seed": 2026,
                  "datasets": {key: len(value) for key, value in selected.items()},
                  "arms": ARMS, "objective_formula_version": "O1-O3-v0",
                  "parameters": objective_config, "command": sys.argv,
                  "target90_labels_or_metrics_read": False,
                  "final_holdout_labels_read": False})
        write_new(out / "provenance.json", {"source_bundle": str(ROOT /
                  "outputs_v2/ssl_aasist/frozen/bundle.json"),
                  "source_resources": str(ROOT / "outputs_v2/ssl_aasist/resources"),
                  "codecfake_cache_ref": args.codecfake_cache,
                  "pa_cache_ref": args.pa_cache,
                  "adaptation_reads_select_only": True, "audit_labels_read": False})
        status = {}
        for domain, rows in selected.items():
            cache_ref = args.codecfake_cache if domain == "codecfake" else \
                args.pa_cache if domain == "asv2019_pa_dev" else None
            if domain in ("codecfake", "asv2019_pa_dev") and not cache_ref:
                raise ValueError("development domain requires explicit production feature cache")
            status[domain] = run_domain(domain, rows, cache_ref, out, cfg,
                                        objective_config)
            print(domain, status[domain], flush=True)
        write_new(out / "diagnostics/score_completion.json", {
            "status": "SCORES_COMPLETE_LABELS_NOT_READ", "domains": status,
            "audit_labels_read": False})
        print(out, flush=True)
    except BaseException:
        write_new(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--domains", nargs="+", choices=("in_the_wild", "codecfake", "asv2019_pa_dev"),
                        default=("in_the_wild", "codecfake"))
    parser.add_argument("--codecfake-cache")
    parser.add_argument("--pa-cache")
    args = parser.parse_args()
    args.run_id = args.run_id or (("objective_smoke_" if args.smoke else "objective_full_") +
                                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run(args)
