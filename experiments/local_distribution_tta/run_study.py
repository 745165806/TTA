"""Label-free fixed-order local-buffer TTA score generation."""
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

import torch

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.math import apply_adapter, view_loss
from eptta.adaptation.objectives import entropy_from_logits
from eptta.adaptation.types import EPConfig, TargetViews
from experiments.multidomain_mechanism.guard_worker import load_context, load_select
from experiments.task_objective_discovery.objective_worker import dev_context
from experiments.task_objective_discovery.objectives import (
    run_objective, soft_affinity, source_geometry,
)
from experiments.local_distribution_tta.local import ordered_buffers, run_local_buffer, source_damage


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECT = ROOT / "experiments/multidomain_mechanism/manifests"
OBJECTIVE_CONFIG = ROOT / "experiments/task_objective_discovery/objective_config.json"
ARMS = ("Frozen", "Per-sample Base", "Per-sample O1", "Local-Base B16",
        "Local-Base B32", "Local-O1 B16", "Local-O1 B32")


def write_json_new(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validated_codecfake_cache(path):
    cache_ref = Path(path).resolve()
    if cache_ref.name != "compat_full512" or cache_ref.parent.name != "feature_cache":
        raise ValueError("full Codecfake compatibility cache is required")
    report = json.loads((cache_ref.parent.parent / "cache_validation.json").read_text(encoding="utf-8"))
    if (report.get("status") != "PASS" or report.get("sample_count") != 512 or
            report.get("unique_ids") != 512 or report.get("exact_selected_id_coverage") is not True or
            report.get("all_finite") is not True or report.get("cache_ref") != str(cache_ref) or
            report.get("target_labels_read") is not False):
        raise ValueError("Codecfake compatibility cache has not passed full validation")
    return cache_ref


def per_sample_diagnostic(arm, target, resources, R, frozen_score, score_after,
                          geometry, objective_config, elapsed, buffer_index):
    with torch.no_grad():
        adapted = apply_adapter(target.features, resources.U, R)
        before_logits = target.features @ resources.w + resources.b
        after_logits = adapted @ resources.w + resources.b
        affinity_before = soft_affinity(target.features, resources, geometry,
                                        objective_config["affinity_temperature"]).mean(0)
        affinity_after = soft_affinity(adapted, resources, geometry,
                                       objective_config["affinity_temperature"]).mean(0)
        row = {"sample_id": target.sample_id, "arm": arm, "buffer_index": buffer_index,
               "buffer_size": 1, "score_frozen": frozen_score, "score_after": score_after,
               "score_delta": score_after - frozen_score,
               "entropy_before": float(entropy_from_logits(before_logits).mean()),
               "entropy_after": float(entropy_from_logits(after_logits).mean()),
               "affinity_before": affinity_before.tolist(),
               "affinity_after": affinity_after.tolist(),
               "update_norm": float(torch.linalg.vector_norm(R)),
               "source_evidence_damage": float(source_damage(R, resources)),
               "runtime_seconds": elapsed, "numeric_status": "ok"}
    if any(type(value) is float and not math.isfinite(value) for value in row.values()):
        raise FloatingPointError("nonfinite per-sample diagnostic")
    return row


def score_domain(domain, rows, cache_ref, out, cfg, objective_config):
    wanted = {row["sample_id"] for row in rows}
    if domain == "in_the_wild":
        resources, features, cache_id, provenance = load_context(domain, wanted)
    else:
        resources, features, cache_id, provenance = dev_context(domain, cache_ref, wanted)
    geometry = source_geometry(resources)
    write_json_new(out / "diagnostics" / (domain + "_provenance.json"), provenance)
    targets = [TargetViews(row["sample_id"], torch.from_numpy(features[row["sample_id"]]),
                           cache_id) for row in rows]
    if len(targets) != len(wanted) or [item.sample_id for item in targets] != sorted(wanted):
        raise ValueError("fixed manifest order/coverage mismatch")
    scores = {}
    dynamics = []
    for index, target in enumerate(targets):
        frozen = run_cache_method("frozen", target, resources, cfg)
        if frozen["status"] != "ok" or frozen["score"] != frozen["score_before"]:
            raise ValueError("Frozen score mismatch")
        scores[(target.sample_id, "Frozen")] = per_sample_diagnostic(
            "Frozen", target, resources,
            torch.zeros((resources.U.shape[1], resources.U.shape[1]), dtype=target.features.dtype),
            frozen["score"], frozen["score"], geometry, objective_config, 0., index)
        start = time.perf_counter()
        base = run_cache_method("ep_no_keep", target, resources, cfg)
        if base["status"] != "ok":
            raise FloatingPointError("production Base numeric fallback")
        scores[(target.sample_id, "Per-sample Base")] = per_sample_diagnostic(
            "Per-sample Base", target, resources, base["R"], frozen["score"],
            base["score"], geometry, objective_config, time.perf_counter() - start, index)
        start = time.perf_counter()
        o1 = run_objective("O1", target, resources, cfg, objective_config, geometry)
        scores[(target.sample_id, "Per-sample O1")] = per_sample_diagnostic(
            "Per-sample O1", target, resources, o1["R"], frozen["score"],
            o1["score_after"], geometry, objective_config, time.perf_counter() - start, index)
    for size in (16, 32):
        for objective in ("Base", "O1"):
            arm = "Local-%s B%d" % (objective, size)
            for buffer_index, items in enumerate(ordered_buffers(targets, size)):
                result = run_local_buffer(items, resources, cfg, objective, objective_config,
                                          geometry, domain=domain, buffer_index=buffer_index)
                buffer_row = {"arm": arm, **result["buffer"]}
                if result["buffer"]["trace"] and result["buffer"]["trace"][0]["parameter_norm_before"] != 0:
                    raise ValueError("buffer did not reset adapter state")
                dynamics.append(buffer_row)
                for row in result["samples"]:
                    sample_id = row["sample_id"]
                    row = {"arm": arm, **row,
                           "update_norm": buffer_row["R_norm"],
                           "source_evidence_damage": buffer_row["source_evidence_damage"],
                           "runtime_seconds": buffer_row["runtime_seconds"] / len(items)}
                    if row["score_frozen"] != scores[(sample_id, "Frozen")]["score_frozen"]:
                        raise ValueError("local Frozen reference mismatch")
                    key = (sample_id, arm)
                    if key in scores:
                        raise ValueError("duplicate local sample score")
                    scores[key] = row
    expected = {(target.sample_id, arm) for target in targets for arm in ARMS}
    if set(scores) != expected:
        raise ValueError("incomplete seven-arm sample coverage")
    with (out / "scores" / (domain + ".jsonl")).open("x", encoding="utf-8") as stream:
        for target in targets:
            for arm in ARMS:
                row = {"domain": domain, **scores[(target.sample_id, arm)]}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (out / "diagnostics" / (domain + "_buffer_dynamics.jsonl")).open(
            "x", encoding="utf-8") as stream:
        for row in dynamics:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return {"sample_count": len(targets), "score_rows": len(scores),
            "local_buffer_rows": len(dynamics), "status": "SCORES_COMPLETE"}


def run(args):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment required")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    config = json.loads(OBJECTIVE_CONFIG.read_text(encoding="utf-8"))
    cfg = EPConfig(config["steps"], config["lr"], config["rho"],
                   config["gamma"], config["lambda_keep"])
    if (cfg.steps, cfg.lr, cfg.rho) != (5, .03, .1):
        raise ValueError("fixed objective-discovery budget changed")
    cache_ref = validated_codecfake_cache(args.codecfake_cache)
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for name in ("manifests", "logs", "scores", "diagnostics", "analysis"):
        (out / name).mkdir()
    try:
        rows_by_domain = {domain: load_select(domain, 32 if args.smoke else None)
                          for domain in ("in_the_wild", "codecfake")}
        for domain in rows_by_domain:
            shutil.copyfile(SELECT / (domain + "_mechanism_select.json"),
                            out / "manifests" / (domain + "_mechanism_select.json"))
        branch = subprocess.check_output(["git", "branch", "--show-current"],
                                         cwd=ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip()
        write_json_new(out / "run_config.json", {
            "run_id": args.run_id, "role": "engineering_smoke_no_labels" if args.smoke else
            "two_domain_development_scores_no_labels", "branch": branch, "commit": commit,
            "python": sys.version, "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": 2026, "datasets": {key: len(value) for key, value in rows_by_domain.items()},
            "arms": ARMS, "sample_order": "fixed_manifest_order_no_shuffle",
            "parameters": {"steps": cfg.steps, "lr": cfg.lr, "rho": cfg.rho,
                           "gamma": cfg.gamma, "lambda_keep": cfg.lambda_keep,
                           "buffer_sizes": [16, 32], "reset_policy": "per_buffer_zero_R",
                           "parameter_scope": "8x8_R", "objective_config": config},
            "command": sys.argv, "target_labels_read": False,
            "target90_metrics_accessed": False, "final_holdout_metrics_accessed": False})
        write_json_new(out / "provenance.json", {
            "source_bundle": str(ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"),
            "source_resources": str(ROOT / "outputs_v2/ssl_aasist/resources"),
            "codecfake_cache": str(cache_ref), "codecfake_compat_validation":
            str(cache_ref.parent.parent / "cache_validation.json"),
            "selection_role": "select_only", "audit_labels_read": False})
        status = {}
        for domain, rows in rows_by_domain.items():
            status[domain] = score_domain(domain, rows, cache_ref, out, cfg, config)
            print(domain, status[domain], flush=True)
        write_json_new(out / "diagnostics/score_completion.json", {
            "status": "SCORES_COMPLETE_LABELS_NOT_READ", "domains": status,
            "audit_labels_read": False})
        print(out, flush=True)
    except BaseException:
        write_json_new(out / "failure.json", {"status": "FAIL",
                                                  "traceback": traceback.format_exc(),
                                                  "audit_labels_read": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--codecfake-cache", required=True)
    args = parser.parse_args()
    args.run_id = args.run_id or (("local_smoke_" if args.smoke else "local_dev_") +
                                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run(args)
