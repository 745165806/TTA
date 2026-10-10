"""Label-free, six-order confirmation scores for exactly four fixed methods."""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.types import EPConfig, TargetViews
from experiments.multidomain_mechanism.guard_worker import load_context
from experiments.task_objective_discovery.objective_worker import dev_context
from experiments.task_objective_discovery.objectives import source_geometry
from experiments.local_distribution_tta.local import ordered_buffers, run_local_buffer, source_damage


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CFG_REF = ROOT / "experiments/task_objective_discovery/objective_config.json"
DOMAINS = ("in_the_wild", "codecfake", "asv2019_la_dev")
ORDERS = ("manifest_order", "seed2026", "seed2027", "seed2028", "seed2029", "seed2030")
ARMS = ("Frozen", "Per-sample Base", "Local-Base B32", "Local-O1 B32")
FORBIDDEN = {"label", "raw_label", "canonical_label", "attack_id", "correct_before",
             "correct_after", "helpful_update", "harmful_update"}


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def ordered_ids(ids, key):
    result = list(ids)
    if key != "manifest_order":
        if key not in ORDERS:
            raise ValueError("unregistered order")
        random.Random(int(key.removeprefix("seed"))).shuffle(result)
    return result


def load_assignment(domain, limit=None):
    path = HERE / "manifests" / (domain + "_confirmation_select.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    full_count = 3178 if domain == "in_the_wild" else 5000
    rows = doc.get("records")
    if (doc.get("role") != "confirmation_select" or doc.get("dataset_id") != domain or
            doc.get("count") != full_count or len(rows) != full_count or
            FORBIDDEN.intersection(doc) or any(FORBIDDEN.intersection(row) for row in rows)):
        raise ValueError("label-free fixed assignment invalid")
    ids = [row["sample_id"] for row in rows]
    if ids != sorted(ids) or len(set(ids)) != full_count:
        raise ValueError("fixed assignment ID order invalid")
    return rows[:limit] if limit else rows


def checked_cache(path, domain):
    folder = Path(path).resolve()
    report = json.loads((folder.parent / "cache_validation.json").read_text(encoding="utf-8"))
    if (report.get("status") != "PASS" or report.get("dataset") != domain or
            report.get("sample_count") != 5000 or report.get("exact_selected_id_coverage") is not True or
            report.get("all_finite") is not True or report.get("target_labels_read") is not False or
            report.get("cache_ref") != str(folder)):
        raise ValueError("5,000-sample production feature cache has not passed validation")
    return folder


def load_domain(domain, rows, caches):
    wanted = {row["sample_id"] for row in rows}
    if domain == "in_the_wild":
        resources, features, cache_id, provenance = load_context(domain, wanted)
    else:
        resources, features, cache_id, provenance = dev_context(domain, caches[domain], wanted)
    targets = {row["sample_id"]: TargetViews(row["sample_id"],
               torch.from_numpy(features[row["sample_id"]]), cache_id) for row in rows}
    if set(targets) != wanted or resources.U.device.type != "cpu":
        raise ValueError("target feature coverage or production CPU path mismatch")
    return resources, targets, provenance


def per_sample_controls(ids, targets, resources, cfg):
    baseline = {}
    for sid in ids:
        target = targets[sid]
        frozen = run_cache_method("frozen", target, resources, cfg)
        start = time.perf_counter()
        base = run_cache_method("ep_no_keep", target, resources, cfg)
        elapsed = time.perf_counter() - start
        if (frozen["status"] != "ok" or base["status"] != "ok" or
                frozen["score"] != frozen["score_before"] or
                base["score_before"] != frozen["score"]):
            raise ValueError("Frozen/production per-sample control failure")
        norm = float(torch.linalg.vector_norm(base["R"]))
        damage = float(source_damage(base["R"], resources))
        if not all(math.isfinite(value) for value in
                   (frozen["score"], base["score"], norm, damage, elapsed)):
            raise FloatingPointError("nonfinite per-sample control")
        baseline[sid] = {
            "Frozen": {"score_frozen": frozen["score"], "score_after": frozen["score"],
                       "score_delta": 0., "R_norm": 0., "source_evidence_damage": 0.,
                       "runtime_seconds": 0., "numeric_status": "ok"},
            "Per-sample Base": {"score_frozen": frozen["score"], "score_after": base["score"],
                                "score_delta": base["score"] - frozen["score"],
                                "R_norm": norm, "source_evidence_damage": damage,
                                "runtime_seconds": elapsed, "numeric_status": "ok"}}
    return baseline


def score_order(domain, key, ids, targets, resources, cfg, config, geometry, out):
    order = ordered_ids(ids, key)
    if len(order) != len(ids) or set(order) != set(ids):
        raise ValueError("order coverage changed")
    write_once(out / "manifests" / (domain + "_" + key + "_order.json"), {
        "domain": domain, "order": key, "role": "label_free_order", "sample_ids": order})
    baseline = per_sample_controls(order, targets, resources, cfg)
    local = {sid: {} for sid in order}
    buffer_path = out / "diagnostics" / (domain + "_" + key + "_buffers.jsonl")
    with buffer_path.open("x", encoding="utf-8") as stream:
        for objective in ("Base", "O1"):
            arm = "Local-" + objective + " B32"
            for index, buffer_ids in enumerate(ordered_buffers(
                    [targets[sid] for sid in order], 32)):
                result = run_local_buffer(buffer_ids, resources, cfg, objective, config, geometry,
                                          domain=domain, buffer_index=index)
                record = {"domain": domain, "order": key, "arm": arm, **result["buffer"]}
                if (record["trace"][0]["parameter_norm_before"] != 0 or
                        record["R_norm"] > cfg.rho + 1e-6):
                    raise ValueError("buffer did not reset or exceeded parameter budget")
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                for row in result["samples"]:
                    sid = row["sample_id"]
                    if sid not in local or arm in local[sid] or row["score_frozen"] != \
                            baseline[sid]["Frozen"]["score_after"]:
                        raise ValueError("local coverage or Frozen mismatch")
                    local[sid][arm] = {"score_frozen": row["score_frozen"],
                                       "score_after": row["score_after"],
                                       "score_delta": row["score_delta"],
                                       "R_norm": record["R_norm"],
                                       "source_evidence_damage": record["source_evidence_damage"],
                                       "runtime_seconds": record["runtime_seconds"] / len(buffer_ids),
                                       "numeric_status": row["numeric_status"],
                                       "buffer_index": index, "buffer_size": len(buffer_ids)}
    score_path = out / "scores" / (domain + "_" + key + ".jsonl")
    with score_path.open("x", encoding="utf-8") as stream:
        for sid in order:
            if set(local[sid]) != set(ARMS[2:]):
                raise ValueError("incomplete local score coverage")
            for arm in ARMS:
                fields = baseline[sid][arm] if arm in baseline[sid] else local[sid][arm]
                stream.write(json.dumps({"domain": domain, "order": key, "sample_id": sid,
                                         "arm": arm, **fields}, allow_nan=False) + "\n")
    controls = {sid: (baseline[sid]["Frozen"]["score_after"],
                      baseline[sid]["Per-sample Base"]["score_after"]) for sid in order}
    return ({"sample_count": len(order), "score_rows": len(order) * len(ARMS),
             "buffer_rows": 2 * math.ceil(len(order) / 32), "numeric_failures": 0}, controls)


def run(args):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta environment required")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    config = json.loads(CFG_REF.read_text(encoding="utf-8"))
    cfg = EPConfig(config["steps"], config["lr"], config["rho"],
                   config["gamma"], config["lambda_keep"])
    if (cfg.steps, cfg.lr, cfg.rho, cfg.gamma, cfg.lambda_keep) != (5, .03, .1, .1, 0.):
        raise ValueError("fixed adaptation budget changed")
    caches = {"codecfake": checked_cache(args.codecfake_cache, "codecfake"),
              "asv2019_la_dev": checked_cache(args.la_cache, "asv2019_la_dev")}
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for name in ("manifests", "logs", "scores", "diagnostics", "analysis"):
        (out / name).mkdir()
    try:
        rows_by_domain = {domain: load_assignment(domain, 32 if args.smoke else None)
                          for domain in DOMAINS}
        for domain in DOMAINS:
            shutil.copyfile(HERE / "manifests" / (domain + "_confirmation_select.json"),
                            out / "manifests" / (domain + "_confirmation_select.json"))
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                         text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        write_once(out / "run_config.json", {
            "run_id": args.run_id, "branch": branch, "commit": commit,
            "role": "engineering_smoke_no_labels" if args.smoke else "large_dev_scores_no_labels",
            "python": sys.version, "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "datasets": {domain: len(rows) for domain, rows in rows_by_domain.items()},
            "orders": ORDERS, "arms": ARMS,
            "parameters": {"steps": cfg.steps, "lr": cfg.lr, "rho": cfg.rho,
                           "gamma": cfg.gamma, "lambda_keep": cfg.lambda_keep,
                           "buffer_size": 32, "parameter_scope": "8x8_R",
                           "optimizer": "projected_SGD", "reset": "per_buffer_zero_R",
                           "objective_config": config},
            "command": sys.argv, "target_labels_read": False,
            "target90_metrics_accessed": False, "final_holdout_metrics_accessed": False})
        write_once(out / "provenance.json", {"source_bundle":
                    str(ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json"),
                    "codecfake_cache": str(caches["codecfake"]),
                    "asv2019_la_dev_cache": str(caches["asv2019_la_dev"]),
                    "itw_cache": str(ROOT / "outputs_v2/ssl_aasist/cache-target-in_the_wild"),
                    "target_labels_read": False})
        completion = {}
        for domain, rows in rows_by_domain.items():
            ids = [row["sample_id"] for row in rows]
            resources, targets, provenance = load_domain(domain, rows, caches)
            write_once(out / "diagnostics" / (domain + "_provenance.json"), provenance)
            geometry = source_geometry(resources)
            completion[domain] = {}
            reference_controls = None
            for key in ORDERS:
                completion[domain][key], controls = score_order(domain, key, ids, targets, resources,
                    cfg, config, geometry, out)
                if reference_controls is None:
                    reference_controls = controls
                elif reference_controls != controls:
                    raise ValueError("independently rerun Frozen/per-sample controls changed with order")
                print(domain, key, completion[domain][key], flush=True)
        write_once(out / "diagnostics/score_completion.json", {
            "status": "SCORES_COMPLETE_LABELS_NOT_READ", "domains": completion,
            "audit_labels_read": False})
    except BaseException:
        write_once(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc(),
                                          "audit_labels_read": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--codecfake-cache", required=True)
    parser.add_argument("--la-cache", required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
