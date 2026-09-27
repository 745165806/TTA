"""Label-free guarded EP contrast on the fixed mechanism select manifests."""
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
from eptta.adaptation.objectives import entropy_from_logits, target_objective
from eptta.adaptation.regularizers import regularizer
from eptta.adaptation.types import EPConfig, TargetViews
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECT = HERE / "manifests"
RESULTS = HERE / "results"
DOMAINS = {
    "in_the_wild": ("cache-target-in_the_wild", "in_the_wild"),
    "asv2021_la": ("cache-target-asv2021_la_eval", "asvspoof2021_la"),
    "asv2021_df": ("cache-target-asv2021_df_eval", "asvspoof2021_df"),
}
SETTINGS = {
    "A": (5, 0.01, 0.05),
    "B": (5, 0.03, 0.1),
    "C": (10, 0.3, 0.2),
}
ARMS = {
    "hard_guard": "ep_tta_guarded",
    "relaxed_guard": "ep_tta_guard_relaxed",
    "unguarded": "ep_tta",
}
FORBIDDEN = frozenset(("label", "raw_label", "original_label", "attack_id", "source_labels",
                       "correct_before", "correct_after", "helpful_update", "harmful_update"))


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_select(domain, limit=None):
    path = SELECT / f"{domain}_mechanism_select.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("role") != "mechanism_select" or doc.get("dataset_id") != domain:
        raise ValueError("wrong mechanism select manifest")
    rows = doc.get("records")
    if not isinstance(rows, list) or len(rows) != doc.get("count") or not rows:
        raise ValueError("incomplete select manifest")
    if any(FORBIDDEN.intersection(row) for row in rows) or FORBIDDEN.intersection(doc):
        raise ValueError("target label/attack field entered select")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("select IDs must be sorted and unique")
    if any(row.get("dataset_id") != domain or row.get("selection_seed") != 2026
           or not row.get("root_ref") or not row.get("audio_relpath") for row in rows):
        raise ValueError("select provenance incomplete")
    return rows[:limit] if limit else rows


def load_context(domain, wanted):
    from experiments.oracle_diagnosis.common import contained_file
    base = ROOT / "outputs_v2/ssl_aasist"
    bundle, *_ = verify_frozen_export(base / "frozen/bundle.json")
    resources, _, meta = load_frozen_resources(base / "resources", bundle)
    cache_name, expected_dataset = DOMAINS[domain]
    cache = FeatureCache(base / cache_name)
    identity = cache.index["identity"]
    expected = {"source_run_id": bundle["source_run_id"],
                "checkpoint_ref": bundle["checkpoint_ref"],
                "preprocess": bundle["preprocess"],
                "dataset_id": expected_dataset, "split_role": "target_test"}
    if cache.index.get("format") != "sharded_npy_v2" or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("cache/frozen-bundle scientific provenance mismatch")
    if cache.index.get("num_views") != 3 or cache.index.get("feature_dim") != bundle["embedding_dim"]:
        raise ValueError("cache shape mismatch")
    for chunk in cache.index["chunks"]:
        contained_file(cache.root, chunk["array_ref"])
        contained_file(cache.root, chunk["ids_ref"])
    features = {}
    for chunk_ids, array in cache.iter_chunks():
        for i, sample_id in enumerate(chunk_ids):
            if sample_id in wanted:
                features[sample_id] = array[i].copy()
    if set(features) != wanted or resources.U.device.type != "cpu":
        raise ValueError("selected feature coverage or production CPU path mismatch")
    return resources, features, cache.cache_id, {
        "baseline_id": bundle["baseline_id"], "source_run_id": bundle["source_run_id"],
        "checkpoint_ref": bundle["checkpoint_ref"], "cache_id": cache.cache_id,
        "cache_identity": identity, "tau0": float(meta["scalars"]["tau0"]), "device": "cpu"}


def diagnostic(target, resources, cfg, result, frozen_score, *, domain, setting, arm, elapsed):
    if result["status"] != "ok":
        raise RuntimeError(f"numeric fallback for {target.sample_id}: {result.get('error_message')}")
    R = result["R"]
    z0 = target.features
    with torch.no_grad():
        z1 = apply_adapter(z0, resources.U, R)
        logits0 = z0 @ resources.w + resources.b
        logits1 = z1 @ resources.w + resources.b
        anchors = apply_adapter(resources.anchors_z, resources.U, R)
        signs = 2 * resources.anchors_y.to(anchors.dtype) - 1
        margins1 = signs * (anchors @ resources.w + resources.b - resources.tau0)
        m0 = resources.anchors_m0
        damage = torch.relu(0.9 * m0 - margins1) / m0
        # The gradient is evaluated at the final R, not the last pre-update R.
    with torch.enable_grad():
        probe = R.detach().clone().requires_grad_(True)
        loss = target_objective("view_variance", apply_adapter(z0, resources.U, probe),
                                resources.w, resources.b)
        loss = loss + cfg.lambda_keep * regularizer("margin", probe, resources, cfg.gamma)
        grad, = torch.autograd.grad(loss, probe)
    trace = result.get("trace") or []
    row = {
        "sample_id": target.sample_id, "domain": domain, "setting": setting, "arm": arm,
        "score_frozen": frozen_score, "score_before": float(result["score_before"]),
        "score_after": float(result["score"]),
        "entropy_before": float(entropy_from_logits(logits0).mean()),
        "entropy_after": float(entropy_from_logits(logits1).mean()),
        "view_before": float(view_loss(z0)), "view_after": float(view_loss(z1)),
        "gradient_norm_final": float(torch.linalg.vector_norm(grad)),
        "update_norm": float(torch.linalg.vector_norm(R)),
        "distance_from_source": float(torch.linalg.vector_norm(R)),
        "source_margin_before": float(m0.mean()),
        "source_margin_after": float(margins1.mean()),
        "evidence_damage": float(damage.mean()),
        "guard_activation": sum(bool(t.get("margin_guard_applied")) for t in trace),
        "guard_backtrack": sum(int(t.get("margin_guard_backtracks", 0)) for t in trace),
        "guard_revert": sum(bool(t.get("margin_guard_reverted")) for t in trace),
        "numeric_status": result["status"], "runtime_seconds": elapsed,
    }
    if any(type(v) is float and not math.isfinite(v) for v in row.values()):
        raise ValueError("non-finite diagnostic")
    return row


def run_one(domain, rows, output):
    wanted = {row["sample_id"] for row in rows}
    resources, features, cache_id, provenance = load_context(domain, wanted)
    write_once(output / "diagnostics" / f"{domain}_provenance.json", provenance)
    with (output / "scores" / f"{domain}.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            sample_id = row["sample_id"]
            target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache_id)
            frozen = run_cache_method("frozen", target, resources, EPConfig(steps=0))
            if frozen["status"] != "ok" or frozen["score"] != frozen["score_before"]:
                raise ValueError("Frozen reference mismatch")
            stream.write(json.dumps({"sample_id": sample_id, "domain": domain, "setting": "Frozen",
                                     "arm": "Frozen", "score_frozen": frozen["score"],
                                     "score_before": frozen["score_before"], "score_after": frozen["score"],
                                     "numeric_status": "ok"}, allow_nan=False) + "\n")
            for setting, (steps, lr, rho) in SETTINGS.items():
                cfg = EPConfig(steps=steps, lr=lr, rho=rho, gamma=0.1, lambda_keep=1.0)
                for arm, method in ARMS.items():
                    start = time.perf_counter()
                    result = run_cache_method(method, target, resources, cfg)
                    elapsed = time.perf_counter() - start
                    if result["score_before"] != frozen["score"]:
                        raise ValueError("adaptation/Frozen score mismatch")
                    record = diagnostic(target, resources, cfg, result, frozen["score"],
                                        domain=domain, setting=setting, arm=arm, elapsed=elapsed)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    return {"sample_count": len(rows), "score_rows": len(rows) * 10, "status": "COMPLETE"}


def run(args):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(1)
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("run under conda environment tta")
    run_id = args.run_id or ("smoke_" if args.smoke else "guard_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS / run_id
    output.mkdir(parents=True, exist_ok=False)
    for sub in ("manifests", "logs", "scores", "diagnostics", "analysis"):
        (output / sub).mkdir()
    try:
        domains = args.domains or tuple(DOMAINS)
        if any(domain not in DOMAINS for domain in domains):
            raise ValueError("domain has no production cache; mark NOT_RUN")
        selected = {domain: load_select(domain, 32 if args.smoke else None) for domain in domains}
        for domain in domains:
            shutil.copyfile(SELECT / f"{domain}_mechanism_select.json",
                            output / "manifests" / f"{domain}_mechanism_select.json")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        config = {"schema_version": "0.1.0", "run_id": run_id, "scientific_role": "engineering_smoke" if args.smoke else "mechanism_dev",
                  "branch": branch, "commit": commit, "python": sys.version, "python_executable": sys.executable,
                  "pytorch": torch.__version__, "cuda": torch.version.cuda, "gpu": None,
                  "seed": 2026, "datasets": {d: len(r) for d, r in selected.items()},
                  "method": "production_guard_capacity_contrast", "settings": SETTINGS,
                  "arms": ARMS, "gamma": 0.1, "lambda_keep": 1.0,
                  "command": " ".join(sys.argv), "target90_accessed": False,
                  "final_holdout_labels_accessed": False}
        write_once(output / "run_config.json", config)
        write_once(output / "provenance.json", {"source_bundle": "outputs_v2/ssl_aasist/frozen/bundle.json",
                                                   "source_resources": "outputs_v2/ssl_aasist/resources",
                                                   "select_only": True, "audit_labels_read": False})
        status = {}
        for domain, rows in selected.items():
            status[domain] = run_one(domain, rows, output)
            print(domain, status[domain], flush=True)
        write_once(output / "analysis" / "summary.json", {"status": "SCORES_COMPLETE_LABELS_NOT_READ",
                                                          "domains": status,
                                                          "missing_domains": ["codecfake", "wavefake"],
                                                          "audit_labels_read": False})
        print(output)
    except Exception:
        write_once(output / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--domains", nargs="*", choices=tuple(DOMAINS))
    parser.add_argument("--run-id")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
