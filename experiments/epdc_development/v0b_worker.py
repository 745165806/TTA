"""Label-free EPDC v0-B score generation on fixed mechanism select manifests."""
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
from experiments.epdc_development.retired_margin import normalized_margin_loss, run_soft_preserve
from eptta.adaptation.math import apply_adapter, view_loss
from eptta.adaptation.objectives import entropy_from_logits
from eptta.adaptation.types import EPConfig, TargetViews
from experiments.multidomain_mechanism.guard_worker import DOMAINS, load_context, load_select


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SELECT = ROOT / "experiments/multidomain_mechanism/manifests"
CFG = EPConfig(steps=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0.0)
LAMBDA_PRESERVE = 1.0
RETENTION = 0.9


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def record(target, resources, result, domain, arm, frozen, elapsed):
    if result.get("status") != "ok" or result["score_before"] != frozen:
        raise ValueError("numeric fallback or Frozen mismatch")
    R = result["R"]
    with torch.no_grad():
        before = target.features
        after = apply_adapter(before, resources.U, R)
        anchors = apply_adapter(resources.anchors_z, resources.U, R)
        signs = 2 * resources.anchors_y.to(anchors.dtype) - 1
        margins = signs * (anchors @ resources.w + resources.b - resources.tau0)
        evidence_damage = (torch.relu(.9 * resources.anchors_m0 - margins) /
                           resources.anchors_m0).mean()
        preservation_loss = normalized_margin_loss(R, resources, RETENTION)
        row = {"sample_id": target.sample_id, "domain": domain, "arm": arm,
               "score_frozen": frozen, "score_before": result["score_before"],
               "score_after": result["score"],
               "entropy_before": float(entropy_from_logits(before @ resources.w + resources.b).mean()),
               "entropy_after": float(entropy_from_logits(after @ resources.w + resources.b).mean()),
               "view_before": float(view_loss(before)), "view_after": float(view_loss(after)),
               "update_norm": float(torch.linalg.vector_norm(R)),
               "distance_from_source": float(torch.linalg.vector_norm(R)),
               "source_margin_before": float(resources.anchors_m0.mean()),
               "source_margin_after": float(margins.mean()),
               "evidence_damage": float(evidence_damage),
               "preservation_loss": float(preservation_loss),
               "numeric_status": "ok", "runtime_seconds": elapsed}
    if any(type(v) is float and not math.isfinite(v) for v in row.values()):
        raise ValueError("non-finite EPDC diagnostic")
    return row


def run_one(domain, rows, out):
    ids = {r["sample_id"] for r in rows}
    resources, features, cache_id, provenance = load_context(domain, ids)
    write_once(out / "diagnostics" / f"{domain}_provenance.json", provenance)
    with (out / "scores" / f"{domain}.jsonl").open("x", encoding="utf-8") as stream:
        for item in rows:
            target = TargetViews(item["sample_id"], torch.from_numpy(features[item["sample_id"]]), cache_id)
            frozen = run_cache_method("frozen", target, resources, EPConfig(steps=0))
            if frozen["score"] != frozen["score_before"]:
                raise ValueError("Frozen mismatch")
            stream.write(json.dumps({"sample_id": target.sample_id, "domain": domain,
                                     "arm": "Frozen", "score_frozen": frozen["score"],
                                     "score_before": frozen["score"], "score_after": frozen["score"],
                                     "numeric_status": "ok"}, allow_nan=False) + "\n")
            for arm, lam in (("Base Adapt", 0.0), ("Base + Preserve", LAMBDA_PRESERVE)):
                start = time.perf_counter()
                result = run_soft_preserve(target, resources, CFG,
                                           lambda_preserve=lam, retention=RETENTION)
                elapsed = time.perf_counter() - start
                stream.write(json.dumps(record(target, resources, result, domain, arm,
                                               frozen["score"], elapsed), allow_nan=False) + "\n")
            stream.flush()
    return {"status": "SCORES_COMPLETE", "sample_count": len(rows), "score_rows": len(rows) * 3}


def run(args):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(1)
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("requires conda environment tta")
    run_id = args.run_id or ("smoke_v0b_" if args.smoke else "v0b_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("manifests", "logs", "scores", "diagnostics", "analysis"):
        (out / sub).mkdir()
    try:
        domains = args.domains or tuple(DOMAINS)
        selected = {d: load_select(d, 32 if args.smoke else None) for d in domains}
        for domain in domains:
            shutil.copyfile(SELECT / f"{domain}_mechanism_select.json",
                            out / "manifests" / f"{domain}_mechanism_select.json")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        write_once(out / "run_config.json", {
            "schema_version": "0.1.0", "run_id": run_id,
            "scientific_role": "engineering_smoke" if args.smoke else "mechanism_dev",
            "branch": branch, "commit": commit, "python": sys.version,
            "python_executable": sys.executable, "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "gpu_available": torch.cuda.is_available(), "gpu": None,
            "seed": 2026, "datasets": {d: len(r) for d, r in selected.items()},
            "method": "EPDC v0-B normalized source-margin evidence",
            "arms": ["Frozen", "Base Adapt", "Base + Preserve"],
            "parameters": {"steps": CFG.steps, "lr": CFG.lr, "rho": CFG.rho,
                           "gamma": CFG.gamma, "lambda_keep": CFG.lambda_keep,
                           "lambda_preserve": LAMBDA_PRESERVE, "retention": RETENTION,
                           "reset_policy": "episodic", "score_view": "original"},
            "command": " ".join(sys.argv), "target90_accessed": False,
            "final_holdout_labels_accessed": False})
        write_once(out / "provenance.json", {"source_bundle": "outputs_v2/ssl_aasist/frozen/bundle.json",
                                                "source_resources": "outputs_v2/ssl_aasist/resources",
                                                "select_only": True, "audit_labels_read": False})
        status = {}
        for domain, rows in selected.items():
            status[domain] = run_one(domain, rows, out)
            print(domain, status[domain], flush=True)
        write_once(out / "diagnostics/score_completion.json", {
            "status": "SCORES_COMPLETE_LABELS_NOT_READ", "domains": status,
            "not_run_domains": ["codecfake", "wavefake"], "audit_labels_read": False})
        print(out)
    except Exception:
        write_once(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--domains", nargs="*", choices=tuple(DOMAINS))
    parser.add_argument("--run-id")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
