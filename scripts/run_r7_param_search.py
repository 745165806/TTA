#!/usr/bin/env python
"""R7 Stage 4: SSL-AASIST EP parameter search on source-select only.

Grid: steps in {1,3,5} x lr in {0.001,0.003,0.01} x rho in {0.1,0.2} = 18 combos.
Selection basis: source-select only (role="select", asv2019 dev select subset).
Output root: outputs_v2/ssl_aasist/experiments/<ep_k{steps}_lr{lr}_rho{rho}>.
Skips existing runs (no overwrite). Uses existing source-select feature cache.
"""
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/media/dell/data/fakeAudioDection/TTA")
GRID_CFG_DIR = ROOT / "configs/local_v2/ssl_aasist_target/grid"
GRID_CFG_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MANIFEST = "data/manifests_v2/asv2019_la/inference/select.jsonl"
SOURCE_LABELS = "data/manifests_v2/asv2019_la/labels/select.jsonl"
SOURCE_CACHE = "outputs_v2/ssl_aasist/cache-select"
RESOURCES = "outputs_v2/ssl_aasist/resources"
FROZEN_BUNDLE = "outputs_v2/ssl_aasist/frozen/bundle.json"
OUTPUT_ROOT = "outputs_v2/ssl_aasist/experiments"
SCOPE_ID = "asvspoof2019_la/dev/select_v2"

bundle = json.loads((ROOT / FROZEN_BUNDLE).read_text())
CHECKPOINT_REF = bundle["checkpoint_ref"]

STEPS = [1, 3, 5]
LRS = [0.001, 0.003, 0.01]
RHOS = [0.1, 0.2]


def fmt(v):
    return f"{v:g}"


def run(cmd, cwd, env):
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_path = ROOT / "logs/r7_param_search.log"
    csv_path = ROOT / "logs/r7_param_search.csv"
    rows = []
    log = []
    log.append("R7 Stage 4: SSL-AASIST EP parameter search (source-select only)")
    log.append("start: %s" % now)
    log.append("grid: steps=%s lr=%s rho=%s" % (STEPS, LRS, RHOS))
    log.append("checkpoint_ref: %s" % CHECKPOINT_REF)
    log.append("resources: %s" % RESOURCES)
    log.append("frozen_bundle: %s" % FROZEN_BUNDLE)
    log.append("selection basis: source-select only (no target_test selection)")
    log.append("")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"

    total = len(STEPS) * len(LRS) * len(RHOS)
    done = 0
    for steps in STEPS:
        for lr in LRS:
            for rho in RHOS:
                name = f"ep_k{fmt(steps)}_lr{fmt(lr)}_rho{fmt(rho)}"
                out_dir = ROOT / OUTPUT_ROOT / name
                cfg_path = GRID_CFG_DIR / (name + ".yaml")
                method = {
                    "method_id": "ep_tta",
                    "config": {"gamma": 0.1, "lambda_keep": 1.0, "lr": lr, "rho": rho, "steps": steps},
                    "params": {},
                }
                config = {
                    "schema_version": "0.1.0",
                    "command": "run-tta",
                    "run_name": name,
                    "output_root": OUTPUT_ROOT,
                    "role": "select",
                    "scope_id": SCOPE_ID,
                    "input_manifest": SOURCE_MANIFEST,
                    "feature_cache": SOURCE_CACHE,
                    "resources": RESOURCES,
                    "frozen_bundle": FROZEN_BUNDLE,
                    "method": method,
                    "threshold": None,
                    "threshold_source": None,
                    "fallback_rate_max": 0.01,
                    "selection_source": None,
                    "seed": 13,
                }
                cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

                record = {
                    "config": str(cfg_path.relative_to(ROOT)),
                    "checkpoint": CHECKPOINT_REF,
                    "resources": RESOURCES,
                    "dataset": "asvspoof2019_la (source select)",
                    "method": "ep_tta",
                    "steps": steps,
                    "lr": lr,
                    "rho": rho,
                    "gpu": "CPU (no CUDA device in execution env)",
                }

                metrics_json = out_dir / "metrics.json"
                if metrics_json.exists():
                    metrics = json.loads(metrics_json.read_text())["metrics"]
                    record.update({
                        "status": "SKIPPED_EXISTS",
                        "eer": metrics.get("eer"),
                        "auroc": metrics.get("auroc"),
                        "fpr": metrics.get("fpr"),
                        "fnr": metrics.get("fnr"),
                        "time_s": None,
                    })
                    rows.append(record)
                    log.append(f"SKIP (exists) {name}")
                    done += 1
                    continue

                t0 = time.time()
                rc, so, se = run([sys.executable, "-m", "eptta.cli", "run-tta",
                                  "--config", str(cfg_path)], cwd=ROOT, env=env)
                t_run = time.time() - t0
                if rc != 0:
                    record.update({"status": "RUN_TTA_ERROR", "time_s": round(t_run, 2),
                                   "error": (so + " " + se)[:500]})
                    rows.append(record)
                    log.append(f"ERROR run-tta {name}: rc={rc} {se[:300]}")
                    done += 1
                    continue

                t0 = time.time()
                rc2, so2, se2 = run([sys.executable, "-m", "eptta.cli", "evaluate",
                                     "--run", str(out_dir), "--labels",
                                     str(ROOT / SOURCE_LABELS)], cwd=ROOT, env=env)
                t_eval = time.time() - t0
                if rc2 != 0:
                    record.update({"status": "EVAL_ERROR", "time_s": round(t_run + t_eval, 2),
                                   "error": (so2 + " " + se2)[:500]})
                    rows.append(record)
                    log.append(f"ERROR evaluate {name}: rc={rc2} {se2[:300]}")
                    done += 1
                    continue

                metrics = json.loads((out_dir / "metrics.json").read_text())["metrics"]
                record.update({
                    "status": "EVALUATED",
                    "eer": metrics.get("eer"),
                    "auroc": metrics.get("auroc"),
                    "fpr": metrics.get("fpr"),
                    "fnr": metrics.get("fnr"),
                    "time_s": round(t_run + t_eval, 2),
                })
                rows.append(record)
                log.append(f"OK {name} | EER={metrics.get('eer')} AUROC={metrics.get('auroc')} "
                           f"FPR={metrics.get('fpr')} FNR={metrics.get('fnr')} t={t_run + t_eval:.1f}s")
                done += 1

    # baseline frozen EER from existing selection (for gain comparison)
    selection = json.loads((ROOT / "outputs_v2/ssl_aasist/selection.json").read_text())
    frozen_eer = None
    for fam in selection.get("families", {}).values():
        if fam.get("family") == "frozen":
            frozen_eer = fam.get("eer")
    log.append("")
    log.append("frozen source-select EER (from selection.json): %s" % frozen_eer)
    best = min([r for r in rows if r.get("eer") is not None], key=lambda r: r["eer"], default=None)
    log.append("best grid EER: %s (config=%s)" % (
        best.get("eer") if best else None,
        (best.get("steps"), best.get("lr"), best.get("rho")) if best else None))
    if best and frozen_eer is not None and best["eer"] < frozen_eer:
        log.append("GAIN: best source-select EER < frozen EER")
    else:
        log.append("NO_GAIN: no grid config improves source-select EER over frozen")
    log.append("end: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    log.append("TOTAL: %d" % done)

    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")

    fieldnames = ["config", "checkpoint", "resources", "dataset", "method", "steps", "lr", "rho",
                  "gpu", "status", "eer", "auroc", "fpr", "fnr", "time_s"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n".join(log))


if __name__ == "__main__":
    main()
