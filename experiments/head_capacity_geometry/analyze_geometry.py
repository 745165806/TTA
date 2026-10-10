"""Fold-specific source/target decision-head geometry, development only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def write_new(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cosine(a, b):
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    if denom <= 1e-12:
        raise ValueError("zero head displacement")
    return float(np.clip(np.dot(a, b)/denom, -1., 1.))


def geometry(domain, path):
    doc = json.loads((path / "diagnostics/head_weights.json").read_text())
    source_w = np.asarray(doc["source_w"], dtype=np.float64)
    source_b = float(doc["source_b"])
    source_u = source_w/np.linalg.norm(source_w)
    if source_w.shape != (160,) or len(doc["folds"]) != 5:
        raise ValueError("head geometry source/fold schema invalid")
    rows, deltas = [], []
    for fold in doc["folds"]:
        head = fold["H3_full_linear"]
        w = np.asarray(head["w"], dtype=np.float64)
        if w.shape != (160,) or not np.isfinite(w).all() or np.linalg.norm(w) <= 0:
            raise ValueError("invalid fitted target head")
        target_u = w/np.linalg.norm(w)
        delta = target_u-source_u
        c = cosine(source_u, target_u)
        rows.append({"domain":domain, "fold":fold["fold"], "cos_source_target":c,
                     "angle_degrees":math.degrees(math.acos(c)),
                     "normalized_delta_norm":float(np.linalg.norm(delta)),
                     "bias_delta":float(head["b"]-source_b),
                     "source_w_norm":float(np.linalg.norm(source_w)),
                     "target_w_norm":float(np.linalg.norm(w))})
        deltas.append(delta)
    return rows, deltas, source_w, source_b


def run(args):
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "scores", "diagnostics", "analysis"):
        (out / part).mkdir()
    try:
        inputs = {"itw": HERE / "results/itw_head_geometry_20260927a",
                  "wavefake": HERE / "results/wavefake_head_geometry_20260927a"}
        all_rows, deltas, sources = [], {}, {}
        for domain, path in inputs.items():
            rows, vectors, source_w, source_b = geometry(domain, path)
            all_rows += rows
            deltas[domain] = vectors
            sources[domain] = (source_w, source_b)
        if not np.allclose(sources["itw"][0], sources["wavefake"][0], atol=0, rtol=0) or \
                sources["itw"][1] != sources["wavefake"][1]:
            raise ValueError("domains used different frozen source heads")
        mean_itw = np.mean(deltas["itw"], axis=0)
        mean_wave = np.mean(deltas["wavefake"], axis=0)
        pairwise = [cosine(a,b) for a in deltas["itw"] for b in deltas["wavefake"]]
        with (out / "analysis/geometry.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        with (out / "analysis/crossdomain_pairwise_cosine.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["itw_fold", "wavefake_fold", "cosine_normalized_direction_delta"])
            for i, a in enumerate(deltas["itw"]):
                for j, b in enumerate(deltas["wavefake"]):
                    writer.writerow([i,j,cosine(a,b)])
        summary = {"status":"PASS", "role":"SUPERVISED_DEVELOPMENT_GEOMETRY_NOT_TTA",
            "mean_delta_crossdomain_cosine":cosine(mean_itw, mean_wave),
            "pairwise_delta_cosine_min":min(pairwise),
            "pairwise_delta_cosine_mean":float(np.mean(pairwise)),
            "pairwise_delta_cosine_max":max(pairwise),
            "per_domain": {domain:{"mean_angle_degrees":float(np.mean([row["angle_degrees"] for row in all_rows if row["domain"]==domain])),
                "angle_range_degrees":[min(row["angle_degrees"] for row in all_rows if row["domain"]==domain),
                                       max(row["angle_degrees"] for row in all_rows if row["domain"]==domain)],
                "mean_normalized_delta_norm":float(np.mean([row["normalized_delta_norm"] for row in all_rows if row["domain"]==domain]))}
                for domain in inputs},
            "target90_labels_accessed":False,"target90_metrics_accessed":False}
        write_new(out / "run_config.json", {"run_id":args.run_id,"branch":subprocess.check_output(
            ["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "python":sys.version,"command":sys.argv,"input_runs":{k:str(v) for k,v in inputs.items()},
            "definition":"delta_unit_head=w_target/||w_target||-w_source/||w_source||"})
        write_new(out / "provenance.json", {"source_head_same_across_domains":True,
            "source_w":sources["itw"][0].tolist(), "source_b":sources["itw"][1]})
        write_new(out / "analysis/summary.json", summary)
        (out / "analysis/report.md").write_text("# Supervised development head geometry\n\n"
            "Unit-direction displacements are compared across five folds per domain. "
            "No target90 or final labels/metrics were used. Cross-domain cosine is descriptive only.\n\n"
            f"Mean ITW/WaveFake delta cosine: {summary['mean_delta_crossdomain_cosine']:.6f}.\n")
        print(summary, flush=True)
    except BaseException:
        write_new(out / "failure.json", {"status":"FAIL", "traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    run(parser.parse_args())
