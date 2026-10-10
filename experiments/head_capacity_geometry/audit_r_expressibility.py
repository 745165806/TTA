"""Actual EP adapter's effective-head span, radius, and supervised correction projection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

from eptta.adaptation.math import apply_adapter, project_frobenius_
from experiments.head_capacity_geometry.resources import ROOT, load_domain


HERE = Path(__file__).resolve().parent


def write_new(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cosine(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm > 1e-12 else 0.


def audit(run_id):
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "scores", "diagnostics", "analysis"):
        (out / part).mkdir()
    try:
        ids, views, _, resources, provenance = load_domain("itw")
        U = resources.U.numpy().astype(np.float64)
        w = resources.w.numpy().astype(np.float64)
        b = float(resources.b)
        gram_error = float(np.max(np.abs(U.T@U-np.eye(8))))
        if U.shape != (160,8) or w.shape != (160,) or gram_error > 1e-4:
            raise ValueError("actual frozen U is not an orthonormal 160x8 basis")
        v = U.T@w
        vnorm = float(np.linalg.norm(v))
        if vnorm <= 0:
            raise ValueError("source head orthogonal to R subspace")
        torch.manual_seed(2026)
        Z = torch.from_numpy(views[:32,0,:].copy())
        parity = []
        for _ in range(20):
            R = torch.randn((8,8), dtype=torch.float32)
            project_frobenius_(R, .1)
            direct = apply_adapter(Z, resources.U, R)@resources.w + resources.b
            effective = resources.w + resources.U@R.T@(resources.U.T@resources.w)
            derived = Z@effective + resources.b
            parity.append(float((direct-derived).abs().max()))
        parity_max = max(parity)
        if parity_max > 1e-5:
            raise ValueError(f"effective-head numerical parity failed: {parity_max}")
        geometry_runs = {"itw":HERE/"results/itw_head_geometry_20260927a",
                         "wavefake":HERE/"results/wavefake_head_geometry_20260927a"}
        rows = []
        radius = .1*vnorm
        for domain, path in geometry_runs.items():
            doc = json.loads((path/"diagnostics/head_weights.json").read_text())
            if not np.allclose(doc["source_w"], w, atol=1e-6, rtol=0):
                raise ValueError("head geometry used different source w")
            for fold in doc["folds"]:
                target = np.asarray(fold["H3_full_linear"]["w"], dtype=np.float64)
                target_same_norm = np.linalg.norm(w)*target/np.linalg.norm(target)
                delta = target_same_norm-w
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm <= 1e-10:
                    raise ValueError("zero target direction shift")
                coefficient = U.T@delta
                projected = U@coefficient
                residual = delta-projected
                projected_norm = float(np.linalg.norm(projected))
                fraction = projected_norm**2/delta_norm**2
                limited = coefficient*min(1.,radius/max(float(np.linalg.norm(coefficient)),1e-12))
                bounded_residual = delta-U@limited
                bounded_fraction = 1-float(np.linalg.norm(bounded_residual))**2/delta_norm**2
                rows.append({"domain":domain,"fold":fold["fold"],
                    "delta_norm":delta_norm,"projection_norm":projected_norm,
                    "residual_norm":float(np.linalg.norm(residual)),
                    "explained_fraction":fraction,
                    "cosine_delta_projection":cosine(delta,projected),
                    "bounded_explained_fraction":bounded_fraction,
                    "required_q_norm":float(np.linalg.norm(coefficient)),
                    "allowed_q_norm":radius})
        if len(rows) != 10:
            raise ValueError("expected ten target head folds")
        mismatch = all(sum(row["explained_fraction"]<.5 for row in rows if row["domain"]==domain)>=4
                       for domain in geometry_runs)
        with (out/"analysis/geometry.csv").open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        summary={"status":"PASS","effective_head_formula":"w_eff=w_s+U R.T (U.T w_s)",
            "U_orthogonality_max_abs_error":gram_error,"U_transpose_w_norm":vnorm,
            "q_radius_rho_0_1":radius,"parity_max_abs_score_difference":parity_max,
            "mismatch_rule":"at least 4 of 5 folds per domain have subspace explained_fraction < 0.5",
            "R_PARAMETERIZATION_MISMATCH":mismatch,
            "per_domain":{domain:{"mean_explained_fraction":float(np.mean([r["explained_fraction"] for r in rows if r["domain"]==domain])),
                "mean_bounded_explained_fraction":float(np.mean([r["bounded_explained_fraction"] for r in rows if r["domain"]==domain])),
                "folds_below_half":sum(r["explained_fraction"]<.5 for r in rows if r["domain"]==domain)}
                for domain in geometry_runs},
            "target90_labels_accessed":False,"target90_metrics_accessed":False}
        write_new(out/"run_config.json",{"run_id":run_id,"branch":subprocess.check_output(
            ["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "python":sys.version,"command":sys.argv,"rho":.1,"random_R_count":20,
            "random_seed":2026,"parity_tolerance":1e-5})
        write_new(out/"provenance.json",provenance)
        write_new(out/"diagnostics/parity.json",{"max_abs_difference":parity_max,
            "per_random_R_max_abs_difference":parity})
        write_new(out/"analysis/summary.json",summary)
        report=("# R expressibility from actual production mathematics\n\n"
            "`apply_adapter` implements `z_R=z+((zU)Rᵀ)Uᵀ`. Thus `w_eff=w_s+U Rᵀ(Uᵀw_s)` "
            "and all unconstrained direction changes lie in span(U). With ||R||_F≤0.1, "
            "the coefficient radius is 0.1||Uᵀw_s||.\n\n"
            f"Float32 parity max |direct−derived score|: {parity_max:.9g}. "
            f"U orthogonality max error: {gram_error:.9g}.\n\n"
            "| Domain | Mean subspace explained | Mean radius-limited explained | Folds <0.5 |\n"
            "|---|---:|---:|---:|\n")
        for domain,row in summary["per_domain"].items():
            report+=f"| {domain} | {row['mean_explained_fraction']:.6f} | {row['mean_bounded_explained_fraction']:.6f} | {row['folds_below_half']}/5 |\n"
        report+=f"\nPredeclared R_PARAMETERIZATION_MISMATCH = {mismatch}.\n"
        (out/"analysis/R_EXPRESSIBILITY_REPORT.md").write_text(report)
        print(summary,flush=True)
    except BaseException:
        write_new(out/"failure.json",{"status":"FAIL","traceback":traceback.format_exc()})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    audit(parser.parse_args().run_id)
