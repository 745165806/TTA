"""R6-D1: update-scale and source-shift diagnostics (A/B matrix + keep reachability)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError

BUNDLE_SHA256 = "d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684"
CHECKPOINT_SHA256 = "076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a"
RESOURCE_BUNDLE_ID = "ep-resources-9133770434dc343e7060"
RESOURCES_SHA256 = "a7d9bc3b805267c9115995d773fd965725c31f8add2eaa85fbeeac8fa497d5c2"
CONDITION = {"kind": "awgn", "snr_db": 20.0, "seed": 13, "namespace": "r6_d1_c20", "purpose": "input_condition"}
ETA_GRID = [0.03, 0.3, 3.0]
K = 5
RHO = 0.2
GAMMA = 0.1
LAMBDA_KEEP = 1.0


def d1_methods():
    methods = [
        {"method_id": "frozen", "config": {"steps": 0, "lr": 0.01, "rho": RHO, "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP}, "params": {}},
        {"method_id": "multiview_mean", "config": {"steps": 0, "lr": 0.01, "rho": RHO, "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP}, "params": {}},
    ]
    for eta in ETA_GRID:
        for method_id in ("ep_tta", "ep_no_keep"):
            methods.append({"method_id": method_id,
                            "config": {"steps": K, "lr": eta, "rho": RHO, "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP},
                            "params": {}})
    return methods


def prepare_d1_proposal(bundle_ref, resources_ref, c0_cache_ref, select_labels_ref, output):
    from eptta.models.frozen import verify_frozen_export
    bundle, _e, _p, _s = verify_frozen_export(bundle_ref)
    if sha256_file(bundle_ref) != BUNDLE_SHA256 or bundle["selected_checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ContractError("D1 bundle identity mismatch")
    resources = read_json(Path(resources_ref) / "resources.json")
    if (resources.get("artifact_bundle_id") != RESOURCE_BUNDLE_ID or
            resources.get("baseline_id") != "baseline-d1f0d91901c73eb5027c"):
        raise ContractError("D1 resources identity mismatch")
    c0 = read_json(Path(c0_cache_ref) / "index.json")
    if c0.get("sample_count") != 11520:
        raise DataError("D1 requires the full select cache")
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
                "locks_published": False, "proposal_id": "r6-d1-20260917",
                "bundle_ref": str(Path(bundle_ref).resolve()), "bundle_sha256": BUNDLE_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "resources_ref": str(Path(resources_ref).resolve()), "resources_sha256": RESOURCES_SHA256,
                "c0_cache_ref": str(Path(c0_cache_ref).resolve()), "c0_cache_key": c0["cache_key"],
                "select_labels_ref": str(Path(select_labels_ref).resolve()),
                "select_labels_sha256": sha256_file(select_labels_ref),
                "condition": CONDITION, "condition_sha256": content_hash(CONDITION),
                "K": K, "eta_grid": ETA_GRID, "rho": RHO, "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP,
                "methods": d1_methods(),
                "matrix": {"conditions": ["C0", "C20"], "rows": ["frozen", "multiview_mean",
                           "ep_tta_K5_eta0.03", "ep_tta_K5_eta0.3", "ep_tta_K5_eta3.0",
                           "ep_no_keep_K5_eta0.03", "ep_no_keep_K5_eta0.3", "ep_no_keep_K5_eta3.0"]},
                "numerical_mode": {"dtype": "float32", "tf32_matmul": False, "tf32_cudnn": True,
                                   "block_units": 256, "input_condition": CONDITION},
                "budget": {"gpu_hour_cap": 0.5, "artifact_peak_gib_cap": 2.0, "cpu_threads": 4,
                           "cpu_core_hours_cap": 8.0},
                "forbidden": ["training", "SSL", "target_scoring", "R7_to_R9", "recompute_U_M_tau0",
                              "change_probe_strength", "change_rho"]}
    with AtomicDirectory(output) as temporary:
        write_json_new(temporary / "proposal.json", proposal)
    return {"schema_version": "0.1.0", "status": "PROPOSED",
            "proposal_ref": str((Path(output) / "proposal.json").resolve()),
            "proposal_sha256": sha256_file(Path(output) / "proposal.json"),
            "condition_sha256": content_hash(CONDITION)}


def lock_d1_proposal(proposal_ref, expected_sha256):
    proposal_path = Path(proposal_ref)
    if sha256_file(proposal_path) != expected_sha256:
        raise ContractError("D1 proposal SHA-256 changed")
    proposal = read_json(proposal_path)
    if (proposal.get("schema_version") != "0.1.0" or proposal.get("status") != "PROPOSED" or
            proposal.get("approval_required") is not True):
        raise ContractError("only a PROPOSED D1 proposal may be locked")
    locked_dir = proposal_path.parent / "locked"
    if locked_dir.exists():
        raise ContractError("locked D1 proposal exists; overwrite forbidden")
    with AtomicDirectory(locked_dir) as temporary:
        lock = {"schema_version": "0.1.0", "status": "LOCKED",
                "proposal_ref": str(proposal_path.resolve()), "proposal_sha256": expected_sha256,
                "proposal_id": proposal["proposal_id"],
                "bundle_sha256": proposal["bundle_sha256"], "checkpoint_sha256": proposal["checkpoint_sha256"],
                "resources_sha256": proposal["resources_sha256"], "condition": proposal["condition"],
                "condition_sha256": proposal["condition_sha256"], "K": proposal["K"],
                "eta_grid": proposal["eta_grid"], "methods": proposal["methods"],
                "approval": {"decision": "approved_lock_and_execute_r6_d1", "reviewer": "human_user",
                             "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "scope": "r6_d1_diagnostic"}}
        write_json_new(temporary / "lock.json", lock)
    return lock


def analyze_keep_reachability(resources, rho, gamma):
    """Per-anchor affine margin gradient bound within the Frobenius ball."""
    import torch
    U, w = resources.U, resources.w
    q = resources.anchors_z @ U  # [M, r]
    c = U.T @ w  # [r]
    signs = 2 * resources.anchors_y.to(q.dtype) - 1
    # A_i = signs[i] * outer(c, q_i); ||A_i||_F = ||c||_2 ||q_i||_2
    gradient_norm = c.norm() * q.norm(dim=1)  # [M]
    bound = rho * gradient_norm  # max |margin change| over ||R||_F<=rho
    threshold = gamma * resources.anchors_m0  # margin decrease that triggers keep
    reachable = bound > threshold  # can possibly trigger
    rows = []
    for i in range(resources.anchors_z.shape[0]):
        rows.append({"anchor": i, "y": int(resources.anchors_y[i]), "m0": float(resources.anchors_m0[i]),
                     "gradient_norm": float(gradient_norm[i]), "bound": float(bound[i]),
                     "threshold": float(threshold[i]), "reachable": bool(reachable[i])})
    return {"rho": float(rho), "gamma": float(gamma), "c_norm": float(c.norm()),
            "reachable_count": int(reachable.sum()), "total": int(len(rows)),
            "min_ratio_threshold_to_bound": float((threshold / bound.clamp_min(1e-30)).min()),
            "anchors": rows}


def run_d1_matrix(proposal_ref, cache_ref, condition_label, output_root):
    """Score the 8 D1 rows over one condition cache, collect diagnostics, seal and evaluate."""
    import torch
    from eptta.adaptation.types import EPConfig, TargetViews
    from eptta.baselines.dispatch import run_method
    from eptta.cache.reader import FeatureCache
    from eptta.models.frozen import verify_frozen_export
    from eptta.offline.artifacts import load_frozen_resources
    from eptta.evaluation.seal import seal_scores, evaluate_sealed
    from eptta.execution.r6 import compute_diagnostics
    proposal = read_json(proposal_ref)
    bundle, _e, _p, _s = verify_frozen_export(proposal["bundle_ref"])
    resources, extras, meta = load_frozen_resources(proposal["resources_ref"], bundle)
    cache = FeatureCache(cache_ref)
    features = cache.load_by_id()
    root = Path(output_root).resolve()
    method_runs = []
    for index, item in enumerate(proposal["methods"]):
        method_id = item["method_id"]
        cfg = EPConfig(**item["config"])
        params = dict(item["params"])
        method_resources = resources
        name = "method-%03d" % index
        method_dir = root / name
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "scores.jsonl").open("x", encoding="utf-8") as sf, \
             (method_dir / "diagnostics.jsonl").open("x", encoding="utf-8") as df:
            for sample_id in sorted(features):
                target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache.index["cache_key"])
                frozen = float(target.features[0] @ method_resources.w + method_resources.b)
                try:
                    result = run_method(method_id, target, method_resources, cfg, params)
                except FloatingPointError as exc:
                    result = {"method_id": method_id, "sample_id": sample_id, "score": frozen,
                              "score_before": frozen, "steps_completed": 0,
                              "objective_evaluations": 0, "status": "fallback_numeric",
                              "error_type": type(exc).__name__, "error_message": str(exc)}
                diag = compute_diagnostics(result, target, method_resources, cfg)
                row = {"schema_version": "0.1.0", "sample_id": sample_id, "score": result.get("score"),
                       "score_before": result.get("score_before"), "status": result.get("status", "ok"),
                       "method_id": method_id, "steps_completed": result.get("steps_completed", 0),
                       "objective_evaluations": result.get("objective_evaluations", 0),
                       "r_fro": diag.get("r_fro"), "runtime_ref": None}
                sf.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                diag_row = {"schema_version": "0.1.0", "sample_id": sample_id, "method_id": method_id}
                diag_row.update(diag)
                df.write(json.dumps(diag_row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        seal_scores(method_dir)
        ev = evaluate_sealed(method_dir, proposal["select_labels_ref"], method_dir / "evaluation.json",
                             float(resources.tau0))
        method_runs.append({"index": index, "method_id": method_id, "dir": name,
                            "config": item["config"], "params": params,
                            "metrics": ev["metrics"]})
    manifest = {"schema_version": "0.1.0", "status": "COMPLETE", "condition": condition_label,
                "cache_key": cache.index["cache_key"], "methods": method_runs, "tau0": float(resources.tau0)}
    write_json_new(root / ("run_" + condition_label + ".json"), manifest)
    return manifest
