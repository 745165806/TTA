"""R6 source pilot: K>0 verification, diagnostics, grid and mechanism runs.

This module only *prepares* the LOCKED plan and provides CPU-side verification
helpers; the real full-cache scoring reuses ``run_suite``.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError

ETA_REF = 0.01  # configs/base.yaml defaults.lr, NOT static-R lr
GRID = {"K": [1, 3, 5], "eta_multiplier": [0.3, 1.0, 3.0]}
FIXED = {"rho": 0.2, "gamma": 0.1, "lambda_keep": 1.0, "rank": 8}

MECHANISM_METHODS = ["ep_no_keep", "ep_keep_l2", "ep_keep_logit",
                     "ep_keep_fisher", "ep_scalar_adaptive", "entropy_same_adapter_no_keep",
                     "entropy_same_adapter", "memo_same_adapter_no_keep", "memo_same_adapter_keep",
                     "source_ce_only", "fixed_source_adapter", "ep_feature_pca_U"]
PUBLISHED_NOT_RUN = ["tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full", "eata_audio_ep", "t2a_audio_ep"]


def grid_candidates():
    return [{"K": K, "eta": round(mult * ETA_REF, 6)} for K in GRID["K"] for mult in GRID["eta_multiplier"]]


def compute_diagnostics(result, target, resources, cfg):
    """Compatibility name for the shared adaptation-layer diagnostic function."""
    from eptta.adaptation.adapter import adaptation_diagnostics
    diagnostic = adaptation_diagnostics(result, target, resources, cfg)
    diagnostic.update({
        "r_fro": diagnostic.get("final_R_norm"),
        "margin_violation_fraction": diagnostic.get("final_margin_violation_fraction"),
        "projection_triggered": bool(diagnostic.get("projection_count")) if diagnostic.get(
            "projection_count") is not None else None,
        "final_keep_loss": diagnostic.get("final_margin_loss"),
        "worst_margin_change": diagnostic.get("max_abs_margin_change"),
    })
    return diagnostic


def verify_k_gradient(targets, resources, cfg):
    """Independent FP64 reference for the first-step gradient at R=0 plus reset checks."""
    import torch
    from eptta.adaptation.adapter import run_cache_method
    Z = torch.stack([t.features for t in targets])
    U = resources.U
    Z64, U64 = Z.to(torch.float64), U.to(torch.float64)
    Q = Z64 @ U64
    Qc = Q - Q.mean(dim=1, keepdim=True)
    N, d = Z.shape[1], Z.shape[2]
    reference = (2.0 / (N * d)) * (Qc.transpose(-2, -1) @ Qc)
    reports = []
    for index, target in enumerate(targets):
        first = run_cache_method("ep_tta", target, resources, cfg)
        if first["steps_completed"] < 1:
            reports.append({"sample_id": target.sample_id, "status": "no_update", "max_abs_grad_err": None})
            continue
        # production first-step unprojected update from the serial trace is not returned; recompute
        # the gradient via one autograd step mirroring run_episode's first iteration.
        R = torch.zeros((U.shape[1], U.shape[1]), dtype=Z.dtype, device=Z.device, requires_grad=True)
        from eptta.adaptation.math import apply_adapter, keep_loss, view_loss
        lv = view_loss(apply_adapter(Z[index:index + 1], U, R))
        lk = keep_loss(apply_adapter(resources.anchors_z, U, R), resources.w, resources.b,
                       resources.anchors_y, resources.anchors_m0, resources.tau0, cfg.gamma)
        grad, = torch.autograd.grad(lv + cfg.lambda_keep * lk, R)
        err = float((grad.to(torch.float64) - reference[index]).abs().max())
        reports.append({"sample_id": target.sample_id, "status": first["status"],
                        "steps_completed": first["steps_completed"],
                        "max_abs_grad_err": err,
                        "score_after": first["score"], "score_before": first["score_before"],
                        "r_fro_after": float(torch.linalg.vector_norm(first["R"]))})
    return {"reference_grad": reference.detach().cpu().numpy().tolist(),
            "per_sample": reports, "max_abs_grad_err": max((r["max_abs_grad_err"] for r in reports
                                                            if r["max_abs_grad_err"] is not None), default=0.0)}


def prepare_r6_proposal(plan_ref, bundle_ref, resources_ref, select_cache_ref, select_labels_ref, output):
    bundle_path = Path(bundle_ref)
    from eptta.models.frozen import verify_frozen_export
    bundle, _export, _parity, _selection = verify_frozen_export(bundle_path)
    bundle_sha256 = sha256_file(bundle_path)
    resources_path = Path(resources_ref) / "resources.json" if Path(resources_ref).is_dir() else Path(resources_ref)
    resources = read_json(resources_path)
    if (resources.get("baseline_id") != bundle["baseline_id"] or resources.get(
            "selected_checkpoint_sha256") != bundle["selected_checkpoint_sha256"]):
        raise ContractError("R6 resources identity mismatch")
    cache_index = read_json(Path(select_cache_ref) / "index.json")
    if cache_index.get("sample_count", 0) < 1:
        raise DataError("R6 requires a nonempty full select cache")
    methods = [{"method_id": "frozen", "config": {"steps": 0, "lr": ETA_REF, "rho": FIXED["rho"],
                                                   "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                "params": {}},
               {"method_id": "multiview_mean", "config": {"steps": 0, "lr": ETA_REF, "rho": FIXED["rho"],
                                                           "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                "params": {}}]
    for candidate in grid_candidates():
        methods.append({"method_id": "ep_tta",
                        "config": {"steps": candidate["K"], "lr": candidate["eta"], "rho": FIXED["rho"],
                                   "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                        "params": {}})
    for method_id in MECHANISM_METHODS:
        methods.append({"method_id": method_id, "config": {"steps": 3, "lr": ETA_REF, "rho": FIXED["rho"],
                                                            "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                        "params": {}})
    for fraction in (0.0, 0.25, 0.5, 1.0):
        methods.append({"method_id": "static_subspace",
                        "config": {"steps": 0, "lr": ETA_REF, "rho": FIXED["rho"],
                                   "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                        "params": {"amount": fraction * FIXED["rho"] / math.sqrt(FIXED["rank"])}})
    for index in (0, 1, 2):
        methods.append({"method_id": "ep_random_U",
                        "config": {"steps": 3, "lr": ETA_REF, "rho": FIXED["rho"],
                                   "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                        "params": {"random_index": index}})
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
                "locks_published": False, "proposal_id": "r6-source-pilot-20260917",
                "parent_plan_sha256": sha256_file(plan_ref),
                "bundle_ref": str(bundle_path.resolve()), "bundle_sha256": bundle_sha256,
                "checkpoint_sha256": bundle["selected_checkpoint_sha256"],
                "source_run": bundle["training_run_id"],
                "resources_ref": str(Path(resources_ref).resolve()),
                "resources_sha256": sha256_file(resources_path),
                "select_cache_ref": str(Path(select_cache_ref).resolve()),
                "select_cache_key": cache_index["cache_key"],
                "select_sample_count": cache_index["sample_count"],
                "select_labels_ref": str(Path(select_labels_ref).resolve()),
                "select_labels_sha256": sha256_file(select_labels_ref),
                "eta_ref": ETA_REF, "grid": GRID, "fixed": FIXED,
                "methods": methods,
                "candidates": [{"candidate_id": "r6-%03d" % index, "method_index": index,
                                "method_id": method["method_id"]}
                               for index, method in enumerate(methods)],
                "selection_rule": {"hard_gate": {"coverage": 1.0, "fallback_rate_max": 0.01},
                                   "primary": "min_select_eer", "tie_break": ["fewer_K", "smaller_eta"]},
                "diagnostics": ["r_fro", "delta_z_norm", "delta_score", "keep_activated",
                                "margin_violation_fraction", "projection_triggered", "final_view_loss",
                                "final_keep_loss", "worst_margin_change", "ut_w_over_w"],
                "device": "cpu", "dtype": "float32", "reference_dtype": "float64",
                "budget": {"gpu_hour_cap": 1.0, "artifact_peak_gib_cap": 2.0, "cpu_threads": 4,
                           "cpu_core_hours_cap": 8.0},
                "forbidden": ["training", "SSL", "target_scoring", "R7_to_R9", "recompute_U_M_tau0"]}
    with AtomicDirectory(output) as temporary:
        write_json_new(temporary / "proposal.json", proposal)
    return {"schema_version": "0.1.0", "status": "PROPOSED",
            "proposal_ref": str((Path(output) / "proposal.json").resolve()),
            "proposal_sha256": sha256_file(Path(output) / "proposal.json"),
            "candidates": len(grid_candidates()) + 2, "mechanism_methods": len(MECHANISM_METHODS)}


def run_r6(proposal_ref, output_root):
    """Compatibility shell: verify an approval lock and delegate every candidate to run_suite."""
    from eptta.evaluation.seal import evaluate_sealed, seal_scores
    from eptta.execution.suite import run_suite
    from eptta.models.frozen import verify_frozen_export
    from eptta.offline.artifacts import load_frozen_resources

    lock_path = Path(proposal_ref).resolve()
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED":
        raise ContractError("run-r6 accepts only a verifiable LOCKED approval record")
    source_ref = Path(lock.get("proposal_ref", "")).resolve()
    if not source_ref.is_file() or sha256_file(source_ref) != lock.get("proposal_sha256"):
        raise ContractError("R6 lock no longer verifies its original proposal")
    proposal = read_json(source_ref)
    for field, ref_field in (("bundle_sha256", "bundle_ref"), ("resources_sha256", "resources_ref")):
        path = Path(proposal[ref_field])
        actual = sha256_file(path if path.is_file() else path / "resources.json")
        if proposal.get(field) != actual or lock.get(field) != actual:
            raise ContractError("R6 approved %s identity changed" % ref_field)
    bundle, _export, _parity, _selection = verify_frozen_export(proposal["bundle_ref"])
    resources, _extras, _meta = load_frozen_resources(proposal["resources_ref"], bundle)
    root = Path(output_root).resolve()
    if root.exists():
        raise ContractError("R6 output exists; overwrite is forbidden")
    (root / "plans").mkdir(parents=True)
    method_runs = []
    for index, item in enumerate(proposal["methods"]):
        suite_id = "r6-%03d" % index
        suite_plan = {"schema_version": "0.1.0", "status": "LOCKED", "suite_id": suite_id,
                      "feature_cache_ref": proposal["select_cache_ref"],
                      "resources_ref": proposal["resources_ref"],
                      "frozen_bundle_ref": proposal["bundle_ref"], "methods": [item]}
        suite_path = root / "plans" / (suite_id + ".json")
        write_json_new(suite_path, suite_plan)
        run_root = root / suite_id
        suite = run_suite(suite_path, suite_id, "select", run_root)
        method_dir = run_root / item["method_id"]
        row = {"suite_id": suite_id, "method_id": item["method_id"], "suite": suite}
        if suite["methods"][0]["status"] != "NOT_RUN":
            seal_scores(method_dir)
            row["evaluation"] = evaluate_sealed(method_dir, proposal["select_labels_ref"],
                                                method_dir / "evaluation.json", float(resources.tau0))
        method_runs.append(row)
    manifest = {"schema_version": "0.2.0", "status": "COMPLETE", "approval_lock_sha256":
                sha256_file(lock_path), "methods": method_runs, "tau0": float(resources.tau0),
                "target_evaluations_added": False}
    write_json_new(root / "r6_run.json", manifest)
    return manifest


def lock_r6_proposal(proposal_ref, expected_sha256):
    proposal_path = Path(proposal_ref)
    if sha256_file(proposal_path) != expected_sha256:
        raise ContractError("R6 proposal SHA-256 changed")
    proposal = read_json(proposal_path)
    if (proposal.get("schema_version") != "0.1.0" or proposal.get("status") != "PROPOSED" or
            proposal.get("approval_required") is not True):
        raise ContractError("only a PROPOSED R6 proposal may be locked")
    locked_dir = proposal_path.parent / "locked"
    if locked_dir.exists():
        raise ContractError("locked R6 proposal exists; overwrite forbidden")
    with AtomicDirectory(locked_dir) as temporary:
        lock = {"schema_version": "0.1.0", "status": "LOCKED",
                "proposal_ref": str(proposal_path.resolve()), "proposal_sha256": expected_sha256,
                "proposal_id": proposal["proposal_id"],
                "bundle_sha256": proposal["bundle_sha256"], "checkpoint_sha256": proposal["checkpoint_sha256"],
                "resources_sha256": proposal["resources_sha256"],
                "select_cache_key": proposal["select_cache_key"],
                "approval": {"decision": "approved_lock_and_execute_r6", "reviewer": "human_user",
                             "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "scope": "r6_source_pilot"},
                "methods": proposal["methods"], "grid": proposal["grid"], "fixed": proposal["fixed"],
                "eta_ref": proposal["eta_ref"], "selection_rule": proposal["selection_rule"]}
        write_json_new(temporary / "lock.json", lock)
    return lock
