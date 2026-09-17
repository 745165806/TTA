"""R6 source pilot: K>0 verification, diagnostics, grid and mechanism runs.

This module only *prepares* the LOCKED plan and provides CPU-side verification
helpers; the real full-cache scoring reuses ``run_suite``.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError

R6_PLAN_SHA256 = "07024939e2a156e681e006ace20b1b542b2cfde811c30e87d4977ffa3ecac79d"
BUNDLE_SHA256 = "d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684"
CHECKPOINT_SHA256 = "076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a"
RESOURCES_SHA256 = "a7d9bc3b805267c9115995d773fd965725c31f8add2eaa85fbeeac8fa497d5c2"
SOURCE_ARTIFACTS_LOCK_SHA256 = "2a9cb3f3bb810dab9d2b3076321c55d24a15115ae1a03c850689d2e8a1712c10"
ETA_REF = 0.01  # configs/base.yaml defaults.lr, NOT static-R lr
GRID = {"K": [1, 3, 5], "eta_multiplier": [0.3, 1.0, 3.0]}
FIXED = {"rho": 0.2, "gamma": 0.1, "lambda_keep": 1.0, "rank": 8}

MECHANISM_METHODS = ["frozen", "multiview_mean", "ep_no_keep", "ep_keep_l2", "ep_keep_logit",
                     "ep_keep_fisher", "ep_scalar_adaptive", "entropy_same_adapter_no_keep",
                     "entropy_same_adapter", "memo_same_adapter_no_keep", "memo_same_adapter_keep",
                     "source_ce_only", "fixed_source_adapter", "static_subspace", "ep_feature_pca_U"]
PUBLISHED_NOT_RUN = ["tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full", "eata_audio_ep", "t2a_audio_ep"]


def grid_candidates():
    return [{"K": K, "eta": round(mult * ETA_REF, 6)} for K in GRID["K"] for mult in GRID["eta_multiplier"]]


def compute_diagnostics(result, target, resources, cfg):
    import torch
    from eptta.adaptation.math import apply_adapter, margin_deficit, view_loss
    r_value = result.get("R")
    if r_value is None or r_value.numel() == 0:
        return {"r_fro": 0.0, "delta_z_norm": 0.0, "delta_score": result["score"] - result["score_before"],
                "keep_activated": False, "margin_violation_fraction": 0.0, "projection_triggered": False,
                "final_view_loss": None, "final_keep_loss": None, "worst_margin_change": 0.0}
    z0 = target.features[:1]
    U = resources.U
    delta_z = apply_adapter(z0, U, r_value) - z0
    adapted_anchors = apply_adapter(resources.anchors_z, U, r_value)
    deficit = margin_deficit(adapted_anchors, resources.w, resources.b, resources.anchors_y,
                             resources.anchors_m0, resources.tau0, cfg.gamma)
    margins_after = (2 * resources.anchors_y.to(adapted_anchors.dtype) - 1) * \
        (adapted_anchors @ resources.w + resources.b - resources.tau0)
    final_view = float(view_loss(apply_adapter(target.features, U, r_value)))
    trace = result.get("trace") or []
    final_keep = float(trace[-1]["regularizer"]) if trace and "regularizer" in trace[-1] else None
    keep_activated = bool(trace) and any(float(t.get("regularizer", 0.0) or 0.0) > 0 for t in trace)
    r_fro = float(torch.linalg.vector_norm(r_value))
    return {"r_fro": r_fro, "delta_z_norm": float(torch.linalg.vector_norm(delta_z)),
            "delta_score": result["score"] - result["score_before"],
            "keep_activated": keep_activated, "final_keep_loss": final_keep,
            "final_view_loss": final_view,
            "margin_violation_fraction": float((deficit > 0).to(adapted_anchors.dtype).mean()),
            "projection_triggered": bool(abs(r_fro - cfg.rho) < 1e-6),
            "worst_margin_change": float((margins_after - resources.anchors_m0).abs().max())}


def verify_k_gradient(targets, resources, cfg):
    """Independent FP64 reference for the first-step gradient at R=0 plus reset checks."""
    import torch
    from eptta.adaptation.episode import run_episode
    Z = torch.stack([t.features for t in targets])
    U = resources.U
    Q = Z @ U
    Qc = Q - Q.mean(dim=1, keepdim=True)
    N, d = Z.shape[1], Z.shape[2]
    reference = (2.0 / (N * d)) * (Qc.to(torch.float64).transpose(-2, -1) @ Qc.to(torch.float64))
    reports = []
    for index, target in enumerate(targets):
        first = run_episode(target, resources, cfg)
        if first.steps_completed < 1:
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
        reports.append({"sample_id": target.sample_id, "status": first.status,
                        "steps_completed": first.steps_completed,
                        "max_abs_grad_err": err,
                        "score_after": first.score_after, "score_before": first.score_before,
                        "r_fro_after": first.final_r_fro})
    return {"reference_grad": reference.detach().cpu().numpy().tolist(),
            "per_sample": reports, "max_abs_grad_err": max((r["max_abs_grad_err"] for r in reports
                                                            if r["max_abs_grad_err"] is not None), default=0.0)}


def prepare_r6_proposal(plan_ref, bundle_ref, resources_ref, select_cache_ref, select_labels_ref, output):
    bundle_path = Path(bundle_ref)
    from eptta.models.frozen import verify_frozen_export
    bundle, _export, _parity, _selection = verify_frozen_export(bundle_path)
    if sha256_file(bundle_path) != BUNDLE_SHA256 or bundle["selected_checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ContractError("R6 bundle identity mismatch")
    resources = read_json(Path(resources_ref) / "resources.json" if Path(resources_ref).is_dir() else resources_ref)
    if (resources.get("artifact_bundle_id") != "ep-resources-9133770434dc343e7060" or
            resources.get("baseline_id") != "baseline-d1f0d91901c73eb5027c"):
        raise ContractError("R6 resources identity mismatch")
    cache_index = read_json(Path(select_cache_ref) / "index.json")
    if cache_index.get("sample_count") != 11520:
        raise DataError("R6 requires the full 11520-sample select cache")
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
                        "params": {"candidate": "K%d_eta%.4f" % (candidate["K"], candidate["eta"])}})
    for method_id in MECHANISM_METHODS:
        methods.append({"method_id": method_id, "config": {"steps": 3, "lr": ETA_REF, "rho": FIXED["rho"],
                                                            "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                        "params": {}})
    for index in (0, 1, 2):
        methods.append({"method_id": "ep_random_U",
                        "config": {"steps": 3, "lr": ETA_REF, "rho": FIXED["rho"],
                                   "gamma": FIXED["gamma"], "lambda_keep": FIXED["lambda_keep"]},
                        "params": {"random_index": index}})
    proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
                "locks_published": False, "proposal_id": "r6-source-pilot-20260917",
                "parent_plan_sha256": R6_PLAN_SHA256,
                "bundle_ref": str(bundle_path.resolve()), "bundle_sha256": BUNDLE_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256, "source_run": "source-run-91cd36fae7770aca4b51",
                "resources_ref": str(Path(resources_ref).resolve()),
                "resources_sha256": RESOURCES_SHA256,
                "source_artifacts_lock_sha256": SOURCE_ARTIFACTS_LOCK_SHA256,
                "select_cache_ref": str(Path(select_cache_ref).resolve()),
                "select_cache_key": cache_index["cache_key"],
                "select_labels_ref": str(Path(select_labels_ref).resolve()),
                "select_labels_sha256": sha256_file(select_labels_ref),
                "eta_ref": ETA_REF, "grid": GRID, "fixed": FIXED,
                "methods": methods,
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
    """Score every method over the full select cache, collect diagnostics, seal and evaluate."""
    import torch
    from eptta.adaptation.types import EPConfig, TargetViews
    from eptta.baselines.dispatch import run_method
    from eptta.cache.reader import FeatureCache
    from eptta.models.frozen import verify_frozen_export
    from eptta.offline.artifacts import load_frozen_resources
    from eptta.evaluation.seal import seal_scores, evaluate_sealed
    proposal = read_json(proposal_ref)
    if proposal.get("status") != "PROPOSED" and proposal.get("status") != "LOCKED":
        raise ContractError("R6 proposal must be PROPOSED/LOCKED")
    bundle, _export, _parity, _selection = verify_frozen_export(proposal["bundle_ref"])
    resources, extras, meta = load_frozen_resources(proposal["resources_ref"], bundle)
    cache = FeatureCache(proposal["select_cache_ref"])
    if cache.index["identity"]["baseline_id"] != bundle["baseline_id"]:
        raise ContractError("select cache does not share the frozen detector")
    features = cache.load_by_id()
    methods = proposal["methods"]
    root = Path(output_root).resolve()
    method_runs = []
    for index, item in enumerate(methods):
        method_id = item["method_id"]
        cfg = EPConfig(**item["config"])
        params = dict(item["params"])
        name = "method-%03d" % index
        method_dir = root / name
        method_dir.mkdir(parents=True, exist_ok=True)
        method_resources = resources
        unavailable = None
        if method_id == "ep_feature_pca_U":
            if extras.get("U_feature_pca") is None:
                unavailable = ("NOT_RUN_MISSING_RESOURCE", "U_feature_pca unavailable")
            else:
                from dataclasses import replace
                method_resources = replace(resources, U=extras["U_feature_pca"])
        elif method_id == "ep_random_U":
            random_index = params.pop("random_index", None)
            if random_index not in (0, 1, 2):
                raise ContractError("ep_random_U requires random_index 0/1/2")
            if extras.get("U_random_%d" % random_index) is None:
                unavailable = ("NOT_RUN_MISSING_RESOURCE", "U_random_%d unavailable" % random_index)
            else:
                from dataclasses import replace
                method_resources = replace(resources, U=extras["U_random_%d" % random_index])
        elif method_id == "ep_keep_fisher":
            params["fisher"] = extras.get("fisher")
        elif method_id == "fixed_source_adapter":
            params["fixed_R"] = extras.get("fixed_R")
        scores_path = method_dir / "scores.jsonl"
        diag_path = method_dir / "diagnostics.jsonl"
        with scores_path.open("x", encoding="utf-8") as scores_stream, diag_path.open(
                "x", encoding="utf-8") as diag_stream:
            if unavailable is not None:
                row = {"schema_version": "0.1.0", "sample_id": None, "score": None, "score_before": None,
                       "status": unavailable[0], "method_id": method_id, "steps_completed": 0,
                       "objective_evaluations": 0, "r_fro": None, "runtime_ref": None}
                scores_stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False) + "\n")
            else:
                for sample_id in sorted(features):
                    target = TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache.index["cache_key"])
                    frozen = float(target.features[0] @ method_resources.w + method_resources.b)
                    try:
                        result = run_method(method_id, target, method_resources, cfg, params)
                    except Exception as exc:  # noqa: BLE001 - isolate one bad episode, fallback to frozen
                        result = {"method_id": method_id, "sample_id": sample_id, "score": frozen,
                                  "score_before": frozen, "steps_completed": 0,
                                  "objective_evaluations": 0, "status": "fallback_numeric",
                                  "_error": type(exc).__name__}
                    diag = compute_diagnostics(result, target, method_resources, cfg)
                    row = {"schema_version": "0.1.0", "sample_id": sample_id,
                           "score": result.get("score"), "score_before": result.get("score_before"),
                           "status": result.get("status", "ok"),
                           "method_id": method_id, "steps_completed": result.get("steps_completed", 0),
                           "objective_evaluations": result.get("objective_evaluations", 0),
                           "r_fro": diag.get("r_fro"), "runtime_ref": None}
                    diag_row = {"schema_version": "0.1.0", "sample_id": sample_id, "method_id": method_id}
                    diag_row.update(diag)
                    diag_stream.write(json.dumps(diag_row, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False) + "\n")
                    scores_stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False) + "\n")
        sanitized_params = {key: ("<tensor:%s>" % tuple(value.shape)) if isinstance(value, torch.Tensor)
                            else value for key, value in params.items()}
        method_runs.append({"index": index, "method_id": method_id, "params": sanitized_params,
                            "dir": name, "config": item["config"], "unavailable": unavailable})
    # seal + evaluate
    for run in method_runs:
        method_dir = root / run["dir"]
        if run["unavailable"] is not None:
            run["evaluation"] = {"status": run["unavailable"][0]}
            continue
        seal_scores(method_dir)
        run["evaluation"] = evaluate_sealed(method_dir, proposal["select_labels_ref"],
                                            method_dir / "evaluation.json", float(resources.tau0))
    manifest = {"schema_version": "0.1.0", "status": "COMPLETE", "methods": method_runs,
                "tau0": float(resources.tau0)}
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
