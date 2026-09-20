"""Numerical diagnostics retained from the former staged R6 wrappers."""

ETA_REF = 0.01
GRID = {"K": [1, 3, 5], "eta_multiplier": [0.3, 1.0, 3.0]}
ETA_GRID = [0.03, 0.3, 3.0]
K, RHO, GAMMA, LAMBDA_KEEP = 5, 0.2, 0.1, 1.0


def grid_candidates():
    return [{"K": steps, "eta": round(mult * ETA_REF, 6)}
            for steps in GRID["K"] for mult in GRID["eta_multiplier"]]


def compute_diagnostics(result, target, resources, cfg):
    from eptta.adaptation.adapter import adaptation_diagnostics
    diagnostic = adaptation_diagnostics(result, target, resources, cfg)
    diagnostic.update({"r_fro": diagnostic.get("final_R_norm"),
                       "margin_violation_fraction": diagnostic.get("final_margin_violation_fraction"),
                       "projection_triggered": bool(diagnostic.get("projection_count")) if diagnostic.get(
                           "projection_count") is not None else None,
                       "final_keep_loss": diagnostic.get("final_margin_loss"),
                       "worst_margin_change": diagnostic.get("max_abs_margin_change")})
    return diagnostic


def verify_k_gradient(targets, resources, cfg):
    """Independent FP64 reference for first-step gradient plus reset checks."""
    import torch
    from eptta.adaptation.adapter import run_cache_method
    Z = torch.stack([target.features for target in targets])
    U = resources.U
    Q = Z.to(torch.float64) @ U.to(torch.float64)
    Qc = Q - Q.mean(dim=1, keepdim=True)
    count, dimension = Z.shape[1], Z.shape[2]
    reference = (2.0 / (count * dimension)) * (Qc.transpose(-2, -1) @ Qc)
    reports = []
    for index, target in enumerate(targets):
        first = run_cache_method("ep_tta", target, resources, cfg)
        if first["steps_completed"] < 1:
            reports.append({"sample_id": target.sample_id, "status": "no_update", "max_abs_grad_err": None})
            continue
        R = torch.zeros((U.shape[1], U.shape[1]), dtype=Z.dtype, device=Z.device, requires_grad=True)
        from eptta.adaptation.math import apply_adapter, keep_loss, view_loss
        view = view_loss(apply_adapter(Z[index:index + 1], U, R))
        keep = keep_loss(apply_adapter(resources.anchors_z, U, R), resources.w, resources.b,
                         resources.anchors_y, resources.anchors_m0, resources.tau0, cfg.gamma)
        grad, = torch.autograd.grad(view + cfg.lambda_keep * keep, R)
        error = float((grad.to(torch.float64) - reference[index]).abs().max())
        reports.append({"sample_id": target.sample_id, "status": first["status"],
                        "steps_completed": first["steps_completed"], "max_abs_grad_err": error,
                        "score_after": first["score"], "score_before": first["score_before"],
                        "r_fro_after": float(torch.linalg.vector_norm(first["R"]))})
    return {"reference_grad": reference.detach().cpu().numpy().tolist(), "per_sample": reports,
            "max_abs_grad_err": max((row["max_abs_grad_err"] for row in reports
                                     if row["max_abs_grad_err"] is not None), default=0.0)}


def d1_methods():
    methods = [
        {"method_id": "frozen", "config": {"steps": 0, "lr": 0.01, "rho": RHO,
         "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP}, "params": {}},
        {"method_id": "multiview_mean", "config": {"steps": 0, "lr": 0.01, "rho": RHO,
         "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP}, "params": {}},
    ]
    for eta in ETA_GRID:
        for method_id in ("ep_tta", "ep_no_keep"):
            methods.append({"method_id": method_id, "config": {"steps": K, "lr": eta,
                            "rho": RHO, "gamma": GAMMA, "lambda_keep": LAMBDA_KEEP}, "params": {}})
    return methods


def analyze_keep_reachability(resources, rho, gamma):
    U, w = resources.U, resources.w
    q, c = resources.anchors_z @ U, U.T @ w
    gradient_norm = c.norm() * q.norm(dim=1)
    bound = rho * gradient_norm
    threshold = gamma * resources.anchors_m0
    reachable = bound > threshold
    rows = [{"anchor": index, "y": int(resources.anchors_y[index]),
             "m0": float(resources.anchors_m0[index]), "gradient_norm": float(gradient_norm[index]),
             "bound": float(bound[index]), "threshold": float(threshold[index]),
             "reachable": bool(reachable[index])}
            for index in range(resources.anchors_z.shape[0])]
    return {"rho": float(rho), "gamma": float(gamma), "c_norm": float(c.norm()),
            "reachable_count": int(reachable.sum()), "total": len(rows),
            "min_ratio_threshold_to_bound": float((threshold / bound.clamp_min(1e-30)).min()),
            "anchors": rows}
