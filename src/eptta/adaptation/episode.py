"""Serial production reference: fresh R per call, plain SGD, fixed input views."""
import math

import torch

from eptta.adaptation.math import apply_adapter, keep_loss, margin_deficit, project_frobenius_, view_loss
from eptta.adaptation.types import EPConfig, EpisodeOutput, TargetViews
from eptta.adaptation.validation import validate_inputs


def finite(tensor, reason):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(reason)
    return tensor


@torch.no_grad()
def finish(target, resources, cfg, R, before, trace, steps_completed, reason=None):
    """Recompute diagnostics after the final update. Never select a step by them."""
    Z, U = target.features, resources.U
    lv = lk = violations = None
    after = before
    if reason is None:
        try:
            # K=0 is the frozen path by contract. Reusing the already validated
            # scalar also avoids a zero-matmul roundoff changing its exact value.
            after = before if steps_completed == 0 else float(finite(
                (apply_adapter(Z[:1], U, R) @ resources.w + resources.b)[0], "non-finite final score"))
            final_view = finite(view_loss(apply_adapter(Z, U, R)), "non-finite final view loss")
            deficits = margin_deficit(apply_adapter(resources.anchors_z, U, R), resources.w,
                                      resources.b, resources.anchors_y, resources.anchors_m0,
                                      resources.tau0, cfg.gamma)
            final_keep = finite(deficits.square().mean(), "non-finite final keep loss")
            lv, lk = float(final_view), float(final_keep)
            violations = float((deficits > 0).to(Z.dtype).mean())
        except FloatingPointError as exc:
            reason = str(exc)
    if reason is not None:
        R = torch.zeros_like(R)
        after = before
        lv = lk = violations = None  # failed diagnostics must not become fabricated finite values
    return EpisodeOutput(target.sample_id, target.feature_artifact_id, resources.artifact_bundle_id,
                         R.detach().clone(), before, after, "fallback_numeric" if reason else "ok",
                         reason, steps_completed, lv, lk, violations,
                         float(torch.linalg.vector_norm(R)), tuple(trace))


def run_episode(target, resources, cfg=EPConfig()):
    before = validate_inputs(target, resources, cfg)
    Z, U = target.features, resources.U
    R = torch.zeros((U.shape[1], U.shape[1]), dtype=Z.dtype, device=Z.device, requires_grad=True)
    trace, completed, reason = [], 0, None
    with torch.enable_grad():
        for k in range(cfg.steps):
            try:
                lv = view_loss(apply_adapter(Z, U, R))
                lk = keep_loss(apply_adapter(resources.anchors_z, U, R), resources.w, resources.b,
                               resources.anchors_y, resources.anchors_m0, resources.tau0, cfg.gamma)
                total = finite(lv + cfg.lambda_keep * lk, "non-finite loss")
                grad, = torch.autograd.grad(total, R)
                finite(grad, "non-finite R gradient")
                grad_norm = finite(torch.linalg.vector_norm(grad), "non-finite gradient norm")
                with torch.no_grad():
                    R.add_(grad, alpha=-cfg.lr)  # SGD: no momentum, no weight decay, no other terms
                    finite(R, "non-finite parameter")
                    project_frobenius_(R, cfg.rho)
                completed = k + 1
                trace.append({"step": float(completed), "loss_view_before": float(lv.detach()),
                              "loss_keep_before": float(lk.detach()), "grad_norm": float(grad_norm),
                              "r_fro_after": float(torch.linalg.vector_norm(R.detach()))})
            except FloatingPointError as exc:
                reason = str(exc)
                break
    return finish(target, resources, cfg, R, before, trace, completed, reason)


def run_k0_episode(target, head_w, head_b, artifact_bundle_id):
    """Resource-free K=0 production path.

    K=0 is the legal identity short-circuit: R stays zero, no U/M/tau0 are
    required, and the score is the original-view z0 frozen head score (never a
    three-view mean and never a zero-matmul round trip).  This mirrors the
    ``run_episode(steps=0)`` contract without loading :class:`FrozenResources`.
    """
    if type(target) is not TargetViews:
        raise ValueError("only typed TargetViews are accepted by the K=0 entrypoint")
    if torch.is_inference_mode_enabled():
        raise ValueError("EP requires autograd for R; inference_mode is not supported")
    for name, value in (("sample_id", target.sample_id), ("feature_artifact_id", target.feature_artifact_id),
                        ("artifact_bundle_id", artifact_bundle_id)):
        if type(value) is not str or not value:
            raise ValueError("opaque sample/artifact IDs are required")
    Z = target.features
    if not isinstance(Z, torch.Tensor) or Z.ndim != 2 or Z.shape[0] != 3:
        raise ValueError("K=0 requires exactly three fixed views [N=3, d]")
    if Z.dtype not in (torch.float32, torch.float64):
        raise ValueError("EP supports float32/float64 only")
    if Z.requires_grad or not bool(torch.isfinite(Z).all()):
        raise ValueError("K=0 requires detached, finite views")
    d = Z.shape[1]
    if not isinstance(head_w, torch.Tensor) or head_w.ndim != 1 or head_w.shape != (d,):
        raise ValueError("K=0 requires a detached frozen head weight [d]")
    if head_w.dtype != Z.dtype or head_w.device != Z.device or head_w.requires_grad:
        raise ValueError("K=0 head must share view dtype/device and be detached")
    if type(head_b) not in (int, float) or not math.isfinite(head_b):
        raise ValueError("frozen head bias must be a finite scalar")
    z0 = Z[0]  # original view, not a view mean
    before = float(z0 @ head_w + head_b)
    if not math.isfinite(before):
        raise FloatingPointError("non-finite K=0 original-view score")
    R = torch.zeros((0, 0), dtype=Z.dtype, device=Z.device)
    return EpisodeOutput(target.sample_id, target.feature_artifact_id, artifact_bundle_id,
                         R, before, before, "no_adaptation", None, 0, None, None, None, 0.0, ())
