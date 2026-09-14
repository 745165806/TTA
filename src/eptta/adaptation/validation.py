import math

import torch

from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews


def validate_inputs(target, resources, cfg):
    if type(target) is not TargetViews or type(resources) is not FrozenResources or type(cfg) is not EPConfig:
        raise ValueError("only typed target views, frozen resources and EPConfig are accepted")
    if torch.is_inference_mode_enabled():
        raise ValueError("EP requires autograd for R; inference_mode is not supported")
    if not all(type(v) is str and v for v in (target.sample_id, target.feature_artifact_id, resources.artifact_bundle_id)):
        raise ValueError("opaque sample and artifact IDs are required")
    if resources.schema_version != "0.1.0" or resources.source_role != "fit":
        raise ValueError("fixed source anchors must come from fit with schema 0.1.0")
    if resources.channel not in ("synthetic_test", "frozen_bundle"):
        raise ValueError("resources must come from synthetic tests or a finalized frozen bundle loader")
    Z, U, w, Za, ya, m0, s0 = (target.features, resources.U, resources.w,
                                resources.anchors_z, resources.anchors_y,
                                resources.anchors_m0, resources.anchors_s0)
    if any(not isinstance(t, torch.Tensor) for t in (Z, U, w, Za, ya, m0, s0)):
        raise ValueError("all feature/context arrays must be tensors")
    if Z.ndim != 2 or U.ndim != 2 or Za.ndim != 2 or w.ndim != 1:
        raise ValueError("invalid tensor ranks")
    n, d = Z.shape
    m, r = Za.shape[0], U.shape[1]
    if n != 3 or not 1 <= r <= d or m < 2 or U.shape[0] != d or Za.shape[1] != d or w.shape != (d,):
        raise ValueError("expected three fixed views, compatible dimensions and nonempty source anchors")
    if any(t.shape != (m,) for t in (ya, m0, s0)):
        raise ValueError("anchor arrays must be 1-D; broadcasting is forbidden")
    if Z.dtype not in (torch.float32, torch.float64):
        raise ValueError("EP supports float32/float64 only")
    for t in (Z, U, w, Za, m0, s0):
        if t.dtype != Z.dtype or t.device != Z.device or t.requires_grad or not bool(torch.isfinite(t).all()):
            raise ValueError("features/context must be detached, finite and share dtype/device")
    if ya.requires_grad or ya.device != Z.device or ya.dtype == torch.bool or not bool(((ya == 0) | (ya == 1)).all()):
        raise ValueError("source labels must be canonical 0/1")
    if int((ya == 0).sum()) != int((ya == 1).sum()):
        raise ValueError("source anchors must be balanced across both classes")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (resources.b, resources.tau0)):
        raise ValueError("head bias/threshold must be finite scalars")
    if not torch.allclose(U.T @ U, torch.eye(r, dtype=Z.dtype, device=Z.device), atol=1e-5, rtol=1e-5):
        raise ValueError("U must be orthonormal")
    expected_s0 = Za @ w + resources.b
    expected_m0 = (2 * ya.to(Z.dtype) - 1) * (expected_s0 - resources.tau0)
    if not bool((m0 > 1e-6).all()) or not torch.allclose(m0, expected_m0, atol=1e-5, rtol=1e-5) or not torch.allclose(s0, expected_s0, atol=1e-5, rtol=1e-5):
        raise ValueError("anchors do not match the frozen head/threshold")
    before = float(Z[0] @ w + resources.b)
    if not math.isfinite(before):
        raise ValueError("invalid original frozen score; no valid prediction can be emitted")
    return before
