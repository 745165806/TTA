"""Independent low-dimensional episodes; sum gradients, per-matrix projection."""
import time

import torch

from eptta.adaptation.episode import finite, finish, run_episode
from eptta.adaptation.math import project_frobenius_
from eptta.adaptation.types import BatchOutput, EPConfig
from eptta.adaptation.validation import validate_inputs


def run_batch(targets, resources, cfg=EPConfig()):
    targets = tuple(targets)
    if not targets:
        raise ValueError("empty episode batch")
    # All malformed inputs are contract errors, not numeric fallback.
    before = [validate_inputs(t, resources, cfg) for t in targets]
    shape = targets[0].features.shape
    if any(t.features.shape != shape for t in targets):
        raise ValueError("batch views must share dimensions")
    Z = torch.stack([t.features for t in targets])
    U, w = resources.U, resources.w
    B, N, d = Z.shape
    r = U.shape[1]
    R = torch.zeros((B, r, r), dtype=Z.dtype, device=Z.device, requires_grad=True)
    Q = Z @ U
    Qc = Q - Q.mean(dim=1, keepdim=True)
    Zc = Z - Z.mean(dim=1, keepdim=True)
    constant = (Zc - Qc @ U.T).square().sum(dim=(1, 2)) / (N * d)
    Qa, c = resources.anchors_z @ U, U.T @ w
    signs = 2 * resources.anchors_y.to(Z.dtype) - 1
    traces = [[] for _ in targets]
    try:
        with torch.enable_grad():
            for k in range(cfg.steps):
                coords = Q + torch.einsum("bnj,bij->bni", Q, R)
                centered = coords - coords.mean(dim=1, keepdim=True)
                lv = constant + centered.square().sum(dim=(1, 2)) / (N * d)
                rt_c = torch.einsum("bji,j->bi", R, c)
                margins = resources.anchors_m0[None, :] + signs[None, :] * (rt_c @ Qa.T)
                deficit = torch.relu((1 - cfg.gamma) * resources.anchors_m0[None, :] - margins)
                lk = deficit.square().mean(dim=1)
                loss = finite(lv + cfg.lambda_keep * lk, "non-finite batch loss")
                grad, = torch.autograd.grad(loss.sum(), R)
                finite(grad, "non-finite batch gradient")
                norms = finite(torch.linalg.vector_norm(grad, dim=(-2, -1)), "non-finite batch gradient norm")
                with torch.no_grad():
                    R.add_(grad, alpha=-cfg.lr)
                    finite(R, "non-finite batch parameter")
                    project_frobenius_(R, cfg.rho)
                    rnorms = torch.linalg.vector_norm(R, dim=(-2, -1))
                for i in range(B):
                    traces[i].append({"step": float(k + 1), "loss_view_before": float(lv[i].detach()),
                                      "loss_keep_before": float(lk[i].detach()), "grad_norm": float(norms[i]),
                                      "r_fro_after": float(rnorms[i])})
        results = tuple(finish(t, resources, cfg, R[i], before[i], traces[i], cfg.steps) for i, t in enumerate(targets))
        if any(o.status != "ok" for o in results):
            raise FloatingPointError("final batch diagnostics failed")
        return BatchOutput(results)
    except FloatingPointError:
        # Replay unchanged inputs/config serially; one bad episode cannot discard peers.
        started = time.perf_counter()
        results = tuple(run_episode(t, resources, cfg) for t in targets)
        return BatchOutput(results, len(targets), time.perf_counter() - started)
