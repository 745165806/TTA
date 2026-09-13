import torch
from torch import Tensor


def run_batch_valid(Z: Tensor, U: Tensor, w: Tensor, b: float,
                    Za: Tensor, ya: Tensor, m0: Tensor, tau0: float,
                    cfg) -> tuple[Tensor, Tensor, Tensor]:
    """Valid finite inputs only; the caller must apply the Appendix A validator.

    Z: [B,N,d]. Each row has its own independent R. Plain projected SGD.
    This is a mathematical reference, not the full logging/failure executor.
    """
    if Z.ndim != 3 or Z.shape[0] < 1 or Z.shape[1] < 2:
        raise ValueError("expected Z=[B,N,d] with B>=1, N>=2")
    B, N, d = Z.shape
    r = U.shape[1]
    a = Z @ U
    ac = a - a.mean(dim=1, keepdim=True)
    zc = Z - Z.mean(dim=1, keepdim=True)
    pc = zc - ac @ U.T
    constant = pc.square().sum(dim=(1, 2)) / (N * d)
    aa = Za @ U
    c = U.T @ w
    sign = 2 * ya.to(Z.dtype) - 1
    before = Z[:, 0] @ w + b
    R = torch.zeros((B, r, r), dtype=Z.dtype,
                    device=Z.device, requires_grad=True)

    with torch.enable_grad():
        for _ in range(cfg.steps):
            adapted_a = a + torch.einsum("bnj,bij->bni", a, R)
            centered = adapted_a - adapted_a.mean(dim=1, keepdim=True)
            lv = constant + centered.square().sum(dim=(1, 2)) / (N * d)
            t = torch.einsum("bji,j->bi", R, c)  # R^T c, per episode
            delta_anchor = torch.einsum("mi,bi->bm", aa, t)
            margin = m0[None, :] + sign[None, :] * delta_anchor
            deficit = torch.relu((1 - cfg.gamma) * m0[None, :] - margin)
            lk = deficit.square().mean(dim=1)
            per_episode = lv + cfg.lambda_keep * lk
            if not bool(torch.isfinite(per_episode).all()):
                raise FloatingPointError("replay this batch serially to isolate failure")
            grad, = torch.autograd.grad(per_episode.sum(), R)
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("non-finite batch gradient: isolate serially")
            with torch.no_grad():
                R.add_(grad, alpha=-cfg.lr)
                norms = torch.linalg.vector_norm(R, dim=(-2, -1), keepdim=True)
                if not bool(torch.isfinite(norms).all()):
                    raise FloatingPointError("non-finite batch norm: isolate serially")
                scale = (cfg.rho / norms.clamp_min(torch.finfo(R.dtype).tiny)).clamp(max=1)
                R.mul_(scale)

    with torch.no_grad():
        t = torch.einsum("bji,j->bi", R, c)
        after = before + (a[:, 0] * t).sum(dim=-1)
        if not bool(torch.isfinite(after).all()):
            raise FloatingPointError("non-finite scores: isolate serially")
    return R.detach().clone(), before.detach(), after.detach()
