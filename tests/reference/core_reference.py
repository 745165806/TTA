import math
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class CoreConfig:
    steps: int = 3
    lr: float = 0.01
    rho: float = 0.2
    gamma: float = 0.1
    lambda_keep: float = 1.0

@dataclass
class CoreOutput:
    R: Tensor
    score_before: float
    score_after: float
    status: str
    reason: Optional[str]
    trace: list[dict[str, float]]


def apply_adapter(Z: Tensor, U: Tensor, R: Tensor) -> Tensor:
    """Row features: [N,d]; mathematical R uses the column-vector convention."""
    return Z + ((Z @ U) @ R.T) @ U.T


def view_loss(Z_adapted: Tensor) -> Tensor:
    centered = Z_adapted - Z_adapted.mean(dim=0, keepdim=True)
    return centered.square().mean()  # denominator = number_of_views * d


def keep_loss(Za_adapted: Tensor, w: Tensor, b: float, ya: Tensor,
              m0: Tensor, tau0: float, gamma: float) -> Tensor:
    margin = (2 * ya.to(Za_adapted.dtype) - 1) * (Za_adapted @ w + b - tau0)
    deficit = torch.relu((1 - gamma) * m0 - margin)
    return deficit.square().mean()


@torch.no_grad()
def project_frobenius_(R: Tensor, rho: float) -> None:
    norm = torch.linalg.vector_norm(R)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("non-finite R norm")
    if float(norm) > rho:
        R.mul_(rho / norm)  # this branch cannot divide by zero for rho > 0


def run_core_episode(Z: Tensor, U: Tensor, w: Tensor, b: float,
                     Za: Tensor, ya: Tensor, m0: Tensor, tau0: float,
                     cfg: CoreConfig) -> CoreOutput:
    """Target Z contains no labels. All adaptation state is local to this call."""
    if Z.ndim != 2 or U.ndim != 2 or Za.ndim != 2 or w.ndim != 1:
        raise ValueError("invalid tensor ranks")
    n, d = Z.shape
    r = U.shape[1]
    m = Za.shape[0]
    if n < 2 or not 1 <= r <= d or m == 0:
        raise ValueError("invalid numbers of views, rank, or anchors")
    if U.shape[0] != d or Za.shape[1] != d or w.shape != (d,):
        raise ValueError("embedding/head dimensions disagree")
    if ya.shape != (m,) or m0.shape != (m,):
        raise ValueError("anchor arrays must be 1-D; refuse broadcasting")
    scalars = (b, tau0, cfg.lr, cfg.rho, cfg.gamma, cfg.lambda_keep)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("configuration and head scalars must be finite")
    if not isinstance(cfg.steps, int):
        raise ValueError("steps must be an integer")
    if not (cfg.steps >= 0 and cfg.lr > 0 and 0 < cfg.rho < 1
            and 0 <= cfg.gamma < 1 and cfg.lambda_keep >= 0):
        raise ValueError("invalid adaptation configuration")
    for t in (Z, U, w, Za, m0):
        if t.dtype != Z.dtype or t.device != Z.device:
            raise ValueError("floating tensors must share dtype and device")
        if t.requires_grad or not bool(torch.isfinite(t).all()):
            raise ValueError("context must be detached and finite")
    if ya.device != Z.device or not bool(((ya == 0) | (ya == 1)).all()):
        raise ValueError("invalid source labels")
    eye = torch.eye(r, dtype=U.dtype, device=U.device)
    if not torch.allclose(U.T @ U, eye, atol=1e-5, rtol=1e-5):
        raise ValueError("U is not orthonormal")
    expected_m0 = (2 * ya.to(Z.dtype) - 1) * (Za @ w + b - tau0)
    if not bool((m0 > 0).all()) or not torch.allclose(
            m0, expected_m0, atol=1e-5, rtol=1e-5):
        raise ValueError("anchors are invalid or belong to a different head/threshold")

    R = nn.Parameter(torch.zeros((r, r), dtype=Z.dtype, device=Z.device))
    optimizer = torch.optim.SGD([R], lr=cfg.lr, momentum=0.0, weight_decay=0.0)
    before = float(Z[0] @ w + b)
    if not math.isfinite(before):
        raise ValueError("invalid frozen score")
    trace: list[dict[str, float]] = []
    status, reason = "ok", None

    with torch.enable_grad():
        for k in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            lv = view_loss(apply_adapter(Z, U, R))
            lk = keep_loss(apply_adapter(Za, U, R), w, b, ya, m0, tau0, cfg.gamma)
            total = lv + cfg.lambda_keep * lk
            if not bool(torch.isfinite(total)):
                status, reason = "fallback_numeric", "non-finite loss"
                break
            total.backward()
            if R.grad is None or not bool(torch.isfinite(R.grad).all()):
                status, reason = "fallback_numeric", "non-finite or missing R gradient"
                break
            grad_norm = float(torch.linalg.vector_norm(R.grad))
            if not math.isfinite(grad_norm):
                status, reason = "fallback_numeric", "non-finite gradient norm"
                break
            optimizer.step()
            if not bool(torch.isfinite(R).all()):
                status, reason = "fallback_numeric", "non-finite parameter"
                break
            try:
                project_frobenius_(R, cfg.rho)
            except FloatingPointError:
                status, reason = "fallback_numeric", "non-finite projection norm"
                break
            trace.append({"step": float(k + 1), "loss_view_before": float(lv.detach()),
                          "loss_keep_before": float(lk.detach()), "grad_norm": grad_norm,
                          "r_fro_after": float(torch.linalg.vector_norm(R.detach()))})

    with torch.no_grad():
        after = float((apply_adapter(Z[0:1], U, R) @ w + b)[0])
        if not math.isfinite(after):
            status, reason = "fallback_numeric", "non-finite final score"
        if status != "ok":
            R.zero_()
            after = before
    return CoreOutput(R.detach().clone(), before, after, status, reason, trace)
