import pytest
import torch

from eptta.adaptation.math import apply_adapter
from experiments.ep_capacity_audit.run_geometry import geometry


def test_score_capacity_bound_matches_reachable_rank_one_change():
    generator = torch.Generator().manual_seed(731)
    U, _ = torch.linalg.qr(torch.randn(12, 8, generator=generator))
    Z = torch.randn(3, 12, generator=generator)
    w = torch.randn(12, generator=generator)
    rho = 0.2

    measured = geometry(Z, U, w, rho)
    c = U.T @ w
    q = Z[0] @ U
    expected = rho * float(torch.linalg.vector_norm(c) * torch.linalg.vector_norm(q))
    R = rho * torch.outer(c / torch.linalg.vector_norm(c), q / torch.linalg.vector_norm(q))
    actual = float((apply_adapter(Z, U, R)[0] - Z[0]) @ w)

    assert measured["score_change_bound"] == pytest.approx(expected, rel=1e-6)
    assert measured["score_gradient_norm"] == pytest.approx(expected / rho, rel=1e-6)
    assert actual == pytest.approx(expected, rel=1e-5)
