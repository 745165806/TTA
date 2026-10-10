import pytest
import torch

from eptta.offline.task_subspaces import source_mixed_subspace, source_task_subspace


def fixture():
    generator = torch.Generator().manual_seed(8105)
    w = torch.randn(160, generator=generator)
    z = torch.randn(40, 160, generator=generator)
    y = torch.arange(40) % 2
    z[y == 1] += 0.2 * w
    original = torch.linalg.qr(torch.randn(160, 8, generator=generator))[0]
    return original, w, z, y


def test_source_task_and_mixed_geometry():
    original, w, z, y = fixture()
    task = source_task_subspace(w, z, y)
    mixed = source_mixed_subspace(original, w, z, y)
    for U in (task, mixed):
        assert U.shape == (160, 8)
        torch.testing.assert_close(U.T @ U, torch.eye(8), atol=1e-6, rtol=1e-6)
        assert float(torch.linalg.vector_norm(U.T @ w) / torch.linalg.vector_norm(w)) > 0.999999
    torch.testing.assert_close(source_task_subspace(w, z, y), task)
    torch.testing.assert_close(source_mixed_subspace(original, w, z, y), mixed)
    assert not torch.allclose(task @ task.T, mixed @ mixed.T)


def test_rejects_invalid_source_inputs():
    original, w, z, y = fixture()
    with pytest.raises(ValueError):
        source_task_subspace(w, z, torch.zeros_like(y))
    with pytest.raises(ValueError):
        source_task_subspace(w, z.nan_to_num().fill_(float("nan")), y)
    with pytest.raises(ValueError):
        source_mixed_subspace(original * 2, w, z, y)
