import torch
import pytest

from eptta.offline.anchors import build_anchor_memory
from eptta.offline.subspace import balanced_response_subspace


def _views(d, groups, labels, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for group, label in zip(groups, labels):
        z0 = torch.randn(d, generator=gen)
        # distinct treatment responses so the second moment is full-rank-ish
        noise = z0 + 0.1 * torch.randn(d, generator=gen)
        fir = z0 - 0.05 * torch.randn(d, generator=gen)
        rows.append(torch.stack([z0, noise, fir]))
    return torch.stack(rows)


def test_balanced_response_subspace_shapes_and_orthonormal():
    d, rank = 12, 4
    groups = []
    labels = []
    for label in (0, 1):
        for g in range(4):
            for _ in range(10):
                groups.append("g%d_%d" % (label, g))
                labels.append(label)
    views = _views(d, groups, labels, seed=1)
    labels_t = torch.tensor(labels)
    U, diag = balanced_response_subspace(views, labels_t, groups, rank, ("noise", "fir"),
                                         samples_per_group=8, pair_seed=13)
    assert U.shape == (d, rank)
    assert torch.allclose(U.T @ U, torch.eye(rank), atol=1e-5, rtol=1e-5)
    assert set(diag["cells"]) == {"0:noise", "0:fir", "1:noise", "1:fir"}
    assert diag["accumulator_dtype"] == "torch.float64"


def test_balanced_response_subspace_rejects_missing_cell():
    d, rank = 8, 2
    groups = ["g0"] * 8 + ["g1"] * 8
    labels = [0] * 16  # only one class present
    views = _views(d, groups, labels)
    with pytest.raises(ValueError, match="missing cell"):
        balanced_response_subspace(views, torch.tensor(labels), groups, rank, ("noise", "fir"),
                                   samples_per_group=4, pair_seed=13)


def test_balanced_response_subspace_rejects_undersized_group():
    d, rank = 8, 2
    groups = ["g0"] * 4 + ["g1"] * 8
    labels = [0] * 12
    views = _views(d, groups, labels)
    with pytest.raises(ValueError, match="fewer than samples_per_group"):
        balanced_response_subspace(views, torch.tensor(labels), groups, rank, ("noise", "fir"),
                                   samples_per_group=6, pair_seed=13)


def test_anchor_memory_stratifies_margins_across_bins():
    torch.manual_seed(0)
    d = 6
    # 64 bonafide features with a spread of margins; 64 spoof likewise.
    def make(label):
        z = torch.randn(96, d)
        w = torch.randn(d)
        b = 0.0
        return z, torch.full((96,), label)
    z0, y0 = make(0)
    z1, y1 = make(1)
    features = torch.cat([z0, z1])
    labels = torch.cat([y0, y1])
    w = torch.randn(d)
    b = 0.0
    tau0 = 0.0
    memory = build_anchor_memory(features, labels, w, b, tau0, per_class=32, seed=13,
                                 margin_bins=4, margin_epsilon=1e-6)
    assert memory["anchors_z"].shape == (64, d)
    assert memory["anchors_y"].shape == (64,)
    assert int((memory["anchors_y"] == 0).sum()) == 32
    assert int((memory["anchors_y"] == 1).sum()) == 32
    assert bool((memory["anchors_m0"] > 1e-6).all())
    # margins must span more than one quantile bin (not all easiest/hardest)
    q = torch.quantile(memory["anchors_m0"], torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert q[0] < q[-1]
    assert bool((memory["anchors_m0"].min() < q[2]) and (memory["anchors_m0"].max() > q[2]))


def test_anchor_memory_preserves_count_when_quota_remainder_hits_small_bin():
    features = torch.tensor([[float(i)] for i in range(1, 6)] +
                            [[-float(i)] for i in range(1, 6)])
    labels = torch.tensor([1] * 5 + [0] * 5)
    memory = build_anchor_memory(features, labels, torch.tensor([1.0]), 0.0, 0.0,
                                 per_class=4, seed=13, margin_bins=3)
    assert len(memory["indices"]) == 8
    assert int((memory["anchors_y"] == 0).sum()) == 4
    assert int((memory["anchors_y"] == 1).sum()) == 4
