"""Contract checks for held-out supervised development diagnostics."""

from types import SimpleNamespace

import numpy as np
import torch

from experiments.capacity_audit.run_ladder import fit_arm, group_stratified_folds, inner_group_split
from experiments.capacity_audit.verify_c1_convex import solve_fold


def test_group_folds_keep_wavefake_content_pairs_together():
    groups = [f"content-{index}" for index in range(25) for _ in range(2)]
    labels = np.array([0, 1] * 25, dtype=np.int64)
    held_out = []
    for train, validation in group_stratified_folds(labels, groups):
        assert not set(np.array(groups)[train]) & set(np.array(groups)[validation])
        assert sorted(np.bincount(labels[validation]).tolist()) == [5, 5]
        held_out.extend(validation.tolist())
    assert sorted(held_out) == list(range(50))


def test_inner_wavefake_epoch_split_keeps_pairs_together():
    groups = [f"content-{index}" for index in range(100) for _ in range(2)]
    labels = np.array([0, 1] * 100, dtype=np.int64)
    train, validation = inner_group_split(np.arange(200), labels, groups, 2026)
    assert not set(np.array(groups)[train]) & set(np.array(groups)[validation])
    assert np.bincount(labels[validation]).tolist() == [10, 10]


def test_supervised_R_is_projected_and_predictions_are_held_out():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(120, 160)).astype(np.float32)
    y = np.array([0, 1] * 60, dtype=np.int64)
    U = torch.from_numpy(np.eye(160, 8, dtype=np.float32))
    w = torch.ones(160, dtype=torch.float32) / 160
    resources = SimpleNamespace(U=U, w=w, b=0.)
    train = np.arange(100)
    heldout = np.arange(100, 120)
    for arm, count in (("C1_supervised_R", 64), ("C2_linear", 161)):
        prediction, detail = fit_arm(arm, X, y, train, heldout, resources, 0)
        assert prediction.shape == (20,)
        assert np.isfinite(prediction).all()
        assert detail["parameter_count"] == count
        assert detail["train_count"] + detail["inner_val_count"] == len(train)
        if arm == "C1_supervised_R":
            assert detail["R_norm"] <= .100001


def test_convex_C1_solver_obeys_same_score_radius():
    rng = np.random.default_rng(2027)
    X = rng.normal(size=(100, 160)).astype(np.float32)
    y = np.array([0, 1] * 50, dtype=np.int64)
    U = torch.from_numpy(np.eye(160, 8, dtype=np.float32))
    w = torch.ones(160, dtype=torch.float32) / 160
    resources = SimpleNamespace(U=U, w=w, b=0.)
    prediction, detail = solve_fold(X, y, np.arange(80), np.arange(80, 100), resources)
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()
    assert detail["success"]
    assert detail["q_norm"] <= detail["radius"] + 1e-7
