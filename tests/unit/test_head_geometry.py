"""Head geometry and selected-feature I/O contract tests."""

import json
from types import SimpleNamespace

import numpy as np
import torch

from eptta.adaptation.math import apply_adapter
from experiments.head_capacity_geometry.resources import sparse_selected_views
from experiments.head_capacity_geometry.run_geometry import fit_h0_h1


def test_sparse_reader_returns_only_selected_feature_rows(tmp_path):
    (tmp_path / "chunks").mkdir()
    rows = np.arange(4*3*160, dtype=np.float32).reshape(4, 3, 160)
    np.save(tmp_path / "chunks/first.npy", rows, allow_pickle=False)
    (tmp_path / "chunks/first.ids.json").write_text(json.dumps(["holdout-a", "dev-b",
                                                               "holdout-c", "dev-d"]))
    cache = SimpleNamespace(root=tmp_path, index={"chunks": [{"array_ref":"chunks/first.npy",
        "ids_ref":"chunks/first.ids.json", "count":4, "shape":[4,3,160]}]})
    selected, metadata_count = sparse_selected_views(cache, ["dev-d", "dev-b"])
    assert selected.shape == (2, 3, 160)
    assert metadata_count == 4
    np.testing.assert_array_equal(selected[0], rows[3])
    np.testing.assert_array_equal(selected[1], rows[1])


def test_bias_and_positive_scale_preserve_within_fold_ranking():
    score = np.array([-3., -2., -1., 0., 1., 2., 3., 4.])
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    offset, scale, b = fit_h0_h1(score, labels, np.arange(len(score)))
    assert scale > 0
    assert np.array_equal(np.argsort(score), np.argsort(score + offset))
    assert np.array_equal(np.argsort(score), np.argsort(scale*score + b))


def test_adapter_score_equals_derived_effective_head():
    generator = torch.Generator().manual_seed(2026)
    z = torch.randn((7, 160), generator=generator)
    u, _ = torch.linalg.qr(torch.randn((160, 8), generator=generator))
    r = torch.randn((8, 8), generator=generator)*.01
    w = torch.randn((160,), generator=generator)
    bias = .3
    direct = apply_adapter(z, u, r)@w+bias
    effective = z@(w+u@r.T@(u.T@w))+bias
    assert torch.allclose(direct, effective, atol=1e-5, rtol=0)
