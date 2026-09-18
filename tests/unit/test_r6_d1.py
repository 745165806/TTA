import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.types import FrozenResources
from eptta.execution.r6_d1 import analyze_keep_reachability, d1_methods


def _resources():
    generator = torch.Generator().manual_seed(3)
    d, r = 8, 3
    U = torch.linalg.qr(torch.randn(d, r, generator=generator))[0]
    w = torch.randn(d, generator=generator)
    labels = torch.arange(10) % 2
    anchors = torch.randn(10, d, generator=generator)
    b, tau = 0.0, 0.0
    signs = 2 * labels.to(torch.float32) - 1
    anchors += ((tau + signs * 0.5 - anchors @ w - b) / w.square().sum())[:, None] * w
    scores = anchors @ w + b
    margins = signs * (scores - tau)
    return FrozenResources(U, w, b, anchors, labels, margins, scores, tau, "bundle")


def test_d1_methods_is_eight_rows():
    methods = d1_methods()
    assert len(methods) == 8
    ids = [m["method_id"] for m in methods]
    assert ids[:2] == ["frozen", "multiview_mean"]
    assert ids.count("ep_tta") == 3
    assert ids.count("ep_no_keep") == 3


def test_keep_reachability_bound_matches_geometry():
    resources = _resources()
    report = analyze_keep_reachability(resources, 0.2, 0.1)
    assert report["total"] == 10
    # bound = rho * ||c|| * ||q_i|| must be non-negative and finite
    for anchor in report["anchors"]:
        assert anchor["bound"] >= 0.0 and anchor["m0"] > 0.0
    # reachability is affine: bound > gamma*m0 iff can possibly trigger
    for anchor in report["anchors"]:
        assert anchor["reachable"] == (anchor["bound"] > anchor["threshold"])


def test_keep_reachability_reports_nonreachable_when_rho_small():
    resources = _resources()
    report = analyze_keep_reachability(resources, 1e-9, 0.1)
    assert report["reachable_count"] == 0
