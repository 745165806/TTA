"""Synthetic-only engineering tests; run with the project's tta environment."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "experiments/oracle_diagnosis"
sys.path.insert(0, str(DIRECTORY))
import common as oracle
import aggregate as analysis


def test_grid_and_partition():
    rows = oracle.candidates()
    assert len(rows) == len({r["candidate_id"] for r in rows}) == 113
    assert sum(r["K"] == 0 for r in rows) == 1
    assert rows[0]["lr"] is rows[0]["rho"] is None
    for k in (1, 3, 5, 10):
        assert sum(r["K"] == k for r in rows) == 28
    groups = [{r["candidate_id"] for r in oracle.partition(rows, i, 4)} for i in range(4)]
    assert set.union(*groups) == {r["candidate_id"] for r in rows}
    assert sorted(map(len, groups)) == [28, 28, 28, 29]
    for i in range(4):
        for j in range(i):
            assert not groups[i] & groups[j]
    with pytest.raises(ValueError):
        oracle.partition(rows, 4, 4)


@pytest.mark.parametrize("field,bad", [("EER", .2), ("AUC", .7), ("K", 3), ("lr", .03), ("rho", .4)])
def test_tie_break(field, bad):
    best = dict(EER=.1, AUC=.8, K=1, lr=.01, rho=.2)
    other = {**best, field: bad}
    assert min([other, best], key=oracle.oracle_key) == best
    if field != "EER":
        other["EER"] = .09
        assert min([best, other], key=oracle.oracle_key) == other
    assert oracle.selection_regret({"EER": .12}, best) == pytest.approx(.02)


def test_spearman_ties_constant_and_direction():
    assert oracle.ranks([3, 1, 1, 2]) == [4, 1.5, 1.5, 3]
    assert oracle.spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1)
    assert oracle.spearman([1, 1, 1], [1, 2, 3]) is None
    from scipy.stats import spearmanr
    x, y = [1, 2, 2, 4, 5], [2, 1, 4, 4, 3]
    assert oracle.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
    with pytest.raises(ValueError):
        oracle.spearman([1, float("nan")], [1, 2])


def test_folds_stratified_repeatable_and_order_independent():
    ids = [f"sample_{i:03d}" for i in range(53)]
    labels = [0]*31 + [1]*22
    folds = oracle.stratified_folds(ids, labels)
    assert folds == oracle.stratified_folds(ids, labels)
    assert sorted(i for f in folds for i in f) == list(range(53))
    for label in (0, 1):
        counts = [sum(labels[i] == label for i in f) for f in folds]
        assert max(counts) - min(counts) <= 1
    reversed_folds = oracle.stratified_folds(ids[::-1], labels[::-1])
    assert [{ids[i] for i in f} for f in folds] == [{ids[::-1][i] for i in f} for f in reversed_folds]


def test_confirmation_selects_only_training_folds():
    ids = [f"s{i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]
    validation = set(oracle.stratified_folds(ids, labels)[0])
    rows = oracle.candidates()[:2]
    # Adapted candidate perfect in fold0, reversed on its training split.
    scores = {rows[0]["candidate_id"]: [0.]*20,
              rows[1]["candidate_id"]: [float(y if i in validation else 1-y) for i, y in enumerate(labels)]}
    folds, assignments, cv = analysis.confirmation(rows, scores, ids, labels, .5)
    assert folds[0]["selected_K"] == 0
    assert folds[0]["train_EER"] == .5
    assert len(assignments) == len({r["sample_id"] for r in assignments}) == 20
    expected = {r["sample_id"]: r["score"] for r in assignments}
    assert cv == analysis.metrics([expected[i] for i in ids], labels, .5)


def synthetic_sweep(root, ids=None):
    (root / "scores").mkdir()
    ids = ["a", "b"] if ids is None else ids
    for g in range(4):
        rows = []
        for candidate in oracle.partition(oracle.candidates(), g, 4):
            row = {**candidate, **{k: 0. for k in oracle.SIGNALS},
                   "margin_guard_activation_rate": 0., "margin_guard_backtrack_count": 0,
                   "margin_guard_revert_rate": 0., "numeric_failures": 0, "numeric_fallback_count": 0}
            rows.append(row)
            records = [{"sample_id": sid, "score": float(i), "score_before": float(i)} for i, sid in enumerate(ids)]
            (root / "scores" / (candidate["candidate_id"] + ".jsonl")).write_text(
                "\n".join(json.dumps(r) for r in records), encoding="utf-8")
        oracle.write_json(root / f"group_{g}.json", dict(group=g, groups=4, sample_count=len(ids),
                          target_labels_read=False, provenance={"tau0": .5}, candidates=rows))
    return ids


def test_full_score_validation_and_no_labels_before_complete(tmp_path, monkeypatch):
    ids = synthetic_sweep(tmp_path)
    assert len(analysis.complete_scores(tmp_path, ids, 4)[0]) == 113
    (tmp_path / "scores/k0_frozen.jsonl").unlink()
    monkeypatch.setattr(analysis, "sample_ids", lambda: ids)
    def forbidden(*args):
        pytest.fail("target labels opened before complete score validation")
    monkeypatch.setattr(analysis, "target_labels", forbidden)
    with pytest.raises(ValueError, match="score file coverage"):
        analysis.aggregate(tmp_path, 4)


def test_synthetic_end_to_end_report(tmp_path, monkeypatch):
    ids = synthetic_sweep(tmp_path, [f"s{i:02d}" for i in range(20)])
    monkeypatch.setattr(analysis, "sample_ids", lambda: ids)
    monkeypatch.setattr(analysis, "target_labels", lambda _: [0]*10 + [1]*10)
    monkeypatch.setattr(analysis, "legacy_selected", lambda _: None)
    summary = analysis.aggregate(tmp_path, 4)
    assert summary["oracle_best"]["K"] == 0
    assert summary["oracle_gain_EER"]["absolute"] == 0
    assert summary["cross_validated_oracle_EER"] == 0
    assert summary["unsupervised_selected"] == "unavailable"
    assert summary["target90_labels_read"] is False
    assert all(value is None for value in summary["spearman_vs_negative_EER"].values())
    import csv
    with (tmp_path / "analysis/oracle_surface.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 113
    with (tmp_path / "analysis/oracle_5fold.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 5
    with pytest.raises(FileExistsError):
        analysis.aggregate(tmp_path, 4)


@pytest.mark.parametrize("kind", ["duplicate", "nan", "before", "identity"])
def test_invalid_score_rejected(tmp_path, kind):
    ids = synthetic_sweep(tmp_path)
    path = tmp_path / "scores/k0_frozen.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if kind == "duplicate":
        rows[1]["sample_id"] = "a"
    elif kind == "nan":
        rows[0]["score"] = float("nan")
    elif kind == "before":
        rows[0]["score_before"] = 9.
    else:
        rows[0]["score"] = 9.
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.complete_scores(tmp_path, ids, 4)


def test_no_overwrite_or_path_escape(tmp_path):
    path = tmp_path / "record.json"
    oracle.write_json(path, {"valid": True})
    with pytest.raises(FileExistsError):
        oracle.write_json(path, {})
    with pytest.raises(ValueError):
        oracle.contained_file(tmp_path / "child", "../record.json")


def test_cache_compatibility_checks_ordinary_fields():
    bundle = dict(baseline_id="trained-b", source_run_id="trained-run", checkpoint_ref="epoch_0007.pt",
                  preprocess={"sample_rate": 16000})
    legacy = dict(format="sharded_npy_v1", identity={"baseline_id": "trained-b", "preprocess_sha256": "ignored"})
    ordinary, status = oracle.cache_provenance(legacy, bundle)
    assert ordinary == {"baseline_id": "trained-b"}
    assert "legacy_readonly" in status
    legacy["identity"]["source_run_id"] = "wrong"
    with pytest.raises(ValueError):
        oracle.cache_provenance(legacy, bundle)
    current = dict(format="sharded_npy_v2", identity={k: bundle[k] for k in ("source_run_id", "checkpoint_ref", "preprocess")})
    current["identity"].update(dataset_id="in_the_wild", split_role="target_test")
    assert oracle.cache_provenance(current, bundle)[1] == "explicit_v2_provenance"
    current["identity"]["split_role"] = "select"
    with pytest.raises(ValueError):
        oracle.cache_provenance(current, bundle)


def test_production_guarded_rho_and_score_sink():
    import numpy as np
    import torch
    from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
    from eptta.baselines.dispatch import run_method
    path = DIRECTORY.parent / "target10_selection/scripts/_common.py"
    spec = importlib.util.spec_from_file_location("oracle_test_production_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # The classifier is orthogonal to U; balanced anchors stay feasible while
    # target view loss exercises the radius projection in the first two axes.
    resources = FrozenResources(torch.eye(3)[:, :2], torch.tensor([0., 0., 1.]), 0.,
                                torch.tensor([[0., 0., -1.], [0., 0., 1.]]),
                                torch.tensor([0, 1]), torch.ones(2), torch.tensor([-1., 1.]),
                                0., "synthetic")
    features = {"synthetic": np.array([[1., 0., 0.], [2., 1., 0.], [3., 0., 0.]], dtype=np.float32)}
    norms = []
    for rho in (.05, .4):
        candidate = dict(K=3, steps=3, lr=.3)
        captured = []
        row = helper.evaluate_candidate(candidate, ["synthetic"], features, resources, "synthetic",
                                        rho=rho, score_sink=captured.append)
        direct = run_method("ep_tta_guarded", TargetViews("synthetic", torch.from_numpy(features["synthetic"]), "synthetic"),
                            resources, EPConfig(steps=3, lr=.3, rho=rho, gamma=.1, lambda_keep=1.), {})
        assert captured == [{"sample_id": "synthetic", "score": float(direct["score"]),
                             "score_before": float(direct["score_before"])}]
        assert row["mean_R_norm"] == pytest.approx(float(torch.linalg.vector_norm(direct["R"])))
        norms.append(row["mean_R_norm"])
    assert norms[0] < norms[1]
