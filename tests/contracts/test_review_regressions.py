"""Regression evidence for reviewed dispatch, sealing, sampling and generic R4 contracts."""
import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.adapter import adaptation_diagnostics, run_cache_method
from eptta.adaptation.types import EPConfig, FrozenResources, TargetViews
from eptta.data.io import sha256_file
from eptta.data.preprocess import derive_16k_wav
from eptta.config.validate import content_hash
from eptta.evaluation.seal import seal_scores
from eptta.errors import ContractError, DataError
from eptta.execution.r6 import run_r6
from eptta.execution.suite import freeze_selection, select_methods
from eptta.evaluation.seal import build_report
from eptta.models.contracts import FrozenModelBundle
from eptta.training.selection import select_source_checkpoint
from workers.compat.author_training import EpochShardSampler, ManifestDataset
from scripts.make_plans import lock as lock_review


def _episode(dtype=torch.float64):
    generator = torch.Generator().manual_seed(31)
    d, rank = 6, 2
    U = torch.linalg.qr(torch.randn(d, rank, generator=generator, dtype=dtype))[0]
    w = torch.randn(d, generator=generator, dtype=dtype)
    labels = torch.arange(6) % 2
    anchors = torch.randn(6, d, generator=generator, dtype=dtype)
    signs = 2 * labels.to(dtype) - 1
    anchors += ((signs - anchors @ w) / w.square().sum())[:, None] * w
    scores = anchors @ w
    resources = FrozenResources(U, w, 0.0, anchors, labels, signs * scores, scores, 0.0, "bundle")
    target = TargetViews("sample", torch.randn(3, d, generator=generator, dtype=dtype), "cache")
    return target, resources


def test_static_requires_amount_and_matched_radius_and_zero_identity():
    target, resources = _episode()
    cfg = EPConfig(rho=0.2)
    with pytest.raises(ValueError, match="explicit params.amount"):
        run_cache_method("static_subspace", target, resources, cfg)
    zero = run_cache_method("static_subspace", target, resources, cfg, {"amount": 0.0})
    assert zero["score"] == zero["score_before"]
    amount = cfg.rho / resources.U.shape[1] ** 0.5
    full = run_cache_method("static_subspace", target, resources, cfg, {"amount": amount})
    assert float(torch.linalg.vector_norm(full["R"])) == pytest.approx(cfg.rho)


def test_final_diagnostic_uses_final_margin_and_projection_history():
    target, resources = _episode()
    result = run_cache_method("ep_tta", target, resources, EPConfig(steps=1, lr=0.5, rho=0.05))
    diag = adaptation_diagnostics(result, target, resources, EPConfig(steps=1, lr=0.5, rho=0.05))
    assert diag["final_margin_loss"] >= 0
    assert diag["projection_count"] == sum(row["projection_applied"] for row in result["trace"])
    assert diag["regularizer_active_steps"] == 0  # margin term has zero initial gradient at K=1


def test_numeric_failure_is_typed_and_configuration_error_is_not_fallback(monkeypatch):
    target, resources = _episode()
    monkeypatch.setattr("eptta.adaptation.adapter.target_objective",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(FloatingPointError("synthetic")))
    result = run_cache_method("ep_tta", target, resources)
    assert result["status"] == "fallback_numeric"
    assert result["error_type"] == "FloatingPointError" and result["score"] == result["score_before"]
    with pytest.raises(ValueError):
        run_cache_method("unknown", target, resources)


@pytest.mark.parametrize("size,world,batch,policy,expected", [
    (1, 1, 8, "shuffle_repeat_to_even", 8),
    (5, 2, 8, "shuffle_repeat_to_even", 16),
    (16, 2, 8, "shuffle_repeat_to_even", 16),
    (17, 1, 8, "shuffle_drop_global_tail", 16),
])
def test_small_training_sampler_boundaries(size, world, batch, policy, expected):
    shards = [EpochShardSampler(size, 3, rank, world, batch, policy).indices() for rank in range(world)]
    assert sum(map(len, shards)) == expected
    if policy == "shuffle_repeat_to_even":
        assert set().union(*(set(row) for row in shards)) == set(range(size))


def test_sampler_empty_and_drop_no_batch_fail():
    with pytest.raises(ValueError, match="empty"):
        EpochShardSampler(0, 3, 0, 1, 8, "shuffle_keep_tail").indices()
    with pytest.raises(ValueError, match="no complete"):
        EpochShardSampler(5, 3, 0, 2, 8, "shuffle_drop_global_tail").indices()


def test_checkpoint_selection_rejects_missing_key_with_extras_and_bad_eer():
    with pytest.raises(ContractError, match="missing"):
        select_source_checkpoint([{"epoch": 1, "source_val_eer": 0.2,
                                   "checkpoint_ref": "x", "extra": True}])
    with pytest.raises(ContractError, match=r"\[0,1\]"):
        select_source_checkpoint([{"epoch": 1, "source_val_eer": 1.1,
                                   "checkpoint_ref": "x", "checkpoint_sha256": "a" * 64}])


def test_score_seal_rejects_equal_count_wrong_id_set(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    expected = run / "expected.json"
    expected.write_text(json.dumps(["a", "b"]))
    scores = run / "scores.jsonl"
    scores.write_text("".join(json.dumps({"sample_id": value, "score": 0.0, "status": "ok"}) + "\n"
                              for value in ("a", "c")))
    (run / "run.json").write_text(json.dumps({"expected_ids_ref": "expected.json",
        "expected_ids_sha256": sha256_file(expected), "expected_sample_count": 2,
        "scores_sha256": sha256_file(scores), "feature_cache_key": "cache",
        "artifact_bundle_id": "bundle", "fallback_rate_max": 0.0}))
    with pytest.raises(DataError, match="ID coverage"):
        seal_scores(run)


def test_run_r6_rejects_proposed_before_resources(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"schema_version": "0.2.0", "status": "PROPOSED"}))
    with pytest.raises(ContractError, match="LOCKED"):
        run_r6(proposal, tmp_path / "out")


def test_review_lock_rejects_one_changed_byte_and_locks_unchanged(tmp_path):
    changed = tmp_path / "changed"
    changed.mkdir()
    payload = changed / "plan.txt"
    payload.write_bytes(b"one")
    review = changed / "review.json"
    review.write_text(json.dumps({"status": "PROPOSED", "stage": "select",
                                  "files": {"plan.txt": sha256_file(payload)}}))
    digest = sha256_file(review)
    payload.write_bytes(b"two")
    with pytest.raises(DataError, match="changed"):
        lock_review(review, digest, "reviewer")
    clean = tmp_path / "clean"
    clean.mkdir()
    payload = clean / "plan.txt"
    payload.write_bytes(b"one")
    review = clean / "review.json"
    review.write_text(json.dumps({"status": "PROPOSED", "stage": "select",
                                  "files": {"plan.txt": sha256_file(payload)}}))
    result = lock_review(review, sha256_file(review), "reviewer")
    assert result["status"] == "LOCKED" and (clean / "locked" / "lock.json").is_file()


def test_generic_r4_accepts_new_epoch_counts_and_attack_policy():
    provenance = {"training_endpoint": {"last_epoch": 3, "completed_epoch_count": 4,
                                         "scheduler_horizon_epochs": 4},
                  "completion_evidence": "locked_recipe_endpoint_or_approved_stop"}
    validation = {"status": "PASS", "fit_count": 8, "source_val_count": 5,
                  "parity_policy": {"per_class_budget": 4,
                                    "expected_attack_ids": ["codec-a", "codec-b"]}}
    bundle = FrozenModelBundle("0.1.0", "aasist_source", "baseline", "a" * 64, "run",
        "b" * 64, "c" * 64, "d" * 64, {}, provenance, "e" * 64,
        {"bonafide": 1, "spoof": 0}, "head.pt", 12, "FINALIZED", "full",
        "trained_in_project", "selection.json", "parity.json",
        {"embedding_point": "native_out_layer_input", "freq_aug": False},
        {"formula": "native_logits[spoof]-native_logits[bonafide]", "direction": "larger_is_spoof",
         "output_type": "logit_difference", "unit": "dimensionless"},
        {"atol": 1e-6, "rtol": 1e-5}, validation, "f" * 64)
    assert bundle.r4_validation["source_val_count"] == 5


def test_manifest_uses_reviewed_multi_suffix_paths_and_blocks_escape(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    rows = [{"schema_version": "0.1.0", "sample_id": "a", "root_key": "dataset",
             "audio_relpath": "nested/a.wav", "input_sha256": None, "split_role": "fit",
             "canonical_label": 0},
            {"schema_version": "0.1.0", "sample_id": "b", "root_key": "dataset",
             "audio_relpath": "b.flac", "input_sha256": None, "split_role": "fit",
             "canonical_label": 1}]
    manifest = tmp_path / "fit.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = ManifestDataset(manifest, sha256_file(manifest), {"dataset": str(root)}, "fit")
    assert dataset.rows[0]["audio_path"].endswith("nested/a.wav")
    rows[0]["audio_relpath"] = "../escape.wav"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="escapes"):
        ManifestDataset(manifest, sha256_file(manifest), {"dataset": str(root)}, "fit")


def test_resampling_writes_new_lineage_and_true_16k_audio(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    source = tmp_path / "source.wav"
    soundfile.write(source, np.linspace(-0.2, 0.2, 800, dtype=np.float32), 8000)
    output, log = tmp_path / "derived.wav", tmp_path / "derived.json"
    lineage = derive_16k_wav(source, output, "parent-id", log)
    audio, rate = soundfile.read(output)
    assert rate == 16000 and len(audio) > 800
    assert lineage["source_sha256"] == sha256_file(source)
    assert lineage["derived_sha256"] == sha256_file(output)
    assert lineage["parent_sample_id"] == "parent-id"


def _selection_candidate(tmp_path, candidate_id, method_id, role="select", eer=0.2):
    suite = tmp_path / (candidate_id + ".suite.json")
    method = {"method_id": method_id, "config": {"steps": 0 if method_id == "frozen" else 1,
              "lr": 0.01, "rho": 0.2, "gamma": 0.1, "lambda_keep": 1.0}, "params": {}}
    suite.write_text(json.dumps({"suite_id": candidate_id, "input_role": role, "methods": [method]}))
    run_dir = tmp_path / "select_runs" / candidate_id / method_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"phase": "select", "suite_id": candidate_id,
        "baseline_id": "baseline", "artifact_bundle_id": "resources", "coverage_fraction": 1.0,
        "fallback_rate": 0.0}))
    (run_dir / "score_seal.json").write_text(json.dumps({"status": "SEALED", "score_sha256": "a" * 64}))
    (run_dir / "evaluation.json").write_text(json.dumps({"score_sha256": "a" * 64,
                                                          "metrics": {"eer": eer}}))
    return method, {"candidate_id": candidate_id, "family": method_id, "method_id": method_id,
        "suite_id": candidate_id, "suite_plan_ref": str(suite), "suite_plan_sha256": sha256_file(suite),
        "candidate_config_sha256": content_hash(method),
        "evaluation_ref": str((run_dir / "evaluation.json").relative_to(tmp_path)),
        "coverage_min": 1.0, "fallback_rate_max": 0.01, "rank_order": [method["config"]["steps"],
        method["config"]["lr"], candidate_id]}


def test_selection_reads_only_select_and_freeze_copies_source_choice(tmp_path):
    _frozen_method, frozen = _selection_candidate(tmp_path, "frozen", "frozen", eer=0.3)
    ep_method, ep = _selection_candidate(tmp_path, "ep", "ep_tta", eer=0.2)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"status": "LOCKED", "baseline_id": "baseline",
        "artifact_bundle_id": "resources", "required_ep_families": ["ep_tta"],
        "candidates": [frozen, ep]}))
    selection_ref = tmp_path / "selection.json"
    result = select_methods(tmp_path, policy, selection_ref)
    assert result["selected"]["ep_tta"]["method"] == ep_method
    assert result["selected"]["ep_tta"]["improves_frozen"] is True
    freeze_plan = tmp_path / "freeze-plan.json"
    freeze_plan.write_text(json.dumps({"status": "LOCKED", "selection_policy_sha256": sha256_file(policy),
        "resources_ref": "/resources", "frozen_bundle_ref": "/bundle", "targets": [{
            "target_id": "external", "role": "target_test", "scope_id": "scope",
            "labels_ref": "/labels-only-for-evaluator", "extraction_plan_ref": "/extract.json",
            "feature_cache_ref": "/cache", "input_manifest_ref": "/manifest.jsonl",
            "input_manifest_sha256": "a" * 64}]}))
    frozen_output = tmp_path / "frozen"
    freeze_selection(freeze_plan, selection_ref, frozen_output)
    target_suite = json.loads((frozen_output / "external" / "suite.json").read_text())
    assert target_suite["methods"] == [ep_method, result["selected"]["frozen"]["method"]]
    assert "labels" not in json.dumps(target_suite)


def test_selection_rejects_target_role(tmp_path):
    _method, candidate = _selection_candidate(tmp_path, "ep", "ep_tta", role="target_test")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"status": "LOCKED", "baseline_id": "baseline",
        "artifact_bundle_id": "resources", "candidates": [candidate]}))
    with pytest.raises(ContractError, match="source-select"):
        select_methods(tmp_path, policy, tmp_path / "selection.json")


def test_report_uses_consistent_units_and_preserves_not_run(tmp_path):
    evaluated = tmp_path / "eval"
    evaluated.mkdir()
    (evaluated / "evaluation.json").write_text(json.dumps({"status": "EVALUATED", "metrics": {
        "eer": 0.125, "auroc": 0.8, "fpr": 0.1, "fnr": 0.2,
        "helpful_flips_by_class": {"0": 1, "1": 2}, "harmful_flips_by_class": {"0": 0, "1": 1}}}))
    (evaluated / "run.json").write_text(json.dumps({"sample_count": 12, "fallback_count": 1,
        "cost": {"elapsed_seconds": 2.5}}))
    common = {"source": "src", "model": "aasist", "seed": 1, "target": "data",
              "release": "r", "subset": "eval", "config": "c"}
    (tmp_path / "report-plan.json").write_text(json.dumps({"status": "LOCKED", "rows": [
        {**common, "method": "ep", "evaluation_ref": "eval/evaluation.json", "run_ref": "eval/run.json"},
        {**common, "method": "port", "status": "NOT_RUN", "reason": "BLOCKED_AUDIT"}]}))
    build_report(tmp_path, tmp_path / "report")
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())["rows"]
    assert summary[0]["EER_pct"] == 12.5 and summary[0]["AUROC"] == 0.8
    assert summary[1]["status"] == "NOT_RUN" and summary[1]["reason"] == "BLOCKED_AUDIT"
