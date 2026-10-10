"""CPU-only checks for the fixed source diagnostic analysis contract."""
import math
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/meta_audio_tta"))
from experiments.meta_audio_tta.source_gpu_diagnostics import (
    fixed_indices, summarize_acoustic_pairs, summarize_scores, validate_resource_snapshot,
)


def test_fixed_class_offsets_are_deterministic_and_disjoint():
    rows = [{"sample_id": sample_id, "canonical_label": label}
            for label, prefix in ((0, "real"), (1, "fake"))
            for sample_id in (f"{prefix}-{index:03d}" for index in range(5))]
    assert fixed_indices(rows, 2, 2) == [2, 3, 7, 8]
    with pytest.raises(ValueError, match="incomplete"):
        fixed_indices(rows, 4, 2)


def test_k0_k1_metrics_separate_pair_flips_from_decision_flips():
    rows = [
        {"sample_id": "r1", "canonical_label": 0, "k0": -0.2, "k1": -0.2,
         "backbone_gradient_l1": 1.0, "aux_gradient_l1": 1.0,
         "backbone_update_l1": 0.1, "aux_update_l1": 0.1},
        {"sample_id": "r2", "canonical_label": 0, "k0": -0.1, "k1": 0.05,
         "backbone_gradient_l1": 1.0, "aux_gradient_l1": 1.0,
         "backbone_update_l1": 0.1, "aux_update_l1": 0.1},
        {"sample_id": "f1", "canonical_label": 1, "k0": 0.1, "k1": 0.02,
         "backbone_gradient_l1": 1.0, "aux_gradient_l1": 1.0,
         "backbone_update_l1": 0.1, "aux_update_l1": 0.1},
        {"sample_id": "f2", "canonical_label": 1, "k0": 0.2, "k1": 0.2,
         "backbone_gradient_l1": 1.0, "aux_gradient_l1": 1.0,
         "backbone_update_l1": 0.1, "aux_update_l1": 0.1},
    ]
    summary = summarize_scores(rows)
    assert summary["k0"]["eer"] == 0.0
    assert summary["k0"]["auroc"] == 1.0
    assert summary["k1"]["harmful_flips"] == 1
    assert summary["pair_order_flips"]["correct_to_incorrect"] == 1
    assert summary["score_change"]["nonzero_count"] == 2
    assert math.isfinite(summary["score_change"]["affine_residual_rms"])
    rows[0]["k1"] = None
    with pytest.raises(ValueError, match="partial"):
        summarize_scores(rows)


def test_ce_frozen_never_fabricates_k1():
    rows = [
        {"sample_id": "r", "canonical_label": 0, "k0": -1.0, "k1": None},
        {"sample_id": "f", "canonical_label": 1, "k0": 1.0, "k1": None},
    ]
    summary = summarize_scores(rows)
    assert summary["k0"]["auroc"] == 1.0
    assert summary["k1"].startswith("NOT_RUN")
    assert "pair_order_flips" not in summary


def test_resource_gate_fails_closed():
    limits = {"min_free_gpu_gib": 20, "max_gpu_utilization_fraction": 0.2,
              "min_available_ram_gib": 16,
              "min_disk_free_gib": 100, "max_io_some_avg10_fraction": 0.05,
              "max_cpu_load_fraction": 0.75}
    snapshot = {"free_gpu_gib": 30, "gpu_utilization_fraction": 0,
                "available_ram_gib": 70,
                "disk_free_gib": 600, "io_some_avg10_fraction": 0.0,
                "cpu_load_fraction": 0.1}
    validate_resource_snapshot(snapshot, limits)
    with pytest.raises(RuntimeError, match="GPU free memory"):
        validate_resource_snapshot({**snapshot, "free_gpu_gib": 19}, limits)
    with pytest.raises(RuntimeError, match="already active"):
        validate_resource_snapshot({**snapshot, "gpu_utilization_fraction": 0.21}, limits)
    with pytest.raises(RuntimeError, match="disk I/O pressure"):
        validate_resource_snapshot({**snapshot, "io_some_avg10_fraction": 0.06}, limits)


def test_acoustic_pair_summary_is_json_serializable():
    original = {"a": {"k0": 0.1, "k1": 0.2}}
    perturbed = {"a": {"k0": -0.1, "k1": -0.2}}
    summary = summarize_acoustic_pairs({"original": original,
                                        "deterministic_fir": perturbed})
    assert summary["adaptation_delta_sign_changed"] == 1
    assert json.loads(json.dumps(summary)) == summary
