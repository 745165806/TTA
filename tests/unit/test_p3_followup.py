"""P3.1 statistical and P3.2 class-asymmetric contracts."""
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

from eptta.adaptation.types import FrozenResources, TargetViews

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/p3_calibrated_teacher"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "scripts"))

from followup_stats import task_gain_supported
from workers.asymmetry_worker import class_is_held, held_record


def fixture():
    dtype = torch.float64
    U = torch.tensor([[1.0], [0.0]], dtype=dtype)
    w = torch.tensor([1.0, 0.0], dtype=dtype)
    anchors = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    scores = anchors @ w
    margins = (2 * labels.to(dtype) - 1) * scores
    resources = FrozenResources(U, w, 0.0, anchors, labels, margins, scores, 0.0, "followup")
    target = TargetViews("sample", torch.tensor(
        [[2.0, 0.0], [2.1, 0.0], [1.9, 0.0]], dtype=dtype), "features")
    params = {
        "mu_bona": 0.0, "mu_spoof": 2.0, "var_bona": 0.04, "var_spoof": 0.04,
        "pi_bona": 0.5, "pi_spoof": 0.5, "tau_hat": 1.0,
        "confidence_threshold": 0.9, "min_agreement": 1.0,
    }
    return resources, target, params


def test_task_gain_requires_paired_bootstrap_significance():
    favorable_point_only = {
        "delta_EER": 0.0, "delta_EER_ci95": [-0.001, 0.001],
        "delta_AUC": 0.0002, "delta_AUC_ci95": [-0.0001, 0.0005],
    }
    assert task_gain_supported(favorable_point_only) is False
    significant_auc = dict(favorable_point_only, delta_AUC_ci95=[0.00001, 0.0005])
    assert task_gain_supported(significant_auc) is True


def test_sensitivity_salts_are_fixed():
    config = json.loads((EXP / "configs/followup.json").read_text(encoding="utf-8"))
    assert config["sensitivity_salts"] == ["p3-sensitivity-2027", "p3-sensitivity-2028"]


def test_bona_only_never_adapts_spoof_teacher():
    assert class_is_held(1, "bona_only") is True
    assert class_is_held(0, "bona_only") is False


def test_spoof_only_never_adapts_bona_teacher():
    assert class_is_held(0, "spoof_only") is True
    assert class_is_held(1, "spoof_only") is False


def test_class_asymmetric_abstain_is_exact_frozen():
    resources, target, params = fixture()
    record = held_record("Calibrated-BonaOnly", target, resources, params, teacher=1)
    assert record["score_after"] == record["score_before"]
    assert record["delta_score"] == 0.0
    assert record["adaptation_applied"] is False
    assert record["abstain_reason"] == "class_asymmetric_spoof_hold"
    assert record["source_anchor_flip_count"] == 0


def test_oracle_remains_post_hoc_only():
    resources, target, params = fixture()
    record = held_record("Oracle-BonaOnly", target, resources, params, teacher=1, oracle=True)
    assert record["POST_HOC_ORACLE_ONLY"] is True
    assert record["NOT_DEPLOYABLE"] is True
    assert record["method_id"] == "POST_HOC_ORACLE_ONLY"


def test_source_safety_unchanged():
    resources, target, params = fixture()
    record = held_record("Calibrated-BonaOnly", target, resources, params, teacher=1)
    assert record["final_source_anchor_flip_count"] == 0
    launcher = (EXP / "workers/asymmetry_worker.py").read_text(encoding="utf-8")
    assert 'run_method("ep_tta_calibrated_selective_v1"' in launcher


def test_no_gpu_required():
    launcher = (EXP / "run_p3_followup.sh").read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES=""' in launcher
    assert "export OMP_NUM_THREADS=1" in launcher
    assert "export MKL_NUM_THREADS=1" in launcher
    assert "export OPENBLAS_NUM_THREADS=1" in launcher
    config = json.loads((EXP / "configs/followup.json").read_text(encoding="utf-8"))
    assert config["max_parallel_cpu_workers"] == 4
    assert config["cuda_visible_devices"] == ""
