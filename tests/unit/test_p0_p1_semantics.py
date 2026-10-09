"""Unit tests for the P1 scientific-flag composition semantics."""
import importlib.util
from pathlib import Path

import pytest

AGG_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "p0_p1_taskaware"
    / "p1"
    / "pilot_aggregate.py"
)
SPEC = importlib.util.spec_from_file_location("p1_pilot_aggregate_semantics", AGG_PATH)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)


def _flags(eer_less=False, auc_greater=False, objective_optimized=False,
           signed_delta=-0.01, frozen_accuracy=0.4):
    return AGG.compute_scientific_flags(
        eer_less_than_frozen=eer_less,
        auc_greater_than_frozen=auc_greater,
        objective_optimized=objective_optimized,
        mean_signed_task_delta=signed_delta,
        frozen_accuracy_at_tau0=frozen_accuracy,
    )


def test_no_task_gain_without_objective_optimization_is_not_combined():
    # task metric no gain, objective NOT optimized -> combined flag must be False.
    flags = _flags(eer_less=False, auc_greater=False, objective_optimized=False)
    assert flags["NO_TASK_GAIN"] is True
    assert flags["OBJECTIVE_OPTIMIZED"] is False
    assert flags["OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN"] is False


def test_objective_optimized_but_no_task_gain_requires_both():
    flags = _flags(eer_less=False, auc_greater=False, objective_optimized=True)
    assert flags["NO_TASK_GAIN"] is True
    assert flags["OBJECTIVE_OPTIMIZED"] is True
    assert flags["OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN"] is True


def test_task_metric_gain_suppresses_no_task_gain():
    # EER improves (eer_less=True): NO_TASK_GAIN must be False even if the
    # objective also improved.
    flags = _flags(eer_less=True, auc_greater=False, objective_optimized=True)
    assert flags["NO_TASK_GAIN"] is False
    assert flags["OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN"] is False
    # AUC improvement alone also suppresses NO_TASK_GAIN (strict definition).
    flags_auc = _flags(eer_less=False, auc_greater=True, objective_optimized=True)
    assert flags_auc["NO_TASK_GAIN"] is False
    assert flags_auc["OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN"] is False


def test_direction_and_teacher_flags():
    flags = _flags(signed_delta=-0.5, frozen_accuracy=0.4368)
    assert flags["UPDATE_DIRECTION_NOT_TASK_ALIGNED"] is True
    assert flags["FROZEN_TEACHER_UNRELIABLE"] is True
    flags_ok = _flags(signed_delta=+0.5, frozen_accuracy=0.6)
    assert flags_ok["UPDATE_DIRECTION_NOT_TASK_ALIGNED"] is False
    assert flags_ok["FROZEN_TEACHER_UNRELIABLE"] is False
