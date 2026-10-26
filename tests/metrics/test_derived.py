import numpy as np
import pytest

from fanndyn.metrics.derived import loss_area, recovery_cost, time_to_violation

OPS = np.array([0, 100, 200, 300], dtype=np.float64)


def test_loss_area_is_zero_when_method_matches_oracle():
    series = np.array([1.0, 0.9, 0.8, 0.7])
    assert loss_area(series, series, OPS) == pytest.approx(0.0)


def test_loss_area_integrates_gap_over_operations():
    oracle = np.array([1.0, 1.0, 1.0, 1.0])
    method = np.array([1.0, 0.9, 0.9, 1.0])
    assert loss_area(method, oracle, OPS) == pytest.approx(0.1 * 200)


def test_time_to_violation_returns_first_crossing():
    recalls = np.array([0.99, 0.95, 0.88, 0.80])
    assert time_to_violation(recalls, OPS, threshold=0.9) == 200


def test_time_to_violation_is_none_when_never_crossed():
    recalls = np.array([0.99, 0.98, 0.97, 0.96])
    assert time_to_violation(recalls, OPS, threshold=0.9) is None


def test_recovery_cost_sums_maintenance_until_target_regained():
    recalls = np.array([0.99, 0.80, 0.85, 0.96])
    maintenance = np.array([0.0, 1.0, 2.0, 3.0])
    assert recovery_cost(recalls, maintenance, OPS, target=0.95) == pytest.approx(6.0)


def test_recovery_cost_is_none_when_target_never_regained():
    recalls = np.array([0.99, 0.80, 0.81, 0.82])
    maintenance = np.array([0.0, 1.0, 2.0, 3.0])
    assert recovery_cost(recalls, maintenance, OPS, target=0.95) is None
