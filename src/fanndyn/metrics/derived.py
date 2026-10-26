"""Производные показатели из раздела 4.3 спецификации.

Именно они превращают набор графиков в протокол: дают одно число,
по которому методы ранжируются без спора о том, в какой точке смотреть.
Ось операций — число выполненных операций, а не секунды.
"""

from __future__ import annotations

import numpy as np


def loss_area(method: np.ndarray, oracle: np.ndarray, ops_axis: np.ndarray) -> float:
    """Интеграл отставания метода от оракула по оси операций.

    Отрицательные разности отбрасываются: превзойти полную перестройку
    метод не может, а численный шум не должен зачитываться в его пользу.
    """
    gap = np.clip(oracle - method, 0.0, None)
    return float(np.trapezoid(gap, ops_axis))


def time_to_violation(recalls: np.ndarray, ops_axis: np.ndarray, threshold: float) -> int | None:
    below = np.flatnonzero(recalls < threshold)
    if len(below) == 0:
        return None
    return int(ops_axis[below[0]])


def recovery_cost(
    recalls: np.ndarray,
    maintenance_seconds: np.ndarray,
    ops_axis: np.ndarray,
    target: float,
) -> float | None:
    """Суммарные затраты обслуживания от первого падения ниже цели
    до момента, когда качество снова достигло цели. None — если не достигло."""
    below = np.flatnonzero(recalls < target)
    if len(below) == 0:
        return 0.0
    start = int(below[0])
    after = np.flatnonzero(recalls[start:] >= target)
    if len(after) == 0:
        return None
    end = start + int(after[0])
    return float(maintenance_seconds[start : end + 1].sum())
