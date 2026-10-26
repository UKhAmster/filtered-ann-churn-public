"""Derived metrics of the protocol.

They turn a set of plots into a protocol: each gives a single number that ranks
methods without arguing about the point at which to look.
The x axis is the number of executed operations, not seconds.
"""

from __future__ import annotations

import numpy as np


def loss_area(method: np.ndarray, oracle: np.ndarray, ops_axis: np.ndarray) -> float:
    """Integral of the method's shortfall from the oracle along the operation axis.

    Negative differences are dropped: a method cannot beat a full rebuild,
    and numerical noise must not count in its favor.
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
    """Total maintenance cost from the first drop below the target
    until quality reaches the target again. None if it never does."""
    below = np.flatnonzero(recalls < target)
    if len(below) == 0:
        return 0.0
    start = int(below[0])
    after = np.flatnonzero(recalls[start:] >= target)
    if len(after) == 0:
        return None
    end = start + int(after[0])
    return float(maintenance_seconds[start : end + 1].sum())
