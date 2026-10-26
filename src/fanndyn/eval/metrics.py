"""Exact filtered search ground truth and per-query recall."""

from __future__ import annotations

import numpy as np


def filtered_ground_truth(
    queries: np.ndarray, base: np.ndarray, allowed: np.ndarray, k: int
) -> list[np.ndarray]:
    """Exact k nearest (L2) among ids in allowed; base is indexed by id.

    If fewer than k objects are allowed, the ground truth is shorter than k.
    """
    if len(allowed) == 0:
        return [np.array([], dtype=np.int64) for _ in queries]
    cand = base[allowed]
    d = (
        (queries**2).sum(axis=1, keepdims=True)
        - 2.0 * queries @ cand.T
        + (cand**2).sum(axis=1)[None, :]
    )
    kk = min(k, len(allowed))
    part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    order = np.take_along_axis(d, part, axis=1).argsort(axis=1)
    return [allowed[row] for row in np.take_along_axis(part, order, axis=1)]


def recall_per_query(found: list[np.ndarray], truth: list[np.ndarray]) -> np.ndarray:
    """Share of the ground truth found by the method; NaN for queries with empty ground truth."""
    out = np.full(len(truth), np.nan)
    for i, (f, t) in enumerate(zip(found, truth)):
        if len(t):
            out[i] = len(np.intersect1d(f, t)) / len(t)
    return out
