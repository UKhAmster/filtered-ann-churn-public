"""Эталон точного поиска с фильтром и полнота по запросам."""

from __future__ import annotations

import numpy as np


def filtered_ground_truth(
    queries: np.ndarray, base: np.ndarray, allowed: np.ndarray, k: int
) -> list[np.ndarray]:
    """Точные k ближайших (L2) среди идентификаторов allowed; base индексируется по id.

    Если допустимых объектов меньше k, эталон короче k.
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
    """Доля эталона, найденная методом; NaN для запросов с пустым эталоном."""
    out = np.full(len(truth), np.nan)
    for i, (f, t) in enumerate(zip(found, truth)):
        if len(t):
            out[i] = len(np.intersect1d(f, t)) / len(t)
    return out
