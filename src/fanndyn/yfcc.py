"""Набор YFCC-10M (фильтрованный трек big-ann-benchmarks NeurIPS'23): настоящие теги.

Словарь тегов сокращается до запрашиваемых: каждый объект получает метки из числа
выбранных тегов, а объект без них — метку «прочее». На ответы для фильтров по выбранным
тегам это не влияет, а динамическому Filtered-DiskANN нужно не меньше стартовых точек, чем
меток, что при исходном словаре в 200 тыс. тегов невозможно. Сокращение одинаково для всех
методов.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fanndyn.data import read_spmat


def reduce_tags(ptr: np.ndarray, idx: np.ndarray, keep: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """CSR с метками 0..len(keep)-1 по порядку keep; объекты без них — метка len(keep).

    Метки объекта отсортированы: Filtered-DiskANN пересекает списки меток методом двух
    указателей и без сортировки не находит вершину по её не первой метке.
    """
    remap = {int(t): i for i, t in enumerate(keep)}
    other = len(keep)
    rows = []
    for r in range(len(ptr) - 1):
        kept = sorted(remap[int(t)] for t in idx[ptr[r] : ptr[r + 1]] if int(t) in remap)
        rows.append(kept or [other])
    new_ptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.uint64)
    new_idx = np.array([t for r in rows for t in r], dtype=np.uint32)
    return new_ptr, new_idx


def label_mask(ptr: np.ndarray, idx: np.ndarray, label: int) -> np.ndarray:
    """Булева маска объектов, у которых есть метка label."""
    n = len(ptr) - 1
    rows = np.repeat(np.arange(n), np.diff(ptr).astype(np.int64))
    mask = np.zeros(n, dtype=bool)
    mask[rows[idx == label]] = True
    return mask


def queries_with_tag(meta_path: Path, tag: int, limit: int) -> np.ndarray:
    """Номера первых limit запросов, у которых среди тегов есть tag."""
    ptr, idx = read_spmat(meta_path)
    return np.flatnonzero(label_mask(ptr, idx, tag))[:limit]
