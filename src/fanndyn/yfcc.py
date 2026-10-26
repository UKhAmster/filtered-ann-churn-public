"""YFCC-10M (filtered track of big-ann-benchmarks NeurIPS'23): real tags.

The tag vocabulary is reduced to the queried tags: each object gets labels among the
selected tags, and an object with none of them gets the label "other". This does not
change answers for filters on the selected tags, while dynamic Filtered-DiskANN needs at
least as many start points as labels, which is impossible with the original vocabulary of
200k tags. The reduction is the same for all methods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fanndyn.data import read_spmat


def reduce_tags(ptr: np.ndarray, idx: np.ndarray, keep: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """CSR with labels 0..len(keep)-1 in the order of keep; objects without them get len(keep).

    Labels of an object are sorted: Filtered-DiskANN intersects label lists with two
    pointers and, without sorting, fails to find a vertex by any label but its first.
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
    """Boolean mask of objects that carry label."""
    n = len(ptr) - 1
    rows = np.repeat(np.arange(n), np.diff(ptr).astype(np.int64))
    mask = np.zeros(n, dtype=bool)
    mask[rows[idx == label]] = True
    return mask


def queries_with_tag(meta_path: Path, tag: int, limit: int) -> np.ndarray:
    """Indices of the first limit queries whose tags include tag."""
    ptr, idx = read_spmat(meta_path)
    return np.flatnonzero(label_mask(ptr, idx, tag))[:limit]
