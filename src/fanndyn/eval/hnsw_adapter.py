"""hnswlib adapter with two deletion modes.

mark    - mark-delete (markDelete); new points take new slots;
          marked vertices stay in the graph and take part in traversal.
replace - new points take the slots of marked ones (replace_deleted); the graph
          is rebuilt locally in place of the old vertex.
"""

from __future__ import annotations

from collections.abc import Callable

import hnswlib
import numpy as np

MODES = ("mark", "replace")


class HnswAdapter:
    def __init__(
        self,
        dim: int,
        capacity: int,
        mode: str = "mark",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        seed: int = 0,
        build_threads: int = -1,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown deletion mode: {mode}")
        self.mode = mode
        self._build_threads = build_threads
        self._index = hnswlib.Index(space="l2", dim=dim)
        self._index.init_index(
            max_elements=capacity,
            M=m,
            ef_construction=ef_construction,
            random_seed=seed,
            allow_replace_deleted=(mode == "replace"),
        )
        self._index.set_ef(ef_search)
        self._marked = 0

    @property
    def capacity(self) -> int:
        return self._index.get_max_elements()

    def insert(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(ids)
        reusable = self._marked if self.mode == "replace" else 0
        free = self.capacity - self._index.get_current_count() + reusable
        if len(ids) > free:
            self._index.resize_index(max(2 * self.capacity, self.capacity + len(ids) - free))
        threads = self._build_threads if len(ids) > 1 else 1
        self._index.add_items(
            vectors, ids, num_threads=threads, replace_deleted=(self.mode == "replace")
        )
        if self.mode == "replace":
            self._marked = max(0, self._marked - len(ids))

    def delete(self, ids: np.ndarray) -> None:
        for oid in np.asarray(ids).tolist():
            self._index.mark_deleted(oid)
        self._marked += len(ids)

    def search(self, query: np.ndarray, k: int, allowed: Callable[[int], bool]) -> np.ndarray:
        """k nearest among allowed ids; fewer if the graph yields fewer than k candidates."""
        for kk in range(k, 0, -1):
            try:
                labels, _ = self._index.knn_query(query, k=kk, num_threads=1, filter=allowed)
                return labels[0].astype(np.int64)
            except RuntimeError:
                continue
        return np.array([], dtype=np.int64)
