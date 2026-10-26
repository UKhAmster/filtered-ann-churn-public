"""Replays a runbook on an index and measures recall at checkpoints.

SEARCH operations are skipped: quality is measured only at checkpoints, on a fixed
query set, against exact search over the live collection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fanndyn.eval.metrics import filtered_ground_truth, recall_per_query
from fanndyn.runbook.schema import OpKind, Runbook


@dataclass(frozen=True)
class QueryFilter:
    name: str
    label: int | None  # None: unfiltered search


def evaluate(
    index,
    alive: np.ndarray,
    labels: np.ndarray,
    vectors: np.ndarray,
    queries: np.ndarray,
    flt: QueryFilter,
    k: int,
) -> dict:
    """Metrics of one filter on the current index state."""
    if flt.label is None:
        allowed_ids = alive
        mask = None
    else:
        allowed_ids = alive[labels[alive] == flt.label]
        mask = (labels == flt.label).tolist()
    truth = filtered_ground_truth(queries, vectors, allowed_ids, k)
    check = (lambda i: True) if mask is None else (lambda i: mask[i])
    found, lat = [], []
    for q in queries:
        t0 = time.perf_counter()
        found.append(index.search(q, k, check))
        lat.append((time.perf_counter() - t0) * 1e3)
    return {
        "filter": flt.name,
        **summarize(found, truth, lat, len(allowed_ids), len(alive)),
    }


def summarize(
    found: list[np.ndarray],
    truth: list[np.ndarray],
    latencies_ms: list[float] | np.ndarray,
    n_allowed: int,
    n_alive: int,
) -> dict:
    """Summary metrics of a checkpoint from search results and ground truth."""
    rec = recall_per_query(found, truth)
    valid = rec[~np.isnan(rec)]
    nan = float("nan")
    return {
        "n_allowed": int(n_allowed),
        "selectivity": n_allowed / max(1, n_alive),
        "n_queries": len(valid),
        "recall_mean": float(valid.mean()) if len(valid) else nan,
        "recall_p5": float(np.percentile(valid, 5)) if len(valid) else nan,
        "frac_below_0_9": float((valid < 0.9).mean()) if len(valid) else nan,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    }


def run_runbook(
    rb: Runbook,
    vectors: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    filters: list[QueryFilter],
    index_factory: Callable[[], object],
    k: int = 10,
    progress: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Replays the runbook and returns a row per (checkpoint, filter) pair.

    vectors and labels are indexed by object id; the input label array is not modified,
    attribute updates are applied to a working copy.
    """
    work_labels = labels.copy()
    alive = np.zeros(len(vectors), dtype=bool)
    index = index_factory()
    rows: list[dict] = []
    deleted_total = 0
    steps = 0
    t_start = time.perf_counter()
    for op_i, op in enumerate(rb.ops):
        if op.kind is OpKind.INSERT:
            ids = np.asarray(op.ids)
            index.insert(ids, vectors[ids])
            alive[ids] = True
        elif op.kind is OpKind.DELETE:
            ids = np.asarray(op.ids)
            index.delete(ids)
            alive[ids] = False
            deleted_total += len(ids)
        elif op.kind is OpKind.UPDATE_ATTR:
            for oid, ls in zip(op.ids, op.labels):
                work_labels[oid] = ls[0]
        elif op.kind is OpKind.SEARCH:
            steps += 1
        elif op.kind is OpKind.CHECKPOINT:
            alive_ids = np.flatnonzero(alive)
            for flt in filters:
                row = {
                    "runbook": rb.name,
                    "checkpoint": op.tag,
                    "op_index": op_i,
                    "step": steps,
                    "n_alive": len(alive_ids),
                    "n_deleted_total": deleted_total,
                    "wall_s": time.perf_counter() - t_start,
                    **evaluate(index, alive_ids, work_labels, vectors, queries, flt, k),
                }
                rows.append(row)
                if progress:
                    progress(row)
    return rows
