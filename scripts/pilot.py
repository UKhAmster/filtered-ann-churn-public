"""Pilot measurement: recall trajectory of hnswlib under churn on a SIFT sample.

Example:
    .venv/bin/python scripts/pilot.py --scenario steady_churn --mode mark --out results/pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from fanndyn.data import LABEL_KINDS, make_labels, read_fvecs
from fanndyn.eval.hnsw_adapter import HnswAdapter
from fanndyn.eval.runner import QueryFilter, evaluate, run_runbook
from fanndyn.runbook.export import with_base_checkpoint
from fanndyn.runbook.scenarios import build_scenario
from fanndyn.runbook.schema import OpKind

N_LABELS = 20
WIDE_LABEL = 0  # about 28 % of the collection
NARROW_LABEL = N_LABELS - 1  # about 1.4 %
PURGE_LABEL = 1  # about 14 %
LABEL_SEED = 11


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--labels", choices=LABEL_KINDS, default="zipf")
    p.add_argument("--mode", choices=["mark", "replace"], required=True)
    p.add_argument("--data", type=Path, default=Path.home() / "data/sift")
    p.add_argument("--n-base", type=int, default=100_000)
    p.add_argument("--n-ops", type=int, default=100_000)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--n-queries", type=int, default=300)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("results/pilot"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    n_total = a.n_base + a.n_ops
    vectors = read_fvecs(a.data / "sift_base.fvecs", count=n_total)
    queries = read_fvecs(a.data / "sift_query.fvecs", count=a.n_queries)
    labels = make_labels(a.labels, vectors, N_LABELS, seed=LABEL_SEED)
    rb = with_base_checkpoint(
        build_scenario(
            a.scenario,
            n_base=a.n_base,
            n_ops=a.n_ops,
            label_sets=[(int(x),) for x in labels],
            seed=a.seed,
            checkpoint_every=a.checkpoint_every,
            purge_label=PURGE_LABEL,
        )
    )
    filters = [
        QueryFilter("none", None),
        QueryFilter("wide", WIDE_LABEL),
        QueryFilter("narrow", NARROW_LABEL),
    ]

    def factory() -> HnswAdapter:
        return HnswAdapter(
            dim=vectors.shape[1],
            capacity=a.n_base,
            mode=a.mode,
            m=a.m,
            ef_construction=a.ef_construction,
            ef_search=a.ef_search,
            seed=a.seed,
        )

    def progress(row: dict) -> None:
        print(
            f"{row['checkpoint']:>7} {row['filter']:>6} step={row['step']:>6} "
            f"recall={row['recall_mean']:.4f} p5={row['recall_p5']:.3f} "
            f"lat={row['latency_ms_mean']:.2f}ms wall={row['wall_s']:.0f}s",
            flush=True,
        )

    rows = run_runbook(rb, vectors, labels, queries, filters, factory, k=a.k, progress=progress)

    # Reference point: an index built from scratch on the final live collection.
    alive = np.ones(n_total, dtype=bool)
    for op in rb.ops:
        if op.kind is OpKind.DELETE:
            alive[list(op.ids)] = False
    last_step = rows[-1]["step"]
    inserted = {i for op in rb.ops if op.kind is OpKind.INSERT for i in op.ids}
    alive_ids = np.array(sorted(i for i in inserted if alive[i]))
    t0 = time.perf_counter()
    fresh = HnswAdapter(
        dim=vectors.shape[1],
        capacity=len(alive_ids),
        mode="mark",
        m=a.m,
        ef_construction=a.ef_construction,
        ef_search=a.ef_search,
        seed=a.seed,
    )
    fresh.insert(alive_ids, vectors[alive_ids])
    rebuild_s = time.perf_counter() - t0
    work_labels = labels.copy()
    for op in rb.ops:
        if op.kind is OpKind.UPDATE_ATTR:
            for oid, ls in zip(op.ids, op.labels):
                work_labels[oid] = ls[0]
    for flt in filters:
        row = {
            "runbook": rb.name,
            "checkpoint": "rebuild",
            "op_index": -1,
            "step": last_step,
            "n_alive": len(alive_ids),
            "n_deleted_total": rows[-1]["n_deleted_total"],
            "wall_s": rebuild_s,
            **evaluate(fresh, alive_ids, work_labels, vectors, queries, flt, a.k),
        }
        rows.append(row)
        progress(row)

    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.scenario}-{a.mode}"
    with (a.out / f"{stem}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {
        **{k: str(v) for k, v in vars(a).items()},
        "generator_version": rb.generator_version,
        "n_labels": N_LABELS,
        "wide_label": WIDE_LABEL,
        "narrow_label": NARROW_LABEL,
        "purge_label": PURGE_LABEL,
        "label_seed": LABEL_SEED,
        "rebuild_s": rebuild_s,
    }
    (a.out / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
