"""Summary of repeats over seeds: range of key values for each (method, mode, filter).

For each run it takes recall and the share of queries below 0.9 at the start and end of the
trajectory, the ratio of work at the end to the start, and time-averaged recall; the minimum
and maximum over seeds are printed.

Example:
    .venv/bin/python scripts/aggregate_seeds.py results/yfcc/churn_3x results/yfcc/churn_3x_seed8 \
        results/yfcc/churn_3x_seed9
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def run_summary(path: Path, seed_dir: str) -> list[dict]:
    d = pd.read_csv(path)
    d = d[d.checkpoint != "rebuild"]
    rows = []
    for flt, sub in d.groupby("filter", sort=False):
        sub = sub.sort_values("step")
        start, end = sub.iloc[0], sub.iloc[-1]
        traj = sub[sub.step > 0]
        rows.append({
            "run": seed_dir,
            "method": path.parent.name,
            "mode": path.stem.split("-", 1)[1],
            "filter": flt,
            "recall_start": start.recall_mean,
            "recall_end": end.recall_mean,
            "recall_mean": traj.recall_mean.mean(),
            "recall_min": traj.recall_mean.min(),
            "below09_start": start.frac_below_0_9,
            "below09_end": end.frac_below_0_9,
            "work_ratio_end": end.cmps_mean / start.cmps_mean,
        })  # fmt: skip
    return rows


def main() -> None:
    rows = []
    for root in (Path(x) for x in sys.argv[1:]):
        for path in sorted(root.glob("*/*.csv")):
            if path.name == "derived.csv":
                continue
            rows += run_summary(path, root.name)
    t = pd.DataFrame(rows)
    keys = ["method", "mode", "filter"]
    agg = t.groupby(keys, sort=False).agg(
        n=("run", "count"),
        **{
            f"{c}_{fn}": (c, fn)
            for c in t.columns
            if c not in keys + ["run"]
            for fn in ("min", "max")
        },
    )
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
