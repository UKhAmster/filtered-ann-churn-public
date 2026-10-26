"""Derived metrics of the protocol computed from run results.

For each (run, filter) pair:
  * oracle: recall of a fresh index, the mean of the base checkpoint and of the index rebuilt
    on the final collection (under steady churn the collection is statistically unchanged);
  * mean recall deficit: the loss area divided by the run length in operations;
  * time to violation: operations until the first checkpoint with recall below
    "recall at the start of the run - threshold" (resolution: the checkpoint step); counted
    from the run's own start, because for narrow filters base and the rebuilt index differ
    by more than the threshold due to the noise of small query sets;
  * mean excess work: the same for distance computations relative to base;
  * maintenance: total time of consolidations or rebuilds per run and the time of a full
    rebuild of the reference index.

Example:
    .venv/bin/python scripts/derived_table.py results/yfcc/churn_3x --drop 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fanndyn.metrics.derived import loss_area, time_to_violation


def summarize_run(path: Path, drop: float) -> list[dict]:
    d = pd.read_csv(path)
    method = path.parent.name
    mode = path.stem.split("-", 1)[1]
    rows = []
    for flt, sub in d.groupby("filter", sort=False):
        traj = sub[sub.checkpoint != "rebuild"].sort_values("step")
        reb = sub[sub.checkpoint == "rebuild"]
        base = traj.iloc[0]
        oracle = float(np.mean([base.recall_mean, reb.recall_mean.iloc[0]]))
        ops = traj.step.to_numpy(dtype=float)
        span = ops[-1] - ops[0]
        recall = traj.recall_mean.to_numpy()
        work = traj.cmps_mean.to_numpy() / base.cmps_mean
        ttv = time_to_violation(recall[1:], ops[1:], recall[0] - drop)
        rows.append({
            "метод": method,
            "режим": mode,
            "фильтр": flt,
            "доля": base.selectivity,
            "оракул": oracle,
            "полнота в конце": recall[-1],
            "недодан, п.п.": 100 * loss_area(recall, np.full_like(recall, oracle), ops) / span,
            "наработка до нарушения, тыс.": None if ttv is None else ttv / 1000,
            "избыток работы, %": 100 * loss_area(np.ones_like(work), work, ops) / span,
            "доля < 0,9 в среднем": traj.frac_below_0_9.mean(),
            "обслуживание за прогон, с": traj.consolidate_s_total.max(),
            "полная перестройка, с": reb.driver_wall_s.iloc[0],
        })  # fmt: skip
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--drop", type=float, default=0.02, help="allowed recall loss from the oracle")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    rows = [r for path in sorted(a.root.glob("*/*.csv")) for r in summarize_run(path, a.drop)]
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    out = a.out or a.root / "derived.csv"
    table.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
