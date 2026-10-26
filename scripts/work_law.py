"""Work growth law of hnswlib with mark-delete: W(p) = W0 * p ** (-beta).

p is the live share of graph vertices: under steady churn without slot reuse
p = n_base / (n_base + number of operations). beta is fitted by least squares in
logarithms (regression through the origin) for each filter; beta and the mean
relative prediction error are printed.

Example:
    .venv/bin/python scripts/work_law.py results/yfcc/churn_3x/hnswlib/steady_churn-mark.csv
    .venv/bin/python scripts/work_law.py results/pilot_3x/steady_churn-mark.csv --col latency_ms_mean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_beta(steps: np.ndarray, work: np.ndarray, n_base: int) -> tuple[float, float]:
    """beta and mean relative error; the first point is the fresh index (p = 1)."""
    p = n_base / (n_base + steps)
    ratio = work / work[0]
    x, y = -np.log(p[1:]), np.log(ratio[1:])
    beta = float((x * y).sum() / (x * x).sum())
    pred = p[1:] ** (-beta)
    return beta, float(np.abs(pred - ratio[1:]).mean() / ratio[1:].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--col", default="cmps_mean")
    ap.add_argument("--n-base", type=int, default=100_000)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d = d[d.checkpoint != "rebuild"]
    for flt, sub in d.groupby("filter", sort=False):
        sub = sub.sort_values("step")
        beta, err = fit_beta(sub.step.to_numpy(float), sub[a.col].to_numpy(float), a.n_base)
        print(f"{flt:>7}: beta = {beta:.2f}, средняя относительная ошибка {100 * err:.1f} %")


if __name__ == "__main__":
    main()
