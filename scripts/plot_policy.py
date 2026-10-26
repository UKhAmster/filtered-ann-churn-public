"""Rebuild policy figure for a static index (UNG): recall sawtooth and trade-off.

Left: unfiltered recall over the first 60k operations at three rebuild thresholds.
Right: mean recall versus total build time; the line is the model 1 - t/2
(on average half of the threshold's worth of new objects is invisible).

Example:
    .venv/bin/python scripts/plot_policy.py results/pilot_ung_3x out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {0.2: "#c2451e", 0.1: "#e39a3b", 0.05: "#1f5fa8"}
MARKERS = {"none": "o", "wide": "s", "narrow": "^"}
FILTERS = {"none": "без фильтра", "wide": "широкий", "narrow": "узкий"}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = {}
    for path in sorted(root.glob("rebuild_at_*/steady_churn-rebuild.csv")):
        t = float(path.parent.name.removeprefix("rebuild_at_"))
        d = pd.read_csv(path)
        runs[t] = d[d.checkpoint != "rebuild"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    for t, d in sorted(runs.items(), reverse=True):
        sub = d[(d["filter"] == "none") & (d.step <= 60_000)]
        ax1.plot(sub.step / 1000, sub.recall_mean, "-", color=COLORS[t], lw=1.3,
                 label=f"перестройка при {t:.0%} удалённых")  # fmt: skip
    ax1.set_xlabel("операций изменения, тыс.")
    ax1.set_ylabel("recall@10 (без фильтра)")
    ax1.set_ylim(0.75, 1.01)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.set_title("UNG: полнота между перестройками", fontsize=10)

    for t, d in runs.items():
        traj = d[d.step > 0]
        build_s = traj[traj.step == traj.step.max()].consolidate_s_total.iloc[0]
        for flt, m in MARKERS.items():
            mean = traj[traj["filter"] == flt].recall_mean.mean()
            ax2.scatter(build_s, mean, color=COLORS[t], marker=m, s=36, zorder=3)
        ax2.hlines(1 - t / 2, build_s * 0.85, build_s * 1.15, color="black", lw=1.2, zorder=2)
    for flt, m in MARKERS.items():
        ax2.scatter([], [], color="grey", marker=m, label=FILTERS[flt])
    ax2.plot([], [], color="black", lw=1.2, label="модель 1 − t/2")
    ax2.set_xscale("log")
    ax2.set_xticks([60, 120, 250])
    ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("суммарное время построений за 300 тыс. операций, с")
    ax2.set_ylabel("средняя recall@10")
    ax2.set_ylim(0.88, 1.0)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("компромисс «стоимость — качество»", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
