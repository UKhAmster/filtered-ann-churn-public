"""Recall trajectories of hnswlib and Filtered-DiskANN on one scenario.

Example:
    .venv/bin/python scripts/plot_compare.py results/pilot_3x results/pilot_diskann_3x out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = [
    ("hnswlib", "mark", "hnswlib, удаление пометкой", "#1f5fa8", "-"),
    ("hnswlib", "replace", "hnswlib, переиспользование мест", "#6ea0d8", "--"),
    ("diskann", "lazy", "Filtered-DiskANN, без уплотнения", "#c2451e", "-"),
    ("diskann", "consolidate", "Filtered-DiskANN, уплотнение каждые 10 тыс.", "#e39a3b", "--"),
]
PANELS = [("none", "без фильтра"), ("wide", "широкий фильтр (~28 %)")]


def main() -> None:
    hnsw_dir, diskann_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dirs = {"hnswlib": hnsw_dir, "diskann": diskann_dir}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 4.8))
    for ax, (flt, title) in zip(axes, PANELS):
        for method, mode, label, color, ls in SERIES:
            df = pd.read_csv(dirs[method] / f"steady_churn-{mode}.csv")
            sub = df[(df["filter"] == flt) & (df.checkpoint != "rebuild")]
            ax.plot(
                sub.step / 1000, sub.recall_mean, ls, marker="o", ms=3, color=color, label=label
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("операций изменения, тыс.")
        ax.set_ylabel("recall@10")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
