"""Selectivity sweep figure: metrics as a function of the tag share in the collection.

Reads derived.csv (scripts/derived_table.py) or derived_seeds.csv (scripts/aggregate_derived.py);
in the latter case the mean over repeats is drawn with a min-max range. Unfiltered search
is the point with share 1.
Points where fresh-index recall is below BROKEN are hollow: the method fails on that filter
before any updates, and degradation metrics are meaningless there.

Example:
    .venv/bin/python scripts/plot_selectivity.py results/yfcc/selectivity/derived.csv out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = [
    ("hnswlib", "mark", "hnswlib, удаление пометкой", "#1f5fa8", "o"),
    ("diskann", "lazy", "Filtered-DiskANN, без уплотнения", "#c2451e", "s"),
    ("diskann", "consolidate", "Filtered-DiskANN, уплотнение", "#e39a3b", "^"),
    ("ung", "rebuild", "UNG, перестройка при 20 %", "#2a7f5f", "D"),
]
BROKEN = 0.5
PANELS = [
    ("оракул", "полнота свежего индекса"),
    ("недодан, п.п.", "средний недодан полноты, п. п."),
    ("избыток работы, %", "средний избыток работы, %"),
]


def normalize(table: pd.DataFrame) -> pd.DataFrame:
    """Brings derived_seeds.csv to the columns of derived.csv plus range bounds."""
    if "оракул mean" not in table.columns:
        return table
    out = table[["метод", "режим", "фильтр"]].copy()
    out["доля"] = table["доля mean"]
    for col, _ in PANELS:
        out[col] = table[f"{col} mean"]
        out[f"{col} min"] = table[f"{col} min"]
        out[f"{col} max"] = table[f"{col} max"]
    return out


def main() -> None:
    table, out = normalize(pd.read_csv(Path(sys.argv[1]))), Path(sys.argv[2])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.6))
    for ax, (col, title) in zip(axes, PANELS):
        for method, mode, label, color, marker in SERIES:
            sub = table[(table["метод"] == method) & (table["режим"] == mode)].sort_values("доля")
            ok = sub["оракул"] >= BROKEN
            ax.plot(sub["доля"] * 100, sub[col], "-", color=color, lw=1.2, label=label)
            ax.plot(sub["доля"][ok] * 100, sub[col][ok], marker, color=color, ms=5)
            ax.plot(sub["доля"][~ok] * 100, sub[col][~ok].clip(upper=190), marker, ms=6,
                    markerfacecolor="white", markeredgecolor=color)  # fmt: skip
            if f"{col} min" in sub.columns:
                x = sub["доля"][ok] * 100
                ax.vlines(x, sub[f"{col} min"][ok], sub[f"{col} max"][ok], color=color, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("доля объектов с тегом, % (100 — без фильтра)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, which="both")
    axes[2].set_ylim(-10, 200)
    axes[1].text(0.02, 0.97, "полые точки: полнота свежего индекса < 0,5", transform=axes[1].transAxes,
                 fontsize=8, va="top", color="#555555")  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
