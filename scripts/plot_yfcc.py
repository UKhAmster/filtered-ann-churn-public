"""Графики замеров на YFCC: полнота и работа на запрос по мере обновления коллекции.

Работа — среднее число вычислений расстояний на запрос относительно свежего индекса
(контрольная точка base того же прогона): величина не зависит от машины.

Пример:
    .venv/bin/python scripts/plot_yfcc.py results/yfcc/churn_3x results/yfcc/churn_3x
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
    ("ung", "rebuild", "UNG, перестройка при 20 % удалённых", "#2a7f5f", "-"),
]
FILTERS = {
    "none": "без фильтра",
    "wide": "тег 19 %",
    "narrow": "тег 1,1 %",
    "tiny": "тег 0,4 %",
}


def load(root: Path, method: str, scenario: str, mode: str) -> pd.DataFrame | None:
    path = root / method / f"{scenario}-{mode}.csv"
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    scenario = sys.argv[3] if len(sys.argv) > 3 else "steady_churn"
    fig, axes = plt.subplots(2, len(FILTERS), figsize=(15, 6.6), sharex=True)
    for col, (flt, title) in enumerate(FILTERS.items()):
        for method, mode, label, color, ls in SERIES:
            d = load(root, method, scenario, mode)
            if d is None:
                continue
            sub = d[(d["filter"] == flt) & (d.checkpoint != "rebuild")]
            x = sub.step / 1000
            axes[0][col].plot(x, sub.recall_mean, ls, color=color, lw=1.4, label=label)
            base = sub[sub.checkpoint == "base"].cmps_mean.iloc[0]
            axes[1][col].plot(x, sub.cmps_mean / base, ls, color=color, lw=1.4)
        axes[0][col].set_title(title, fontsize=11)
        axes[0][col].set_ylim(0.5, 1.01)
        axes[1][col].set_xlabel("операций изменения, тыс.")
        for r in (0, 1):
            axes[r][col].grid(alpha=0.3)
    axes[0][0].set_ylabel("recall@10")
    axes[1][0].set_ylabel("работа на запрос,\nот свежего индекса")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = out_dir / f"{scenario}.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
