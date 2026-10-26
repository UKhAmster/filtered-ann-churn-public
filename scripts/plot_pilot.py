"""Pilot plots: recall and latency versus the number of operations.

Example:
    .venv/bin/python scripts/plot_pilot.py results/pilot
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FILTERS = {
    "none": "без фильтра",
    "wide": "широкий фильтр (~28 %)",
    "narrow": "узкий фильтр (~1,4 %)",
}
SCENARIOS = {"steady_churn": "стационарный оборот", "category_purge": "удаление категории"}
MODES = {
    "mark": ("удаление пометкой", "#1f5fa8"),
    "replace": ("переиспользование слотов", "#c2451e"),
}


def load(folder: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(folder.glob("*.csv")):
        scenario, mode = path.stem.rsplit("-", 1)
        frames.append(pd.read_csv(path).assign(scenario=scenario, mode=mode))
    return pd.concat(frames, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    scen = [s for s in SCENARIOS if s in set(df.scenario)]
    fig, axes = plt.subplots(len(scen), 3, figsize=(13, 3.6 * len(scen)), squeeze=False)
    for r, s in enumerate(scen):
        for c, (f, ftitle) in enumerate(FILTERS.items()):
            ax = axes[r][c]
            for mode, (mtitle, color) in MODES.items():
                sub = df[(df.scenario == s) & (df["filter"] == f) & (df["mode"] == mode)]
                if sub.empty:
                    continue
                traj = sub[sub.checkpoint != "rebuild"]
                ax.plot(traj.step / 1000, traj[metric], "o-", color=color, ms=3, label=mtitle)
                if metric == "recall_mean":
                    ax.plot(traj.step / 1000, traj.recall_p5, ":", color=color, lw=1)
                reb = sub[sub.checkpoint == "rebuild"]
                if not reb.empty:
                    ax.axhline(reb[metric].iloc[0], color=color, ls="--", lw=0.8, alpha=0.7)
            if metric == "recall_mean":
                ax.set_ylim(0.85, 1.005)
            ax.set_title(f"{SCENARIOS[s]}: {ftitle}", fontsize=10)
            ax.set_xlabel("операций изменения, тыс.")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.text(
        0.5,
        0.005,
        "сплошная — среднее; пунктир — 5-й перцентиль по запросам; штрих — индекс, "
        "перестроенный заново на итоговой коллекции",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150)
    print("saved", out)


def main() -> None:
    folder = Path(sys.argv[1])
    df = load(folder)
    plot_metric(df, "recall_mean", "recall@10", folder / "recall.png")
    plot_metric(df, "latency_ms_mean", "задержка, мс", folder / "latency.png")


if __name__ == "__main__":
    main()
