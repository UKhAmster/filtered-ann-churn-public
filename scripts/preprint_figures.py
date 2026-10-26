"""Рисунки препринта (на английском) из тех же результатов, что и рисунки ВКР.

fig_mechanisms.png — YFCC, трёхкратный оборот: полнота и работа на запрос по фильтрам;
fig_laws.png       — три закона деградации: наблюдение против модели;
fig_selectivity.png — показатели в зависимости от доли тега.

Пример:
    .venv/bin/python scripts/preprint_figures.py paper/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import effective_l_model as elm
from work_law import fit_beta

YFCC = Path("results/yfcc/churn_3x")
SERIES = [
    ("hnswlib", "mark", "HNSW (hnswlib), mark-delete", "#1f5fa8", "-"),
    ("hnswlib", "replace", "HNSW (hnswlib), slot reuse", "#6ea0d8", "--"),
    ("diskann", "lazy", "Filtered-DiskANN, lazy delete", "#c2451e", "-"),
    ("diskann", "consolidate", "Filtered-DiskANN, consolidate every 10k", "#e39a3b", "--"),
    ("ung", "rebuild", "UNG, rebuild at 20% deleted", "#2a7f5f", "-"),
]
FILTERS = {"none": "no filter", "wide": "tag, 19%", "narrow": "tag, 1.1%", "tiny": "tag, 0.4%"}
plt.rcParams.update({"font.size": 9})


def fig_mechanisms(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
    for col, (flt, title) in enumerate(FILTERS.items()):
        for method, mode, label, color, ls in SERIES:
            d = pd.read_csv(YFCC / method / f"steady_churn-{mode}.csv")
            sub = d[(d["filter"] == flt) & (d.checkpoint != "rebuild")]
            base = sub[sub.checkpoint == "base"].cmps_mean.iloc[0]
            axes[0][col].plot(
                sub.step / 1000, sub.recall_mean, ls, color=color, lw=1.2, label=label
            )
            axes[1][col].plot(sub.step / 1000, sub.cmps_mean / base, ls, color=color, lw=1.2)
        axes[0][col].set_title(title)
        axes[0][col].set_ylim(0.5, 1.01)
        axes[1][col].set_xlabel("update operations, thousands")
        for r in (0, 1):
            axes[r][col].grid(alpha=0.3)
    axes[0][0].set_ylabel("recall@10")
    axes[1][0].set_ylabel("distance computations\nper query (fresh = 1)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=200)


def fig_laws(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # (a) hnswlib: работа от доли живых вершин.
    ax = axes[0]
    d = pd.read_csv(YFCC / "hnswlib" / "steady_churn-mark.csv")
    d = d[d.checkpoint != "rebuild"]
    for (flt, title), color in zip(FILTERS.items(), ["#1f5fa8", "#6ea0d8", "#8a5fbf", "#333333"]):
        sub = d[d["filter"] == flt].sort_values("step")
        steps, work = sub.step.to_numpy(float), sub.cmps_mean.to_numpy(float)
        beta, _ = fit_beta(steps, work, 100_000)
        p = 100_000 / (100_000 + steps)
        ax.plot(p, work / work[0], "o", color=color, ms=3.5, label=f"{title}: beta={beta:.2f}")
        grid = np.linspace(p.min(), 1, 50)
        ax.plot(grid, grid ** (-beta), "-", color=color, lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("live fraction of graph vertices p")
    ax.set_ylabel("work per query (fresh = 1)")
    ax.set_title("(a) HNSW, mark-delete: W = W0 p^(-beta)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # (b) Filtered-DiskANN: полнота против модели L_eff = L p^alpha.
    ax = axes[1]
    curve = elm.fresh_curve()
    runs = elm.trajectories()
    for run, marker in zip(elm.RUNS, ["o", "s", "^"]):
        r = runs[run]
        for f, color in (("none", "#c2451e"), ("wide", "#e39a3b")):
            m = (r["filter"] == f).to_numpy()
            p1 = elm.predict(curve, r, 1.0)[m]
            pa = elm.predict(curve, r, 0.69)[m]
            obs = r.recall_mean.to_numpy()[m]
            ax.plot(obs, pa, marker, color=color, ms=3.5)
            ax.plot(obs, p1, marker, color=color, ms=3.5, mfc="white", alpha=0.6)
    lo = 0.9
    ax.plot([lo, 1], [lo, 1], "k-", lw=0.7)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("observed recall@10")
    ax.set_ylabel("predicted recall@10")
    ax.set_title("(b) Filtered-DiskANN, lazy: L_eff = L p^alpha")
    ax.plot([], [], "o", color="grey", ms=3.5, label="alpha = 0.69 (fit on seed 7)")
    ax.plot([], [], "o", color="grey", ms=3.5, mfc="white", label="alpha = 1 (pure crowding)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    # (c) UNG: средняя полнота против 1 - t/2 (SIFT, три зерна, три фильтра).
    ax = axes[2]
    for root, marker in zip(["results/pilot_ung_3x", "results/pilot_ung_3x_seed8", "results/pilot_ung_3x_seed9"],
                            ["o", "s", "^"]):  # fmt: skip
        for path in sorted(Path(root).glob("rebuild_at_*/steady_churn-rebuild.csv")):
            t = float(path.parent.name.removeprefix("rebuild_at_"))
            d = pd.read_csv(path)
            traj = d[(d.checkpoint != "rebuild") & (d.step > 0)]
            for f in ("none", "wide", "narrow"):
                ax.plot(
                    t, traj[traj["filter"] == f].recall_mean.mean(), marker, color="#2a7f5f", ms=3.5
                )
    grid = np.linspace(0.03, 0.22, 20)
    ax.plot(grid, 1 - grid / 2, "k-", lw=0.8, label="model 1 - t/2")
    ax.set_xlabel("rebuild threshold t (deleted / live)")
    ax.set_ylabel("time-averaged recall@10")
    ax.set_title("(c) UNG, periodic rebuild: 1 - t/2")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig_selectivity(out: Path) -> None:
    """Среднее по повторам с разбросом «минимум — максимум», если есть derived_seeds.csv."""
    seeds = Path("results/yfcc/selectivity/derived_seeds.csv")
    spread = seeds.exists()
    raw = pd.read_csv(seeds if spread else "results/yfcc/selectivity/derived.csv")
    panels = [("оракул", "fresh-index recall@10"), ("недодан, п.п.", "mean recall deficit, p.p."),
              ("избыток работы, %", "mean excess work, %")]  # fmt: skip
    table = raw[["метод", "режим", "фильтр"]].copy()
    table["доля"] = raw["доля mean"] if spread else raw["доля"]
    for col, _ in panels:
        table[col] = raw[f"{col} mean"] if spread else raw[col]
        if spread:
            table[f"{col} min"], table[f"{col} max"] = raw[f"{col} min"], raw[f"{col} max"]
    series = [
        ("hnswlib", "mark", "HNSW, mark-delete", "#1f5fa8", "o"),
        ("diskann", "lazy", "Filtered-DiskANN, lazy", "#c2451e", "s"),
        ("diskann", "consolidate", "Filtered-DiskANN, consolidate", "#e39a3b", "^"),
        ("ung", "rebuild", "UNG, rebuild at 20%", "#2a7f5f", "D"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, title) in zip(axes, panels):
        for method, mode, label, color, marker in series:
            sub = table[(table["метод"] == method) & (table["режим"] == mode)].sort_values("доля")
            ok = sub["оракул"] >= 0.5
            x = sub["доля"] * 100
            ax.plot(x, sub[col].clip(upper=190), "-", color=color, lw=1, label=label)
            ax.plot(x[ok], sub[col][ok], marker, color=color, ms=4)
            ax.plot(x[~ok], sub[col][~ok].clip(upper=190), marker, ms=5, mfc="white", mec=color)
            if spread:
                ax.vlines(x[ok], sub[f"{col} min"][ok], sub[f"{col} max"][ok], color=color, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("tag share of collection, % (100 = no filter)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[2].set_ylim(-10, 200)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig_mechanisms(out / "fig_mechanisms.png")
    fig_laws(out / "fig_laws.png")
    fig_selectivity(out / "fig_selectivity.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
