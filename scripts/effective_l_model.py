"""Check of the "effective list length" model for Filtered-DiskANN without consolidation.

Hypothesis: lazily deleted vertices occupy places in the list of the L best candidates, so the
index behaves like a fresh one with L_eff = L * (live / all graph vertices). The script takes
recall of a fresh index at different L (results/diskann_L_sweep) and compares the prediction
with the trajectory of three-fold churn on three seeds.

The second version (if results/diskann_probe exists) takes L_eff not from the live share of
graph vertices but from a measurement: how many live vertices actually ended up in the
candidate list, normalized by the fresh index of the same run.

The third version adds navigation through deleted vertices with one parameter:
L_eff = L * p ** alpha, where p is the live share of graph vertices; alpha = 1 means crowding
only, alpha = 0 means deleted vertices cost nothing. alpha is fitted on the first seed
and checked on the others.

Example:
    .venv/bin/python scripts/effective_l_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SWEEP = Path("results/diskann_L_sweep")
RUNS = [
    Path("results/pilot_diskann_3x_L300"),
    Path("results/pilot_diskann_3x_L300_seed8"),
    Path("results/pilot_diskann_3x_L300_seed9"),
]
L_SEARCH = 300
N_BASE = 100_000
PROBE = Path("results/diskann_probe/steady_churn-lazy.csv")
# External check: a run with a different L on which alpha was not fitted.
OTHER_L_RUN = Path("results/pilot_diskann_3x")
OTHER_L = 64


def fresh_curve() -> pd.DataFrame:
    rows = []
    for path in sorted(SWEEP.glob("L*/steady_churn-lazy.csv")):
        d = pd.read_csv(path)
        base = d[d.checkpoint == "base"].set_index("filter").recall_mean
        rows.append({"L": int(path.parent.name[1:]), **base.to_dict()})
    return pd.DataFrame(rows).sort_values("L")


def main() -> None:
    curve = fresh_curve()
    rows = []
    for run in RUNS:
        d = pd.read_csv(run / "steady_churn-lazy.csv")
        d = d[(d.checkpoint != "rebuild") & d["filter"].isin(["none", "wide"])]
        for _, r in d.iterrows():
            l_eff = L_SEARCH * N_BASE / (N_BASE + r.step)
            pred = np.interp(l_eff, curve.L, curve[r["filter"]])
            rows.append((r["filter"], r.step, l_eff, r.recall_mean, pred))
    x = pd.DataFrame(rows, columns=["filter", "step", "L_eff", "observed", "predicted"])
    print(x.groupby(["filter", "step"])[["L_eff", "observed", "predicted"]].mean().round(3))
    for f, y in x.groupby("filter"):
        corr = np.corrcoef(y.observed, y.predicted)[0, 1]
        mae = (y.observed - y.predicted).abs().mean()
        print(f"{f}: корреляция {corr:.3f}, средняя абсолютная ошибка {mae:.4f}")
    if PROBE.exists():
        compare_with_probe(curve)
    fit_alpha(curve)


def compare_with_probe(curve: pd.DataFrame) -> None:
    """Model 1 (live share in the graph) versus model 2 (measured candidate list)."""
    d = pd.read_csv(PROBE)
    d = d[(d.checkpoint != "rebuild") & d["filter"].isin(["none", "wide"])]
    rows = []
    for f, y in d.groupby("filter"):
        base_alive = y[y.step == 0].list_alive_mean.iloc[0]
        for _, r in y.iterrows():
            graph_share = N_BASE / (N_BASE + r.step)
            list_share = r.list_alive_mean / base_alive
            rows.append({
                "filter": f,
                "step": r.step,
                "доля живых в графе": graph_share,
                "доля живых в списке": list_share,
                "наблюдение": r.recall_mean,
                "модель 1": np.interp(L_SEARCH * graph_share, curve.L, curve[f]),
                "модель 2": np.interp(L_SEARCH * list_share, curve.L, curve[f]),
            })  # fmt: skip
    x = pd.DataFrame(rows)
    print("\nСравнение с замером списка кандидатов (зерно 7):")
    print(x.round(3).to_string(index=False))
    for f, y in x.groupby("filter"):
        for m in ("модель 1", "модель 2"):
            end = y[y.step == y.step.max()].iloc[0]
            loss_obs = y.iloc[0]["наблюдение"] - end["наблюдение"]
            loss_pred = y.iloc[0][m] - end[m]
            mae = (y["наблюдение"] - y[m]).abs().mean()
            ratio = loss_pred / loss_obs if loss_obs > 0 else float("nan")
            print(f"{f}, {m}: MAE {mae:.4f}, предсказанная потеря / наблюдаемая = {ratio:.2f}")


def trajectories() -> dict[Path, pd.DataFrame]:
    out = {}
    for run in RUNS:
        d = pd.read_csv(run / "steady_churn-lazy.csv")
        out[run] = d[
            (d.checkpoint != "rebuild") & d["filter"].isin(["none", "wide"]) & (d.step > 0)
        ]
    return out


def predict(
    curve: pd.DataFrame, d: pd.DataFrame, alpha: float, l_search: int = L_SEARCH
) -> np.ndarray:
    share = N_BASE / (N_BASE + d.step.to_numpy())
    l_eff = np.log(l_search * share**alpha)
    logl = np.log(curve.L.to_numpy())
    return np.array([
        np.interp(le, logl, curve[f].to_numpy()) for le, f in zip(l_eff, d["filter"])
    ])  # fmt: skip


def fit_alpha(curve: pd.DataFrame) -> None:
    """Fits alpha on the first seed, checks it on the others; compares with alpha = 1."""
    runs = trajectories()
    train = RUNS[0]
    grid = np.linspace(0.0, 1.0, 101)
    errors = [
        np.mean((runs[train].recall_mean.to_numpy() - predict(curve, runs[train], a)) ** 2)
        for a in grid
    ]
    best = float(grid[int(np.argmin(errors))])
    print(f"\nМодель 3: alpha, подобранный на {train.name}: {best:.2f}")
    for run in RUNS:
        d = runs[run]
        role = "подбор" if run == train else "проверка"
        for alpha, name in ((1.0, "alpha = 1"), (best, f"alpha = {best:.2f}")):
            pred = predict(curve, d, alpha)
            for f in ("none", "wide"):
                m = (d["filter"] == f).to_numpy()
                obs = d.recall_mean.to_numpy()[m]
                mae = np.abs(obs - pred[m]).mean()
                print(f"  {run.name} ({role}), {f}, {name}: MAE {mae:.4f}")
    check_other_l(curve, best)


def check_other_l(curve: pd.DataFrame, alpha: float) -> None:
    """The same alpha on a run with a different L: checks that it is not tuned to L = 300."""
    path = OTHER_L_RUN / "steady_churn-lazy.csv"
    if not path.exists() or curve.L.min() > OTHER_L * N_BASE / (N_BASE + 300_000):
        print("\nВнешняя проверка пропущена: нет прогона или свежий индекс не покрывает малые L")
        return
    d = pd.read_csv(path)
    d = d[(d.checkpoint != "rebuild") & d["filter"].isin(["none", "wide"]) & (d.step > 0)]
    print(f"\nВнешняя проверка на {OTHER_L_RUN.name} (L = {OTHER_L}):")
    for a, name in ((1.0, "alpha = 1"), (alpha, f"alpha = {alpha:.2f}")):
        pred = predict(curve, d, a, l_search=OTHER_L)
        for f in ("none", "wide"):
            m = (d["filter"] == f).to_numpy()
            mae = np.abs(d.recall_mean.to_numpy()[m] - pred[m]).mean()
            print(f"  {f}, {name}: MAE {mae:.4f}")


if __name__ == "__main__":
    main()
