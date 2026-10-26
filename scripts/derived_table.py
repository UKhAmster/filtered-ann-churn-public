"""Производные показатели (спецификация, раздел 4.3) по результатам прогонов.

Для каждой пары (прогон, фильтр):
  * оракул — полнота свежего индекса: среднее контрольной точки base и индекса, перестроенного
    на итоговой коллекции (в стационарном обороте состав коллекции статистически не меняется);
  * средний недодан полноты — площадь потерь, делённая на длину прогона в операциях;
  * наработка до нарушения — операции до первой контрольной точки с полнотой ниже
    «полнота в начале прогона − порог» (разрешение — шаг контрольных точек); отсчёт от
    собственного старта, потому что у узких фильтров base и перестроенный индекс различаются
    сильнее порога из-за шума малых выборок запросов;
  * средний избыток работы — то же для числа вычислений расстояний относительно base;
  * обслуживание — суммарное время уплотнений или перестроек за прогон и время полной
    перестройки опорного индекса.

Пример:
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
    p.add_argument("--drop", type=float, default=0.02, help="допустимая потеря полноты от оракула")
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
