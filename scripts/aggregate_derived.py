"""Сводка производных показателей по повторам: среднее, минимум и максимум по зёрнам.

Читает derived.csv из каждого каталога (их строит scripts/derived_table.py) и пишет
derived_seeds.csv в первый каталог.

Пример:
    .venv/bin/python scripts/aggregate_derived.py results/yfcc/selectivity \
        results/yfcc/selectivity_seed8 results/yfcc/selectivity_seed9
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEYS = ["метод", "режим", "фильтр"]
VALUES = ["доля", "оракул", "недодан, п.п.", "наработка до нарушения, тыс.", "избыток работы, %"]


def main() -> None:
    dirs = [Path(x) for x in sys.argv[1:]]
    frames = [pd.read_csv(d / "derived.csv").assign(прогон=d.name) for d in dirs]
    t = pd.concat(frames, ignore_index=True)
    agg = t.groupby(KEYS, sort=False)[VALUES].agg(["mean", "min", "max"])
    agg.columns = [f"{c} {fn}" for c, fn in agg.columns]
    agg.insert(0, "повторов", t.groupby(KEYS, sort=False).size())
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string())
    out = dirs[0] / "derived_seeds.csv"
    agg.reset_index().to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
