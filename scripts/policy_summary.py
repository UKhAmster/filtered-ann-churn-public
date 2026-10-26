"""Сравнение порогов перестройки статического индекса (UNG) под стационарным оборотом.

Для каждого порога: средняя по времени полнота и доля запросов с полнотой ниже 0,9
(усреднение по контрольным точкам, шаг равномерный), число построений индекса (включая первое) и их суммарное время.

Пример:
    .venv/bin/python scripts/policy_summary.py results/pilot_ung_3x
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for path in sorted(root.glob("rebuild_at_*/steady_churn-rebuild.csv")):
        threshold = float(path.parent.name.removeprefix("rebuild_at_"))
        d = pd.read_csv(path)
        traj = d[(d.checkpoint != "rebuild") & (d.step > 0)]
        last = traj[traj.step == traj.step.max()].iloc[0]
        for flt, sub in traj.groupby("filter"):
            rows.append({
                "порог": threshold,
                "фильтр": flt,
                "полнота, среднее": sub.recall_mean.mean(),
                "полнота, минимум": sub.recall_mean.min(),
                "доля < 0,9, среднее": sub.frac_below_0_9.mean(),
                "построений всего": int(last.consolidations),
                "время построений, с": last.consolidate_s_total,
            })  # fmt: skip
    table = pd.DataFrame(rows).sort_values(["фильтр", "порог"], ascending=[True, False])
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
