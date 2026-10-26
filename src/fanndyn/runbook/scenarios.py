"""Пять сценариев динамики из раздела 3.2 спецификации.

Ключевое различие между сценариями — характер удалений. Случайные удаления
и удаление связной группы по метке воздействуют на граф по-разному; вторая
разновидность (category_purge) в опубликованных бенчмарках отсутствует.
"""

from __future__ import annotations

import numpy as np

from fanndyn.runbook.schema import Op, OpKind, Runbook, validate

# Версия 2: живые идентификаторы хранятся в пуле с удалением за O(1).
# Порядок в пуле отличается от версии 1, поэтому при том же зерне выбор жертв иной.
GENERATOR_VERSION = "2"

SCENARIOS: tuple[str, ...] = (
    "steady_churn",
    "growth",
    "category_purge",
    "attr_update",
    "insert_drift",
)


class _AlivePool:
    """Множество живых id с равновероятным выбором и удалением за O(1)."""

    def __init__(self, ids: range) -> None:
        self._items = list(ids)
        self._pos = {oid: i for i, oid in enumerate(self._items)}

    def add(self, oid: int) -> None:
        self._pos[oid] = len(self._items)
        self._items.append(oid)

    def remove(self, oid: int) -> None:
        i = self._pos.pop(oid)
        last = self._items.pop()
        if last != oid:
            self._items[i] = last
            self._pos[last] = i

    def choice(self, rng: np.random.Generator) -> int:
        return self._items[int(rng.integers(len(self._items)))]

    def __iter__(self):
        return iter(list(self._items))


def build_scenario(
    kind: str,
    *,
    n_base: int,
    n_ops: int,
    label_sets: list[tuple[int, ...]],
    seed: int,
    checkpoint_every: int,
    **params: object,
) -> Runbook:
    if kind not in SCENARIOS:
        raise ValueError(f"неизвестный сценарий: {kind}")
    rng = np.random.default_rng(seed)
    ops: list[Op] = [Op(kind=OpKind.INSERT, ids=tuple(range(n_base)))]
    alive = _AlivePool(range(n_base))
    next_id = n_base
    target = int(params.get("purge_label", 0))
    # Для category_purge: живые объекты с удаляемой меткой, ожидающие удаления.
    doomed = (
        [i for i in range(n_base) if target in label_sets[i]] if kind == "category_purge" else []
    )

    for step in range(n_ops):
        if kind == "steady_churn":
            victim = alive.choice(rng)
            alive.remove(victim)
            ops.append(Op(kind=OpKind.DELETE, ids=(victim,)))
            ops.append(Op(kind=OpKind.INSERT, ids=(next_id,)))
            alive.add(next_id)
            next_id += 1
        elif kind == "growth":
            ops.append(Op(kind=OpKind.INSERT, ids=(next_id,)))
            alive.add(next_id)
            next_id += 1
            if step % 10 == 9:
                victim = alive.choice(rng)
                alive.remove(victim)
                ops.append(Op(kind=OpKind.DELETE, ids=(victim,)))
        elif kind == "category_purge":
            if doomed:
                for i in doomed:
                    alive.remove(i)
                ops.append(Op(kind=OpKind.DELETE, ids=tuple(doomed)))
            ops.append(Op(kind=OpKind.INSERT, ids=(next_id,)))
            alive.add(next_id)
            doomed = [next_id] if target in label_sets[next_id] else []
            next_id += 1
        elif kind == "attr_update":
            target_id = alive.choice(rng)
            new_label = (int(rng.integers(0, 5)),)
            ops.append(Op(kind=OpKind.UPDATE_ATTR, ids=(target_id,), labels=(new_label,)))
        elif kind == "insert_drift":
            ops.append(Op(kind=OpKind.INSERT, ids=(next_id,), tag="drift"))
            alive.add(next_id)
            next_id += 1

        ops.append(Op(kind=OpKind.SEARCH, tag=f"s{step}"))
        if (step + 1) % checkpoint_every == 0:
            ops.append(Op(kind=OpKind.CHECKPOINT, tag=f"c{step}"))

    if ops[-1].kind is not OpKind.CHECKPOINT:
        ops.append(Op(kind=OpKind.CHECKPOINT, tag="final"))

    rb = Runbook(
        name=f"{kind}-n{n_base}-o{n_ops}-s{seed}",
        seed=seed,
        generator_version=GENERATOR_VERSION,
        ops=tuple(ops),
    )
    validate(rb)
    return rb
