"""Экспорт runbook для внешних драйверов и восстановление состояния коллекции."""

from __future__ import annotations

import numpy as np

from fanndyn.runbook.schema import Op, OpKind, Runbook

# Коды операций в бинарном потоке драйвера; SEARCH не передаётся.
OP_CODES = {OpKind.INSERT: 1, OpKind.DELETE: 2, OpKind.CHECKPOINT: 3}


def with_base_checkpoint(rb: Runbook) -> Runbook:
    """Добавляет контрольную точку сразу после начальной загрузки — нулевой момент траектории."""
    ops = (rb.ops[0], Op(kind=OpKind.CHECKPOINT, tag="base")) + rb.ops[1:]
    return Runbook(name=rb.name, seed=rb.seed, generator_version=rb.generator_version, ops=ops)


def encode_ops(rb: Runbook) -> np.ndarray:
    """Поток uint32: код операции, число id, сами id. Правки атрибутов не поддерживаются."""
    out: list[int] = []
    for op in rb.ops:
        if op.kind is OpKind.SEARCH:
            continue
        if op.kind not in OP_CODES:
            raise ValueError(f"драйвер не поддерживает операцию {op.kind.value}")
        out += [OP_CODES[op.kind], len(op.ids), *op.ids]
    return np.asarray(out, dtype=np.uint32)


def alive_at_checkpoints(rb: Runbook) -> list[tuple[str, np.ndarray, int]]:
    """Для каждой контрольной точки: тег, отсортированные живые id, число пройденных шагов."""
    alive: set[int] = set()
    steps = 0
    out = []
    for op in rb.ops:
        if op.kind is OpKind.INSERT:
            alive.update(op.ids)
        elif op.kind is OpKind.DELETE:
            alive.difference_update(op.ids)
        elif op.kind is OpKind.SEARCH:
            steps += 1
        elif op.kind is OpKind.CHECKPOINT:
            out.append((op.tag, np.array(sorted(alive), dtype=np.int64), steps))
    return out
