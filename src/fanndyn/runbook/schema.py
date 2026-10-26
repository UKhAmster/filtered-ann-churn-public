"""Типы операций runbook и их сериализация.

Runbook детерминирован: зерно и версия генератора записаны в заголовке,
поэтому состояние коллекции в любой контрольной точке восстановимо.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RunbookError(ValueError):
    """Runbook нарушает инварианты и непригоден для прогона."""


class OpKind(str, Enum):
    INSERT = "insert"
    DELETE = "delete"
    UPDATE_ATTR = "update_attr"
    SEARCH = "search"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class Op:
    kind: OpKind
    ids: tuple[int, ...] = ()
    labels: tuple[tuple[int, ...], ...] = ()
    tag: str = ""


@dataclass(frozen=True)
class Runbook:
    name: str
    seed: int
    generator_version: str
    ops: tuple[Op, ...] = field(default_factory=tuple)


def validate(rb: Runbook) -> None:
    """Проверяет инварианты: удаления и правки только по живым идентификаторам,
    идентификаторы не переиспользуются, есть хотя бы одна контрольная точка."""
    alive: set[int] = set()
    seen: set[int] = set()
    checkpoints = 0
    for i, op in enumerate(rb.ops):
        if op.kind is OpKind.INSERT:
            for oid in op.ids:
                if oid in seen:
                    raise RunbookError(f"операция {i}: повторное использование id {oid}")
                seen.add(oid)
                alive.add(oid)
        elif op.kind in (OpKind.DELETE, OpKind.UPDATE_ATTR):
            for oid in op.ids:
                if oid not in alive:
                    raise RunbookError(f"операция {i}: обращение к отсутствующему id {oid}")
            if op.kind is OpKind.DELETE:
                alive.difference_update(op.ids)
        elif op.kind is OpKind.CHECKPOINT:
            checkpoints += 1
    if checkpoints == 0:
        raise RunbookError("runbook не содержит ни одной контрольной точки")


def to_jsonl(rb: Runbook, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        header = {"name": rb.name, "seed": rb.seed, "generator_version": rb.generator_version}
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for op in rb.ops:
            fh.write(
                json.dumps(
                    {
                        "kind": op.kind.value,
                        "ids": list(op.ids),
                        "labels": [list(ls) for ls in op.labels],
                        "tag": op.tag,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def from_jsonl(path: Path) -> Runbook:
    with path.open(encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        ops = tuple(
            Op(
                kind=OpKind(rec["kind"]),
                ids=tuple(rec["ids"]),
                labels=tuple(tuple(ls) for ls in rec["labels"]),
                tag=rec["tag"],
            )
            for rec in (json.loads(line) for line in fh if line.strip())
        )
    return Runbook(
        name=header["name"],
        seed=header["seed"],
        generator_version=header["generator_version"],
        ops=ops,
    )
