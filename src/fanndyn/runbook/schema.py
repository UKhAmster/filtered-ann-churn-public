"""Runbook operation types and their serialization.

A runbook is deterministic: the seed and generator version are stored in its header,
so the state of the collection at any checkpoint can be reconstructed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RunbookError(ValueError):
    """The runbook violates its invariants and cannot be run."""


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
    """Checks invariants: deletions and updates only touch live ids,
    ids are never reused, and there is at least one checkpoint."""
    alive: set[int] = set()
    seen: set[int] = set()
    checkpoints = 0
    for i, op in enumerate(rb.ops):
        if op.kind is OpKind.INSERT:
            for oid in op.ids:
                if oid in seen:
                    raise RunbookError(f"operation {i}: id {oid} reused")
                seen.add(oid)
                alive.add(oid)
        elif op.kind in (OpKind.DELETE, OpKind.UPDATE_ATTR):
            for oid in op.ids:
                if oid not in alive:
                    raise RunbookError(f"operation {i}: reference to missing id {oid}")
            if op.kind is OpKind.DELETE:
                alive.difference_update(op.ids)
        elif op.kind is OpKind.CHECKPOINT:
            checkpoints += 1
    if checkpoints == 0:
        raise RunbookError("runbook contains no checkpoint")


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
