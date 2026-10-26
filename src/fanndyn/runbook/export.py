"""Runbook export for external drivers and reconstruction of the collection state."""

from __future__ import annotations

import numpy as np

from fanndyn.runbook.schema import Op, OpKind, Runbook

# Operation codes in the driver's binary stream; SEARCH is not passed.
OP_CODES = {OpKind.INSERT: 1, OpKind.DELETE: 2, OpKind.CHECKPOINT: 3}


def with_base_checkpoint(rb: Runbook) -> Runbook:
    """Adds a checkpoint right after the initial load: time zero of the trajectory."""
    ops = (rb.ops[0], Op(kind=OpKind.CHECKPOINT, tag="base")) + rb.ops[1:]
    return Runbook(name=rb.name, seed=rb.seed, generator_version=rb.generator_version, ops=ops)


def encode_ops(rb: Runbook) -> np.ndarray:
    """uint32 stream: operation code, number of ids, the ids. Attribute updates are unsupported."""
    out: list[int] = []
    for op in rb.ops:
        if op.kind is OpKind.SEARCH:
            continue
        if op.kind not in OP_CODES:
            raise ValueError(f"driver does not support operation {op.kind.value}")
        out += [OP_CODES[op.kind], len(op.ids), *op.ids]
    return np.asarray(out, dtype=np.uint32)


def alive_at_checkpoints(rb: Runbook) -> list[tuple[str, np.ndarray, int]]:
    """For each checkpoint: tag, sorted live ids, number of steps done."""
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
