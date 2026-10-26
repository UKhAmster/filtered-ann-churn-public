from pathlib import Path

import pytest

from fanndyn.runbook.schema import Op, OpKind, Runbook, RunbookError, from_jsonl, to_jsonl, validate


def _rb() -> Runbook:
    return Runbook(
        name="tiny",
        seed=42,
        generator_version="1",
        ops=(
            Op(kind=OpKind.INSERT, ids=(0, 1, 2)),
            Op(kind=OpKind.SEARCH, tag="q0"),
            Op(kind=OpKind.CHECKPOINT, tag="c0"),
            Op(kind=OpKind.DELETE, ids=(1,)),
            Op(kind=OpKind.CHECKPOINT, tag="c1"),
        ),
    )


def test_roundtrip_preserves_runbook(tmp_path: Path):
    rb = _rb()
    path = tmp_path / "rb.jsonl"
    to_jsonl(rb, path)
    assert from_jsonl(path) == rb


def test_validate_rejects_delete_of_absent_id():
    bad = Runbook(
        name="bad",
        seed=1,
        generator_version="1",
        ops=(Op(kind=OpKind.DELETE, ids=(7,)),),
    )
    with pytest.raises(RunbookError, match="отсутствующ"):
        validate(bad)


def test_validate_requires_at_least_one_checkpoint():
    bad = Runbook(
        name="bad",
        seed=1,
        generator_version="1",
        ops=(Op(kind=OpKind.INSERT, ids=(0,)),),
    )
    with pytest.raises(RunbookError, match="контрольн"):
        validate(bad)
