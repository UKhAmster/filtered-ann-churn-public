import numpy as np

from fanndyn.runbook.export import alive_at_checkpoints, encode_ops, with_base_checkpoint
from fanndyn.runbook.scenarios import build_scenario
from fanndyn.runbook.schema import Op, OpKind, Runbook


def _rb():
    return Runbook(
        name="t",
        seed=0,
        generator_version="2",
        ops=(
            Op(OpKind.INSERT, ids=(0, 1, 2)),
            Op(OpKind.SEARCH),
            Op(OpKind.DELETE, ids=(1,)),
            Op(OpKind.CHECKPOINT, tag="a"),
            Op(OpKind.INSERT, ids=(3,)),
            Op(OpKind.CHECKPOINT, tag="b"),
        ),
    )


def test_encode_ops_skips_searches_and_uses_kind_count_ids_layout():
    enc = encode_ops(_rb())
    assert enc.dtype == np.uint32
    assert enc.tolist() == [1, 3, 0, 1, 2, 2, 1, 1, 3, 0, 1, 1, 3, 3, 0]


def test_alive_at_checkpoints_follows_inserts_and_deletes():
    res = alive_at_checkpoints(_rb())
    assert [tag for tag, _, _ in res] == ["a", "b"]
    assert [a.tolist() for _, a, _ in res] == [[0, 2], [0, 2, 3]]
    assert [s for _, _, s in res] == [1, 1]


def test_with_base_checkpoint_adds_zero_point():
    labels = [(i % 3,) for i in range(40)]
    rb = build_scenario(
        "steady_churn", n_base=20, n_ops=10, label_sets=labels, seed=1, checkpoint_every=5
    )
    res = alive_at_checkpoints(with_base_checkpoint(rb))
    assert res[0][0] == "base" and len(res[0][1]) == 20 and res[0][2] == 0
