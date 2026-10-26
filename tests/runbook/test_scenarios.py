import pytest

from fanndyn.runbook.scenarios import SCENARIOS, build_scenario
from fanndyn.runbook.schema import OpKind, validate

LABEL_SETS = [(i % 5,) for i in range(200)]


def _build(kind: str, **kw):
    params = {"purge_label": 3} if kind == "category_purge" else {}
    params.update(kw)
    return build_scenario(
        kind,
        n_base=100,
        n_ops=60,
        label_sets=LABEL_SETS,
        seed=7,
        checkpoint_every=20,
        **params,
    )


@pytest.mark.parametrize("kind", SCENARIOS)
def test_every_scenario_is_valid(kind: str):
    validate(_build(kind))


@pytest.mark.parametrize("kind", SCENARIOS)
def test_generation_is_deterministic(kind: str):
    assert _build(kind) == _build(kind)


def test_steady_churn_keeps_collection_size_stable():
    rb = _build("steady_churn")
    size = 0
    sizes = []
    for op in rb.ops:
        if op.kind is OpKind.INSERT:
            size += len(op.ids)
        elif op.kind is OpKind.DELETE:
            size -= len(op.ids)
        elif op.kind is OpKind.CHECKPOINT:
            sizes.append(size)
    assert max(sizes) - min(sizes) <= 1


def test_category_purge_removes_only_target_label():
    rb = _build("category_purge")
    purged = {oid for op in rb.ops if op.kind is OpKind.DELETE for oid in op.ids}
    assert purged
    assert all(LABEL_SETS[oid] == (3,) for oid in purged)


@pytest.mark.parametrize("kind", SCENARIOS)
def test_generation_scales_to_pilot_size(kind: str):
    import time

    n = 50_000
    labels = [(i % 20,) for i in range(3 * n)]
    params = {"purge_label": 1} if kind == "category_purge" else {}
    t0 = time.perf_counter()
    rb = build_scenario(
        kind, n_base=n, n_ops=n, label_sets=labels, seed=1, checkpoint_every=10_000, **params
    )
    assert time.perf_counter() - t0 < 10.0
    validate(rb)
