import numpy as np

from fanndyn.eval.hnsw_adapter import HnswAdapter
from fanndyn.eval.runner import QueryFilter, run_runbook
from fanndyn.runbook.scenarios import build_scenario


def _setup(kind):
    rng = np.random.default_rng(3)
    n_base, n_ops = 1500, 600
    vectors = rng.standard_normal((n_base + n_ops, 16)).astype(np.float32)
    labels = rng.integers(0, 4, size=n_base + n_ops)
    rb = build_scenario(
        kind,
        n_base=n_base,
        n_ops=n_ops,
        label_sets=[(int(x),) for x in labels],
        seed=5,
        checkpoint_every=200,
        purge_label=1,
    )
    queries = rng.standard_normal((30, 16)).astype(np.float32)
    filters = [QueryFilter("all", None), QueryFilter("label0", 0)]
    return rb, vectors, labels, queries, filters


def _factory(mode):
    return lambda: HnswAdapter(dim=16, capacity=1500, mode=mode, ef_search=100, seed=0)


def test_rows_per_checkpoint_and_filter():
    rb, vectors, labels, queries, filters = _setup("steady_churn")
    rows = run_runbook(rb, vectors, labels, queries, filters, _factory("mark"), k=10)
    assert len(rows) == 3 * 2
    for row in rows:
        assert 0.0 <= row["recall_mean"] <= 1.0
        assert row["n_alive"] == 1500
    assert rows[0]["recall_mean"] > 0.9


def test_category_purge_removes_label_from_collection():
    rb, vectors, labels, queries, filters = _setup("category_purge")
    rows = run_runbook(rb, vectors, labels, queries, filters, _factory("replace"), k=10)
    assert rows[-1]["n_alive"] < 1500 + 600
    assert rows[-1]["n_deleted_total"] > 0


def test_input_labels_are_not_mutated_by_attr_update():
    rb, vectors, labels, queries, filters = _setup("attr_update")
    before = labels.copy()
    run_runbook(rb, vectors, labels, queries, filters, _factory("mark"), k=10)
    assert np.array_equal(labels, before)
