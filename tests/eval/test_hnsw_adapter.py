import numpy as np
import pytest

from fanndyn.eval.hnsw_adapter import HnswAdapter


@pytest.fixture
def vectors():
    return np.random.default_rng(1).standard_normal((400, 8)).astype(np.float32)


@pytest.mark.parametrize("mode", ["mark", "replace"])
def test_deleted_ids_never_returned(vectors, mode):
    idx = HnswAdapter(dim=8, capacity=300, mode=mode, ef_search=200, seed=0)
    idx.insert(np.arange(300), vectors[:300])
    idx.delete(np.arange(0, 300, 2))
    idx.insert(np.arange(300, 400), vectors[300:400])
    found = idx.search(vectors[0], k=10, allowed=lambda i: True)
    assert len(found) == 10
    assert not any(i < 300 and i % 2 == 0 for i in found)


@pytest.mark.parametrize("mode", ["mark", "replace"])
def test_filter_is_respected(vectors, mode):
    idx = HnswAdapter(dim=8, capacity=400, mode=mode, ef_search=200, seed=0)
    idx.insert(np.arange(400), vectors)
    found = idx.search(vectors[5], k=10, allowed=lambda i: i % 7 == 0)
    assert len(found) == 10 and all(i % 7 == 0 for i in found)


def test_replace_mode_reuses_slots_without_growth(vectors):
    idx = HnswAdapter(dim=8, capacity=200, mode="replace", seed=0)
    idx.insert(np.arange(200), vectors[:200])
    idx.delete(np.arange(100))
    idx.insert(np.arange(200, 300), vectors[200:300])
    assert idx.capacity == 200


def test_returns_fewer_results_when_filter_is_too_narrow(vectors):
    idx = HnswAdapter(dim=8, capacity=400, mode="mark", ef_search=50, seed=0)
    idx.insert(np.arange(400), vectors)
    found = idx.search(vectors[0], k=10, allowed=lambda i: i in (3, 17))
    assert set(found) <= {3, 17}
