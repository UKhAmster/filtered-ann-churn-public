import numpy as np

from fanndyn.eval.metrics import filtered_ground_truth, recall_per_query


def _data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((500, 8)).astype(np.float32), rng.standard_normal((20, 8)).astype(
        np.float32
    )


def test_ground_truth_matches_brute_force_over_allowed_ids():
    base, queries = _data()
    allowed = np.arange(0, 500, 3)
    gt = filtered_ground_truth(queries, base, allowed, k=5)
    for q, row in zip(queries, gt):
        d = ((base[allowed] - q) ** 2).sum(axis=1)
        expected = allowed[np.argsort(d)[:5]]
        assert list(row) == list(expected)


def test_ground_truth_is_shorter_when_few_ids_allowed():
    base, queries = _data()
    gt = filtered_ground_truth(queries, base, np.array([7, 9]), k=5)
    assert all(len(row) == 2 for row in gt)


def test_recall_counts_overlap_against_available_truth():
    gt = [np.array([1, 2, 3, 4]), np.array([5, 6])]
    found = [np.array([1, 2, 9, 8]), np.array([6])]
    assert recall_per_query(found, gt).tolist() == [0.5, 0.5]


def test_recall_skips_queries_with_empty_truth():
    rec = recall_per_query([np.array([1])], [np.array([], dtype=np.int64)])
    assert np.isnan(rec[0])
