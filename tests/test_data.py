import numpy as np
import pytest

from fanndyn.data import (
    kmeans_labels,
    make_labels,
    read_fvecs,
    read_spmat,
    read_u8bin,
    write_spmat,
    zipf_labels,
)


def test_read_fvecs_roundtrip(tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw = np.hstack([np.full((3, 1), 4, dtype=np.int32).view(np.float32), x])
    path = tmp_path / "x.fvecs"
    raw.tofile(path)
    assert np.array_equal(read_fvecs(path), x)
    assert np.array_equal(read_fvecs(path, count=2), x[:2])


def test_zipf_labels_are_skewed_and_deterministic():
    a = zipf_labels(20_000, n_labels=20, s=1.0, seed=4)
    assert np.array_equal(a, zipf_labels(20_000, n_labels=20, s=1.0, seed=4))
    freq = np.bincount(a, minlength=20) / len(a)
    assert freq[0] > 0.2 and freq[19] < 0.03


def test_kmeans_labels_follow_geometry_and_are_ordered_by_size():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    sizes = [500, 300, 100]
    x = np.vstack([c + rng.standard_normal((n, 2)) for c, n in zip(centers, sizes)]).astype(
        np.float32
    )
    labels = kmeans_labels(x, n_labels=3, seed=1)
    assert np.bincount(labels).tolist() == [500, 300, 100]
    assert len(set(labels[:500])) == 1 and len(set(labels[500:800])) == 1
    assert np.array_equal(labels, kmeans_labels(x, n_labels=3, seed=1))


def test_make_labels_dispatches_and_rejects_unknown_kind():
    x = np.random.default_rng(0).standard_normal((300, 4)).astype(np.float32)
    assert np.array_equal(make_labels("zipf", x, 5, seed=2), zipf_labels(300, 5, s=1.0, seed=2))
    assert np.array_equal(make_labels("kmeans", x, 5, seed=2), kmeans_labels(x, 5, seed=2))
    with pytest.raises(ValueError):
        make_labels("other", x, 5, seed=2)


def test_read_u8bin_with_prefix(tmp_path):
    x = np.arange(24, dtype=np.uint8).reshape(4, 6)
    path = tmp_path / "x.u8bin"
    with path.open("wb") as fh:
        np.array([4, 6], dtype=np.int32).tofile(fh)
        x.tofile(fh)
    assert np.array_equal(read_u8bin(path), x)
    assert np.array_equal(read_u8bin(path, count=2), x[:2])


def test_spmat_roundtrip_matches_big_ann_layout(tmp_path):
    indptr = np.array([0, 2, 2, 5], dtype=np.int64)
    indices = np.array([3, 7, 1, 2, 9], dtype=np.int32)
    path = tmp_path / "m.spmat"
    write_spmat(path, indptr, indices, ncol=10)
    raw = np.fromfile(path, dtype=np.int64, count=3)
    assert raw.tolist() == [3, 10, 5]
    got_ptr, got_idx = read_spmat(path)
    assert got_ptr.tolist() == indptr.tolist() and got_idx.tolist() == indices.tolist()
