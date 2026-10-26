import numpy as np

from fanndyn.data import write_spmat
from fanndyn.yfcc import label_mask, queries_with_tag, reduce_tags


def _csr(rows):
    ptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
    idx = np.array([t for r in rows for t in r], dtype=np.int32)
    return ptr, idx


def test_reduce_tags_keeps_selected_and_marks_rest_as_other():
    ptr, idx = _csr([[5, 9], [1], [9, 7, 5], []])
    new_ptr, new_idx = reduce_tags(ptr, idx, keep=[9, 5])
    rows = [new_idx[new_ptr[i] : new_ptr[i + 1]].tolist() for i in range(4)]
    # 9 -> 0, 5 -> 1, other -> 2; labels of an object are sorted: DiskANN compares
    # label lists with two pointers and requires sorting.
    assert rows == [[0, 1], [2], [0, 1], [2]]


def test_label_mask_marks_rows_containing_label():
    ptr, idx = _csr([[0, 2], [1], [2], [0]])
    assert label_mask(ptr, idx, 2).tolist() == [True, False, True, False]


def test_queries_with_tag_returns_first_matching_rows(tmp_path):
    ptr, idx = _csr([[3], [4, 3], [4], [3]])
    path = tmp_path / "q.spmat"
    write_spmat(path, ptr, idx, ncol=10)
    assert queries_with_tag(path, 3, limit=2).tolist() == [0, 1]
    assert queries_with_tag(path, 4, limit=5).tolist() == [1, 2]
