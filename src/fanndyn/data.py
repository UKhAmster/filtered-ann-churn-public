"""Загрузка наборов в формате TEXMEX и синтетические метки."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2


def read_fvecs(path: Path, count: int | None = None) -> np.ndarray:
    """Читает .fvecs: каждая запись — int32 размерность, затем столько же float32."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    if count is not None:
        rows = rows[:count]
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def read_u8bin(path: Path, count: int | None = None) -> np.ndarray:
    """Читает .u8bin из big-ann-benchmarks: int32 n, int32 d, затем n*d байт."""
    with Path(path).open("rb") as fh:
        n, d = np.fromfile(fh, dtype=np.int32, count=2)
        n = int(n) if count is None else min(int(n), count)
        data = np.fromfile(fh, dtype=np.uint8, count=n * int(d))
    return data.reshape(n, int(d))


def read_spmat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Читает .spmat (CSR): int64 nrow, ncol, nnz; int64 indptr; int32 indices; float32 data.

    Возвращает indptr и indices; значения не нужны — в YFCC это единицы наличия тега.
    """
    with Path(path).open("rb") as fh:
        nrow, _ncol, nnz = np.fromfile(fh, dtype=np.int64, count=3)
        indptr = np.fromfile(fh, dtype=np.int64, count=int(nrow) + 1)
        indices = np.fromfile(fh, dtype=np.int32, count=int(nnz))
    return indptr, indices


def write_spmat(path: Path, indptr: np.ndarray, indices: np.ndarray, ncol: int) -> None:
    """Пишет CSR в формате .spmat; значения — единицы."""
    nnz = int(indptr[-1])
    with Path(path).open("wb") as fh:
        np.array([len(indptr) - 1, ncol, nnz], dtype=np.int64).tofile(fh)
        np.asarray(indptr, dtype=np.int64).tofile(fh)
        np.asarray(indices[:nnz], dtype=np.int32).tofile(fh)
        np.ones(nnz, dtype=np.float32).tofile(fh)


def zipf_labels(n: int, n_labels: int, s: float, seed: int) -> np.ndarray:
    """Одна метка на объект, частота метки i пропорциональна 1 / (i + 1)^s.

    Метки не связаны с геометрией векторов — как в большинстве бенчмарков
    фильтрованного поиска с синтетическими атрибутами.
    """
    p = 1.0 / np.arange(1, n_labels + 1) ** s
    return np.random.default_rng(seed).choice(n_labels, size=n, p=p / p.sum())


def kmeans_labels(
    vectors: np.ndarray, n_labels: int, seed: int, sample: int = 20_000, chunk: int = 50_000
) -> np.ndarray:
    """Одна метка на объект — номер кластера k-means, то есть метки связаны с геометрией.

    Центры обучаются на подвыборке, затем каждый объект относится к ближайшему центру.
    Метки упорядочены по убыванию размера кластера: 0 — самый крупный.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(sample, len(vectors)), replace=False)
    centers, _ = kmeans2(vectors[idx].astype(np.float64), n_labels, minit="++", seed=seed)
    centers = centers.astype(np.float32)
    raw = np.empty(len(vectors), dtype=np.int64)
    c_sq = (centers**2).sum(axis=1)
    for start in range(0, len(vectors), chunk):
        part = vectors[start : start + chunk]
        raw[start : start + chunk] = np.argmin(c_sq[None, :] - 2.0 * part @ centers.T, axis=1)
    order = np.argsort(-np.bincount(raw, minlength=n_labels), kind="stable")
    rank = np.empty(n_labels, dtype=np.int64)
    rank[order] = np.arange(n_labels)
    return rank[raw]


LABEL_KINDS = ("zipf", "kmeans")


def make_labels(kind: str, vectors: np.ndarray, n_labels: int, seed: int) -> np.ndarray:
    """Метки для пилотов: синтетические (zipf) или связанные с геометрией (kmeans)."""
    if kind == "zipf":
        return zipf_labels(len(vectors), n_labels, s=1.0, seed=seed)
    if kind == "kmeans":
        return kmeans_labels(vectors, n_labels, seed=seed)
    raise ValueError(f"неизвестный вид меток: {kind}")
