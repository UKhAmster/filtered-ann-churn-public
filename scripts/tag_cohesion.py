"""Cohesion of a tag with the geometry: lift of the tag's share among nearest neighbors.

For a sample of objects with the tag, the 10 nearest neighbors in the whole collection (except
itself) are found; cohesion = share of neighbors with the same tag / share of the tag in the
collection. 1 means the tag is scattered at random; larger means its objects lie closer together.

Example:
    OMP_NUM_THREADS=1 .venv/bin/python scripts/tag_cohesion.py --tags 23,17,38,0,142,337,345,933,3808
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fanndyn.data import read_spmat, read_u8bin
from fanndyn.yfcc import label_mask


def cohesion(x: np.ndarray, mask: np.ndarray, sample: int, k: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ids = np.flatnonzero(mask)
    pick = rng.choice(ids, size=min(sample, len(ids)), replace=False)
    sq = (x**2).sum(axis=1)
    hits = 0
    for start in range(0, len(pick), 256):
        q = pick[start : start + 256]
        d = sq[q][:, None] - 2.0 * x[q] @ x.T + sq[None, :]
        d[np.arange(len(q)), q] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        hits += int(mask[nn].sum())
    return (hits / (len(pick) * k)) / mask.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "data/yfcc")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--sample", type=int, default=1000)
    a = ap.parse_args()
    x = read_u8bin(next(a.data.glob("base.*.u8bin")), count=a.n).astype(np.float32)
    ptr, idx = read_spmat(next(a.data.glob("base.metadata.*.spmat")))
    ptr, idx = ptr[: a.n + 1], idx[: int(ptr[a.n])]
    for tag in (int(t) for t in a.tags.split(",")):
        mask = label_mask(ptr, idx, tag)
        c = cohesion(x, mask, a.sample, 10, seed=tag)
        print(f"тег {tag:>5}: доля {100 * mask.mean():6.3f} %, сцепленность {c:6.1f}")


if __name__ == "__main__":
    main()
