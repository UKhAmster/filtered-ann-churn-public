"""Download of a YFCC-10M prefix (filtered track of big-ann-benchmarks NeurIPS'23).

Takes the first N base vectors and their tags, by partial download with byte ranges instead
of fetching the whole 2.9 GB, and the full public queries with tags. The script uses only
the standard library: from the testbed network the CDN served data at hundreds of bytes
per second, so the download ran on another machine and the files were copied to the testbed.
The result is checked with fanndyn.data.read_u8bin and read_spmat.

Example:
    python3 scripts/fetch_yfcc.py --n 1000000 --out ~/data/yfcc
"""

from __future__ import annotations

import argparse
import shutil
import struct
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
DIM = 192
CHUNK = 1 << 22


def copy_range(name: str, start: int, end: int, out) -> None:
    """Appends bytes [start, end) of a dataset file to out."""
    req = urllib.request.Request(BASE_URL + name, headers={"Range": f"bytes={start}-{end - 1}"})
    got = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            got += len(chunk)
    if got != end - start:
        raise OSError(f"{name}: expected {end - start} bytes, got {got}")


def read_range(name: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(BASE_URL + name, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    if len(data) != end - start:
        raise OSError(f"{name}: expected {end - start} bytes, got {len(data)}")
    return data


def fetch_whole(name: str, dst: Path, attempts: int = 5) -> None:
    """Whole file, with retries: the CDN connection sometimes drops during the TLS handshake."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=600) as resp, dst.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--out", type=Path, default=Path.home() / "data/yfcc")
    a = p.parse_args()
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.n // 1_000_000}M" if a.n % 1_000_000 == 0 else str(a.n)

    with (out / f"base.{tag}.u8bin").open("wb") as fh:
        fh.write(struct.pack("<ii", a.n, DIM))
        copy_range("base.10M.u8bin", 8, 8 + a.n * DIM, fh)
    print("векторы базы:", a.n, flush=True)

    meta = "base.metadata.10M.spmat"
    nrow, ncol, nnz = struct.unpack("<qqq", read_range(meta, 0, 24))
    indptr = read_range(meta, 24, 24 + 8 * (a.n + 1))
    nnz_prefix = struct.unpack("<q", indptr[-8:])[0]
    idx_start = 24 + 8 * (nrow + 1)
    with (out / f"base.metadata.{tag}.spmat").open("wb") as fh:
        fh.write(struct.pack("<qqq", a.n, ncol, nnz_prefix))
        fh.write(indptr)
        copy_range(meta, idx_start, idx_start + 4 * nnz_prefix, fh)
        one = struct.pack("<f", 1.0)
        for start in range(0, nnz_prefix, CHUNK):
            fh.write(one * min(CHUNK, nnz_prefix - start))
    print(f"теги базы: {nnz_prefix} из {nnz}, словарь {ncol}", flush=True)

    for name in ("query.public.100K.u8bin", "query.metadata.public.100K.spmat"):
        fetch_whole(name, out / name)
    print("запросы загружены", flush=True)


if __name__ == "__main__":
    main()
