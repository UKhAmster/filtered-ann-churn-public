"""Пилотный замер методов через драйверы на C++ в Docker (hnswlib, Filtered-DiskANN, ACORN, UNG).

Формат обмена с драйверами — adapters/common/fanndyn_io.h. Два источника данных:
  sift — SIFT с синтетическими (zipf) или кластерными (kmeans) метками, одна метка на объект,
         одни и те же запросы для всех фильтров;
  yfcc — YFCC-10M с настоящими тегами (префикс, см. scripts/fetch_yfcc.py): у фильтра по тегу
         свои запросы — публичные запросы, среди тегов которых есть этот тег.

Пример:
    .venv/bin/python scripts/pilot_driver.py --method diskann --scenario steady_churn --mode lazy
    .venv/bin/python scripts/pilot_driver.py --dataset yfcc --method hnswlib --scenario steady_churn --mode mark
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fanndyn.data import LABEL_KINDS, make_labels, read_fvecs, read_spmat, read_u8bin
from fanndyn.eval.metrics import filtered_ground_truth
from fanndyn.eval.runner import summarize
from fanndyn.runbook.export import alive_at_checkpoints, encode_ops, with_base_checkpoint
from fanndyn.runbook.scenarios import build_scenario
from fanndyn.runbook.schema import Op, OpKind, Runbook
from fanndyn.yfcc import label_mask, queries_with_tag, reduce_tags

IMAGES = {
    "hnswlib": "fanndyn-hnswlib:driver",
    "diskann": "fanndyn-filtered_diskann:driver",
    "acorn": "fanndyn-acorn:driver",
    "ung": "fanndyn-ung:driver",
}
MODES = {
    "hnswlib": ("mark", "replace"),
    "diskann": ("lazy", "consolidate"),
    "acorn": ("tombstone", "rebuild"),
    "ung": ("rebuild",),
}
SIFT_N_LABELS = 20
LABEL_SEED = 11
# Целевые доли тегов YFCC в коллекции: широкий, узкий, очень узкий, удаляемая категория.
YFCC_TARGETS = {"wide": 0.19, "narrow": 0.012, "tiny": 0.004, "purge": 0.07}
MISSING = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class Filter:
    name: str
    label: int  # -1 — без фильтра
    queries: np.ndarray


@dataclass(frozen=True)
class Workload:
    vectors: np.ndarray
    label_ptr: np.ndarray
    label_idx: np.ndarray
    filters: tuple[Filter, ...]
    purge_label: int
    meta: dict

    def label_sets(self) -> list[tuple[int, ...]]:
        p, i = self.label_ptr, self.label_idx
        return [tuple(int(x) for x in i[p[r] : p[r + 1]]) for r in range(len(p) - 1)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--dataset", choices=["sift", "yfcc"], default="sift")
    p.add_argument("--labels", choices=LABEL_KINDS, default="zipf", help="только для sift")
    p.add_argument("--method", choices=sorted(IMAGES), default="diskann")
    p.add_argument("--mode", required=True)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--work", type=Path, default=Path.home() / "data/fanndyn_work")
    p.add_argument("--n-base", type=int, default=100_000)
    p.add_argument("--n-ops", type=int, default=100_000)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--consolidate-every", type=int, default=10_000)
    p.add_argument("--n-queries", type=int, default=300)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--R", type=int, default=32)
    p.add_argument("--L-build", type=int, default=100)
    p.add_argument("--L-search", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--hnsw-M", type=int, default=16)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--acorn-M", type=int, default=32)
    p.add_argument("--acorn-gamma", type=int, default=12)
    p.add_argument("--acorn-M-beta", type=int, default=64)
    p.add_argument("--ef-search", type=int, default=64)
    p.add_argument("--rebuild-at", type=float, default=0.2)
    p.add_argument("--ung-entry-points", type=int, default=16)
    p.add_argument("--ung-fetch", type=int, default=50)
    p.add_argument("--probe-list", action="store_true", help="DiskANN: живые вершины в списке L")
    p.add_argument(
        "--yfcc-targets",
        type=str,
        default="",
        help="доли тегов через запятую для перебора селективности, например 0.3,0.1,0.01",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("results/pilot_driver"))
    a = p.parse_args()
    if a.mode not in MODES[a.method]:
        p.error(f"для {a.method} режим должен быть одним из {MODES[a.method]}")
    if a.data is None:
        a.data = Path.home() / ("data/sift" if a.dataset == "sift" else "data/yfcc")
    return a


def load_sift(a: argparse.Namespace, n_total: int) -> Workload:
    vectors = read_fvecs(a.data / "sift_base.fvecs", count=n_total)
    queries = read_fvecs(a.data / "sift_query.fvecs", count=a.n_queries)
    labels = make_labels(a.labels, vectors, SIFT_N_LABELS, seed=LABEL_SEED)
    ptr = np.arange(n_total + 1, dtype=np.uint64)
    filters = (
        Filter("none", -1, queries),
        Filter("wide", 0, queries),
        Filter("narrow", SIFT_N_LABELS - 1, queries),
    )
    return Workload(vectors, ptr, labels.astype(np.uint32), filters, 1, {"labels": a.labels})


def pick_tag(share: np.ndarray, target: float, allowed: np.ndarray, taken: set[int]) -> int:
    """Тег с долей, ближайшей к целевой, среди допустимых и ещё не выбранных."""
    order = np.argsort(np.abs(share - target))
    return int(next(t for t in order if allowed[t] and int(t) not in taken))


def load_yfcc(a: argparse.Namespace, n_total: int) -> Workload:
    base_file = next(a.data.glob("base.*.u8bin"))
    meta_file = next(a.data.glob("base.metadata.*.spmat"))
    vectors = read_u8bin(base_file, count=n_total).astype(np.float32)
    ptr, idx = read_spmat(meta_file)
    ptr, idx = ptr[: n_total + 1], idx[: int(ptr[n_total])]
    queries_all = read_u8bin(a.data / "query.public.100K.u8bin").astype(np.float32)
    _qptr, qidx = read_spmat(a.data / "query.metadata.public.100K.spmat")
    ncol = int(max(idx.max(), qidx.max())) + 1
    share = np.bincount(idx, minlength=ncol) / n_total
    enough_queries = np.bincount(qidx, minlength=ncol) >= a.n_queries
    targets = dict(YFCC_TARGETS)
    if a.yfcc_targets:
        shares = [float(x) for x in a.yfcc_targets.split(",")]
        targets = {f"s{v:g}": v for v in shares} | {"purge": YFCC_TARGETS["purge"]}
    chosen: dict[str, int] = {}
    for name, target in targets.items():
        chosen[name] = pick_tag(share, target, enough_queries, set(chosen.values()))
    keep = list(chosen.values())
    new_ptr, new_idx = reduce_tags(ptr, idx, keep)
    qmeta = a.data / "query.metadata.public.100K.spmat"
    filters = [Filter("none", -1, queries_all[: a.n_queries])]
    for pos, name in enumerate(chosen):
        if name == "purge":
            continue
        rows = queries_with_tag(qmeta, chosen[name], a.n_queries)
        filters.append(Filter(name, pos, queries_all[rows]))
    meta = {
        "tags": chosen,
        "tag_share": {k: float(share[v]) for k, v in chosen.items()},
        "reduced_labels": {**{k: i for i, k in enumerate(chosen)}, "other": len(keep)},
    }
    return Workload(vectors, new_ptr, new_idx, tuple(filters), keep.index(chosen["purge"]), meta)


def write_bin(path: Path, x: np.ndarray) -> None:
    with path.open("wb") as fh:
        np.array(x.shape, dtype=np.uint32).tofile(fh)
        np.ascontiguousarray(x, dtype=np.float32).tofile(fh)


def driver_args(a: argparse.Namespace) -> list[str]:
    if a.method == "hnswlib":
        return ["--M", str(a.hnsw_M), "--ef-construction", str(a.hnsw_ef_construction),
                "--ef-search", str(a.ef_search), "--seed", str(a.seed)]  # fmt: skip
    if a.method == "diskann":
        return ["--R", str(a.R), "--Lbuild", str(a.L_build), "--Lsearch", str(a.L_search),
                "--consolidate-every", str(a.consolidate_every),
                "--probe-list", "1" if a.probe_list else "0"]  # fmt: skip
    if a.method == "ung":
        return ["--R", str(a.R), "--Lbuild", str(a.L_build), "--Lsearch", str(a.L_search),
                "--entry-points", str(a.ung_entry_points), "--fetch", str(a.ung_fetch),
                "--rebuild-at", str(a.rebuild_at)]  # fmt: skip
    return ["--M", str(a.acorn_M), "--gamma", str(a.acorn_gamma), "--M-beta", str(a.acorn_M_beta),
            "--ef-search", str(a.ef_search), "--rebuild-at", str(a.rebuild_at)]  # fmt: skip


def run_driver(a: argparse.Namespace, work: Path, rb: Runbook, wl: Workload) -> float:
    """Запускает драйвер на runbook; возвращает время работы в секундах."""
    encode_ops(rb).tofile(work / "ops.u32")
    for pattern in ("res_c*", "lat_c*", "cmps_c*", "alive_c*"):
        for old in work.glob(pattern):
            old.unlink()
    common = ["--dir", "/work", "--mode", a.mode, "--K", str(a.k), "--threads", str(a.threads),
              "--filters", ",".join(str(f.label) for f in wl.filters)]  # fmt: skip
    cmd = ["docker", "run", "--rm", "--cpus", str(a.threads), "-v", f"{work}:/work",
           IMAGES[a.method], *common, *driver_args(a)]  # fmt: skip
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def read_stats(work: Path) -> dict[int, dict]:
    stats = {}
    path = work / "stats.tsv"
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            ck, pending, cons, cons_s, ins_s = line.split("\t")
            stats[int(ck)] = {
                "n_lazy_pending": int(pending),
                "consolidations": int(cons),
                "consolidate_s_total": float(cons_s),
                "insert_s_total": float(ins_s),
            }
    return stats


def optional_mean(path: Path, dtype, key: str) -> dict:
    if not path.exists():
        return {}
    x = np.fromfile(path, dtype=dtype)
    return {f"{key}_mean": float(x.mean()), f"{key}_p95": float(np.percentile(x, 95))}


def collect(a, work: Path, rb: Runbook, wl: Workload, masks: list, extra: dict) -> list[dict]:
    rows = []
    stats = read_stats(work)
    for ck, (tag, alive, steps) in enumerate(alive_at_checkpoints(rb)):
        for f_i, flt in enumerate(wl.filters):
            allowed = alive if flt.label < 0 else alive[masks[f_i][alive]]
            truth = filtered_ground_truth(flt.queries, wl.vectors, allowed, a.k)
            res = np.fromfile(work / f"res_c{ck}_f{f_i}.u32", dtype=np.uint32).reshape(-1, a.k)
            lat = np.fromfile(work / f"lat_c{ck}_f{f_i}.f32", dtype=np.float32)
            found = [r[r != MISSING].astype(np.int64) for r in res]
            rows.append({
                "runbook": rb.name,
                "checkpoint": tag,
                "step": steps,
                "n_alive": len(alive),
                "filter": flt.name,
                **summarize(found, truth, lat, len(allowed), len(alive)),
                **optional_mean(work / f"cmps_c{ck}_f{f_i}.u32", np.uint32, "cmps"),
                **optional_mean(work / f"alive_c{ck}_f{f_i}.u32", np.uint32, "list_alive"),
                **stats.get(ck, {}),
                **extra,
            })  # fmt: skip
            r = rows[-1]
            print(
                f"{tag:>7} {flt.name:>6} step={steps:>6} recall={r['recall_mean']:.4f} "
                f"p5={r['recall_p5']:.3f} lat={r['latency_ms_mean']:.2f}ms "
                f"cmps={r.get('cmps_mean', float('nan')):.0f}",
                flush=True,
            )
    return rows


def main() -> None:
    a = parse_args()
    n_total = a.n_base + a.n_ops
    wl = load_sift(a, n_total) if a.dataset == "sift" else load_yfcc(a, n_total)
    rb = with_base_checkpoint(
        build_scenario(
            a.scenario,
            n_base=a.n_base,
            n_ops=a.n_ops,
            label_sets=wl.label_sets(),
            seed=a.seed,
            checkpoint_every=a.checkpoint_every,
            purge_label=wl.purge_label,
        )
    )
    tag = a.labels if a.dataset == "sift" else "yfcc"
    work = a.work / f"{a.method}-{a.scenario}-{a.mode}-{tag}-s{a.seed}"
    work.mkdir(parents=True, exist_ok=True)
    write_bin(work / "base.bin", wl.vectors)
    wl.label_ptr.astype(np.uint64).tofile(work / "labels_ptr.u64")
    wl.label_idx.astype(np.uint32).tofile(work / "labels_idx.u32")
    if np.all(np.diff(wl.label_ptr) == 1):  # драйвер ACORN принимает одну метку на объект
        wl.label_idx.astype(np.uint32).tofile(work / "labels.u32")
    for f_i, flt in enumerate(wl.filters):
        write_bin(work / f"query_f{f_i}.bin", flt.queries)
    write_bin(work / "query.bin", wl.filters[0].queries)
    masks = [
        None if f.label < 0 else label_mask(wl.label_ptr, wl.label_idx, f.label) for f in wl.filters
    ]

    wall = run_driver(a, work, rb, wl)
    rows = collect(a, work, rb, wl, masks, {"driver_wall_s": wall})

    # Опорная точка: индекс, построенный заново на итоговой живой коллекции.
    final_alive = alive_at_checkpoints(rb)[-1][1]
    fresh = Runbook(
        name=rb.name + "-rebuild",
        seed=rb.seed,
        generator_version=rb.generator_version,
        ops=(
            Op(OpKind.INSERT, ids=tuple(final_alive.tolist())),
            Op(OpKind.CHECKPOINT, tag="rebuild"),
        ),
    )
    (work / "stats.tsv").unlink(missing_ok=True)
    rebuild_s = run_driver(a, work, fresh, wl)
    reb = collect(a, work, fresh, wl, masks, {"driver_wall_s": rebuild_s})
    for r in reb:
        r["step"] = rows[-1]["step"]
    rows += reb

    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.scenario}-{a.mode}"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (a.out / f"{stem}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    meta = {k: str(v) for k, v in vars(a).items()}
    meta.update(
        generator_version=rb.generator_version,
        label_seed=LABEL_SEED,
        image=IMAGES[a.method],
        **wl.meta,
    )
    (a.out / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
