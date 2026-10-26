<!-- docs/capability-map.md -->
# Operational capability map of the methods

Compiled from reading the source code, 2026-08-26.
"Emulation" means the operation is implemented by a wrapper on top of the method.

Public header files and application interfaces were checked, not READMEs:
the ACORN README, for example, shows a four-argument index constructor,
while `faiss/IndexACORN.h` requires five, with metadata passed at construction.

| Method | Commit | Build | Insert | Delete | Attribute update | Predicate type | Notes |
|---|---|---|---|---|---|---|---|
| hnswlib | `d9b3608c83d8` (PyPI 0.8.0) | OK | native, `addPoint` with `replace_deleted`, capacity via `resizeIndex` | native (marking), `markDelete` / `unmarkDelete`, no physical deletion | emulation | external filtering: `BaseFilterFunctor::operator()(id)` during search | baseline |
| ACORN | `c259f11cdbec` | OK | native by interface, `add`; metadata of new points must be put into the external vector before the call. **Does not work in practice**: a repeated `add` crashes (see below) | none: `remove_ids` is not overridden, only a full `reset()` | emulation | predicate-agnostic: a boolean mask `filter_ids_map` of size `nq × N` per query | FAISS fork |
| Filtered-DiskANN | `78256bbab468` (branch `cpp_main`) | OK | native with labels: `insert_point(point, tag, labels)` | native with consolidation: `lazy_delete` + `consolidate_deletes` | emulation (delete and reinsert) | one label per query: `search_with_filters(query, filter_label, ...)`; a universal label passes any filter | the only candidate with native dynamics; ships `test_insert_deletes_consolidate` and `test_streaming_scenario` |
| UNG | `92b58ffd600d` | OK | none | none | none | label sets: containment (⊆), equality (=), overlap (∩≠∅), no-filter | fully static: `uni_nav_graph.h` has only `build`, `search`, `save`, `load` |

## Implications for the protocol

**Filtered-DiskANN is the only method with native dynamics.** For the others, dynamics are
emulated by a wrapper, and this is not a flaw of the selection but its point: the work measures
how methods with different designs survive data changes, and "rebuild the whole index" is as
much a response to change as a point insertion, only an expensive one. The cost of a rebuild
enters the measured quantities through the maintenance cost metric.

**Attribute updates are not supported directly anywhere.** In hnswlib and ACORN attributes
live outside the index, in Filtered-DiskANN labels are built into the structure and change only
by recreating the point, in UNG group membership defines the whole graph. The `attr_update`
scenario is therefore implemented by a wrapper everywhere, and the methods will differ in the
cost of that emulation.

**Predicate types are not directly comparable.** ACORN accepts an arbitrary boolean mask,
Filtered-DiskANN one label per query, UNG operations on label sets, hnswlib an external
function. The common subset on which the methods compare fairly is a single-label condition.
More expressive predicates are measured only on the methods that support them, and this is
stated in each results table.

**Insertion in ACORN requires external metadata management.** The constructor takes
`std::vector<int>& metadata` by reference, so the attribute vector lives outside the index and
must be extended before calling `add`. The adapter has to maintain this correspondence itself.

## Build check

**All four methods out of four built and ran.**

Smoke runs on 2026-08-26:

| Method | Result | What the run showed |
|---|---|---|
| hnswlib | `OK` | `smoke ok (5, 10)` |
| ACORN | `OK` | `smoke ok nq=5 k=10 violations=0`: no result violated the predicate |
| Filtered-DiskANN | `OK` | building a filtered index and searching with `--filter_label 1` on 5000 points |
| UNG | `OK` | the project's full chain: labels, ground truth, build, search |

No method built following its authors' instructions without changes. All changes are in the
build environment; the methods' sources were not modified:

- **hnswlib**: the base image `python:3.11-slim` has no compiler, and the PyPI package is
  distributed as source. `g++` was added.
- **ACORN**: `zlib1g-dev` was missing; `BUILD_TESTING=ON` from the README aborts generation on
  CMake 3.22; the root `CMakeLists.txt` adds the `nlohmann_json` path to the `faiss` target as
  PUBLIC without a `$<BUILD_INTERFACE:>` wrapper, so CMake rejects the export set.
  Resolved with `-DCMAKE_SKIP_INSTALL_RULES=ON`: the library need not be installed system-wide.
- **Filtered-DiskANN**: `libmkl-full-dev` asks an interactive question and hangs the build;
  CMakeLists requires Intel OpenMP and tries three fixed paths, none of which is where
  `libomp-14-dev` puts `libiomp5.so`. `-DOMP_PATH` is set explicitly.
- **UNG**: MKL, listed among the dependencies, is not needed by the project itself;
  `codes/CMakeLists.txt` requires only Boost and OpenMP.

### Note on Filtered-DiskANN

The `main` branch of `microsoft/DiskANN` was rewritten in Rust, and its history contains no
C++ code at all: a query for commits older than 2025-01-01 returns nothing. The implementation
matching the Filtered-DiskANN paper (WWW 2023) and comparable with published results lives in
the `cpp_main` branch, last commit May 2026. This work uses it. The Rust version is different
code with different characteristics; it may become a separate object of measurement, but not a
replacement for the C++ implementation.

### Note on build reproducibility

UNG's `codes/CMakeLists.txt` builds Release with `-march=native -mtune=native`. The binary is
tied to the testbed processor (AMD EPYC 9754). When moving measurements to other hardware the
image must be rebuilt, not copied.

## Dynamics in practice (2026-09-29)

Reading the code tells what a method can do; a run shows whether it works. Attaching the
methods to the scenario runner revealed the following.

**ACORN: repeated insertion crashes.** With d = 128 the third call to `IndexACORNFlat::add`
(batches of 10,000 + 2,000 + 2,000) ends in a segfault at any `OMP_NUM_THREADS`; with d = 32
and smaller batches there is no crash. A minimal example without the protocol wrapper is
`adapters/acorn/repro_repeated_add.cpp`. Besides, `add` allocates an array of `ntotal` locks on
every call, so a single insertion costs O(N) and is unusable for an insertion stream even
without the crash. The driver `adapters/acorn/fanndyn_driver.cpp` buffers insertions until a
checkpoint, but that does not avoid the crash. Conclusion for the protocol: ACORN is effectively
a build-once method, and its only response to data changes is a full rebuild.

**Filtered-DiskANN: filtered search in a dynamic index needs a workaround.**
`search_with_tags` with a filter looks up the label in the string dictionary `_label_map`, which
a dynamic index does not fill, and the public `search_with_filters` returns internal slot
numbers and does not drop lazily deleted points. The driver
`adapters/filtered_diskann/fanndyn_driver.cpp` fills the dictionary through explicit template
instantiation; the DiskANN sources are not modified. A narrow filter (1.4 % of the collection)
is not served by the dynamic index even when fresh: the label's start point has almost no
neighbors with the same label (cf. microsoft/DiskANN#550).

**UNG: attached as a static index rebuilt at a threshold.** Driver
`adapters/ung/fanndyn_driver.cpp`: between rebuilds the index serves the previous collection,
deleted objects are filtered out of answers, new ones are invisible; when the share of deleted
objects in the index reaches the threshold relative to the live count, the index is rebuilt.
Labels are shifted by +1, since label 0 is taken by the trie root. Search is batched, so latency
is the batch mean.

**Filtered-DiskANN: an object's labels must be sorted on insertion.**
`detect_common_filters` intersects the labels of a vertex and of the filter with two pointers and
requires both lists to be sorted; `insert_point(point, tag, labels)` does not check this. With
unsorted labels an object is not found by any label but its first: on YFCC, before the fix,
97-99 % of filtered search misses fell on such objects, and recall with a tag covering 19 % of
the collection was stuck at 0.93 for any parameters. The driver sorts labels before insertion;
for single-label objects (SIFT) the bug did not show.
