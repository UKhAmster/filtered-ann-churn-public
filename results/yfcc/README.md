# Measurements on YFCC-10M (real tags)

Data: a 1M-vector prefix of YFCC-10M from the filtered track of big-ann-benchmarks NeurIPS'23
(CLIP, 192 dimensions, uint8; 8.8 tags per object on average in the prefix), public queries.
Download: `scripts/fetch_yfcc.py`; preparation: `load_yfcc` in `scripts/pilot_driver.py`.
Work per query is the number of distance computations (machine-independent).

## Filters

Tags are chosen deterministically by their share in the run's collection, among tags carried by
at least 300 public queries. The choice therefore depends on the collection size:

| Filter | Main runs (400k) | Operating point selection (100k) |
|---|---|---|
| wide | tag 29, 19.3 % | tag 29, 19.1 % |
| narrow | tag 345, 1.09 % | tag 345, 1.16 % |
| tiny | tag 933, 0.39 % | tag 3808, 0.40 % |
| purge (deleted category) | tag 11, 7.0 % | tag 21, 7.0 % |

The queries of a filter are the first 300 public queries with that tag; unfiltered search uses
the first 300 queries. The label vocabulary is reduced to the four tags plus an "other" label:
this does not change answers for these filters, while dynamic Filtered-DiskANN needs at least as
many start points as labels. Labels of an object are sorted (see `docs/capability-map.md` on the
`insert_point` contract).

## Operating points (`operating_point/`)

Rule: recall@10 on a fresh 100k-object index is at least 0.98 without a filter and with the wide
filter; the smallest parameter satisfying the rule is taken.

| Method | Parameter | no filter | wide | narrow | tiny |
|---|---|---|---|---|---|
| hnswlib | ef = 64 | 0.979 | 0.993 | 0.996 | 0.998 |
| hnswlib | **ef = 128** | 0.990 | 0.996 | 0.995 | 0.997 |
| UNG | **L = 100** | 0.995 | 0.990 | 0.994 | 0.998 |
| Filtered-DiskANN | **L = 100** | 0.993 | 0.986 | 0.917 | 0.907 |
| Filtered-DiskANN | L = 200 | 0.995 | 0.995 | 0.936 | 0.908 |
| Filtered-DiskANN | L = 300 | 0.996 | 0.996 | 0.949 | 0.905 |

On a fresh index, Filtered-DiskANN recall on narrow tags is noticeably lower than that of the
other two methods and grows slowly with L.

## Results

`churn_3x/`: steady churn, 300k "delete, insert" operations (the collection is renewed three
times); `purge/`: deletion of a category (the purge tag) at once followed by 100k insertions.
Figure: `churn_3x/steady_churn.png` (`scripts/plot_yfcc.py`).

- **hnswlib, mark-delete:** recall 0.99-1.0 on all filters; work per query grows almost
  linearly, ×3.5 without a filter and ×3.6 / 4.0 / 3.7 with tags of 19 / 1.1 / 0.4 %.
  Slot reuse keeps work within -2…+11 %.
- **Filtered-DiskANN without consolidation:** the narrower the tag, the stronger the degradation:
  recall 0.987 → 0.939, 0.907 → 0.759, 0.861 → 0.758; share of queries below 0.9 2 → 16 %,
  24 → 57 %, 35 → 63 %. Without a filter 0.993 → 0.976. A rebuild restores the initial values.
- **Filtered-DiskANN with consolidation every 10k deletions:** the wide tag holds (0.979), the
  narrow one reaches 0.830, the tiniest drops to 0 during churn and recovers only to 0.56 by the end.
- **UNG, rebuild at 20 % deleted:** mean recall 0.893-0.900 against the model 1 − t/2 = 0.90;
  14 builds.
- **Category purge (7 %):** hnswlib is unaffected; Filtered-DiskANN without consolidation loses
  5-7 p.p. on narrow tags over 100k operations (0.913 → 0.865, 0.864 → 0.793), 1 p.p. with
  consolidation.

## Selectivity sweep (`selectivity/`)

Six tags with target shares of 30, 10, 3, 1, 0.3 and 0.1 % (actual 33.9 / 9.7 / 2.9 / 1.0 / 0.28 /
0.20 %; no tag with at least 300 queries was found for 0.1 %), steady churn of 200k operations,
methods at their operating points. Derived metrics: `selectivity/derived.csv`
(`scripts/derived_table.py`); figure: `selectivity/selectivity.png` (`scripts/plot_selectivity.py`).

- **hnswlib (mark-delete):** recall 1.0 at all levels; mean excess work 83-100 %, almost
  independent of selectivity. Law W = W₀ · p^(−β), p the live share of vertices: β = 0.89-1.01
  with about 1 % error (`scripts/work_law.py`); on SIFT, by latency, β = 0.87-0.94.
- **UNG (rebuild at 20 %):** mean recall deficit 9.4-10.4 p.p. at all levels; the model
  1 − t/2 does not depend on selectivity.
- **Filtered-DiskANN:** fresh-index recall falls as the filter narrows (0.99 → 0.94 → 0.87 →
  0.70 at 34 → 10 → 3 → 1 %), the mean deficit grows (without consolidation 0.3 → 0.9 → 2.2 →
  6.2 → 10.2 p.p.), and the time to a 2 p.p. loss shrinks (200 → 100 → 40k operations). On tags of
  0.28 and 0.20 % the method fails already on a fresh index (0.22-0.47 and about 0.01).

Selectivity does not explain everything: tags of 0.39 and 0.40 % in the main runs gave 0.86-0.91,
while 0.28 and 0.20 % gave almost zero. Tag cohesion with the geometry (lift of the tag's share
among the 10 nearest neighbors, `scripts/tag_cohesion.py`) does not explain the difference either:
the failing tags have the highest cohesion (11-13 against 3-5). A second difficulty factor is
needed; candidates: the relation of the tag to the queries, the absolute number of objects with
the label, the position of the label's start point.

| Tag | Share, % | Cohesion |
|---|---|---|
| 23 | 33.9 | 1.2 |
| 17 | 9.7 | 1.3 |
| 29 | 19.1 | 1.3 |
| 38 | 2.9 | 2.0 |
| 0 | 1.0 | 3.0 |
| 345 | 1.16 | 3.0 |
| 933 | 0.39 | 5.4 |
| 3808 | 0.40 | 4.9 |
| 337 | 0.20 | 11.0 |
| 142 | 0.27 | 13.0 |

## Repeats of three-fold churn (`churn_3x_seed8`, `churn_3x_seed9`)

Three-fold churn was repeated with scenario seeds 8 and 9 at the same operating points; the
summary over three seeds is in `seeds_summary.txt` (`scripts/aggregate_seeds.py`).

- hnswlib, mark-delete: work at the end ×3.47-3.48 without a filter, ×3.61-3.64 / ×4.03-4.10 /
  ×3.73-3.95 with tags of 19 / 1.1 / 0.4 %; β = 0.89 / 0.92-0.93 / 1.01-1.02 / 0.96-0.98, error
  0.5-1.9 %.
- Filtered-DiskANN without consolidation: recall at the end 0.939-0.942 / 0.731-0.759 /
  0.712-0.758, share of queries below 0.9 14-16 / 57-62 / 63-67 %.
- Filtered-DiskANN with consolidation: wide tag 0.977-0.979, narrow 0.830-0.857; the tiniest tag
  collapses to 0-0.02 in two seeds of three and stays at 0.65 or above in the third, so
  consolidation makes the behavior of narrow filters unpredictable.
- UNG: mean recall 0.891-0.903 on all filters and seeds.

### Repeats of the selectivity sweep (`selectivity_seed8`, `selectivity_seed9`)

The sweep was repeated with seeds 8 and 9 (the same tags: the choice depends only on shares). The
summary over three seeds is `selectivity/derived_seeds.csv` (`scripts/aggregate_derived.py`); the
figure `selectivity/selectivity.png` shows the mean with a min-max range.

- hnswlib: deficit 0.01-0.06 p.p.; excess work 83-100 %, growing as the filter narrows
  (83 % without a filter, 97-98 % on tags of 1 % and narrower).
- UNG: deficit 9.4-10.6 p.p. at all levels.
- Filtered-DiskANN without consolidation, means over seeds: fresh-index recall 0.99 / 0.94 / 0.88 /
  0.69 at 34 / 10 / 3 / 1 %; deficit 0.4 without a filter and 1.3 / 2.5 / 6.3 / 9.7 p.p.; time to a
  2 p.p. loss 160-200 / 80-100 / 40-60 / 40-60k operations. On tags of 0.28 and 0.20 %, fresh-index
  recall is 0.30-0.47 and 0.01-0.15.

## Columns of `derived.csv` and `derived_seeds.csv`

Column names are in Russian because the same files feed the thesis. Translation:

| Column | Meaning |
|---|---|
| метод | method |
| режим | maintenance mode |
| фильтр | filter |
| доля | share of objects passing the filter |
| оракул | oracle: fresh-index recall@10 |
| полнота в конце | recall@10 at the end of the run |
| недодан, п.п. | mean recall deficit from the oracle, percentage points |
| наработка до нарушения, тыс. | operations until recall drops by the threshold, thousands |
| избыток работы, % | mean excess work relative to the fresh index, % |
| доля < 0,9 в среднем | mean share of queries with recall below 0.9 |
| обслуживание за прогон, с | total maintenance time per run, s |
| полная перестройка, с | time of a full rebuild, s |
| повторов | number of seeds |

In `derived_seeds.csv` the suffixes `mean`, `min`, `max` give the mean and range over seeds.
