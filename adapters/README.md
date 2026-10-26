# Search method adapters

Each candidate method is attached as a separate Docker image and is not embedded in the
`fanndyn` package. The reason is isolation: the methods are written in C++ with incompatible
dependency sets, and some of them are frozen and need old compiler versions. Building them in
a shared environment would mean tailoring the environment to the most demanding one.

## Layout

```
adapters/
  smoke.sh              builds an image and runs the method's example
  hnswlib/              reference baseline
  acorn/                ACORN: predicate filtering during traversal
  filtered_diskann/     Filtered-DiskANN: labels built into the graph at construction
  ung/                  UNG: separate graphs for label sets
```

Smoke check of one method:

```bash
bash adapters/smoke.sh <method>     # OK / BUILD_FAIL / RUN_FAIL
```

Logs are kept in `/tmp/build-<method>.log` and `/tmp/run-<method>.log`.

## Rules

**A revision is pinned by SHA, not by branch.** A branch moves on, and results silently stop
being reproducible. The SHA of each method is set in `ARG COMMIT_SHA` of its Dockerfile and
repeated in `docs/capability-map.md`.

**The example is taken from the project, not written anew**, except when the project's own
example needs an external dataset. Such cases are explained in a Dockerfile comment.

**A method's capabilities are checked in the code.** READMEs lag behind implementations: the
ACORN README, for example, shows a four-argument index constructor, while the header file
requires five.

## Protocol drivers

To run scenarios, hnswlib, Filtered-DiskANN and UNG have C++ drivers
(`<method>/fanndyn_driver.cpp`) with shared input/output in `common/fanndyn_io.h`: vectors,
labels in CSR, an operation stream, per-filter queries; the output is results, latency and the
number of distance computations per query. Driver images are built from the `adapters/`
directory so that the shared header is in the build context:

```bash
cd adapters
docker build -f hnswlib/Dockerfile.driver -t fanndyn-hnswlib:driver .
docker build -f filtered_diskann/Dockerfile.driver -t fanndyn-filtered_diskann:driver .
docker build -f ung/Dockerfile.driver -t fanndyn-ung:driver .
```

The DiskANN and UNG images are built on top of `fanndyn-<method>:smoke`. The ACORN driver still
uses the older format (one label per object) and is built from its own directory:
`docker build -f acorn/Dockerfile.driver -t fanndyn-acorn:driver acorn`.
