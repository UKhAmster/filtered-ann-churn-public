#!/usr/bin/env bash
# UNG smoke run on synthetic data.
# The chain follows the project README exactly, only on 5000 vectors instead of SIFT1M:
# label generation -> query label generation -> ground truth -> build -> search.
set -euo pipefail
cd /src

N=5000
NQ=100
D=32
K=10
DATA=/tmp/smoke
mkdir -p "$DATA" /tmp/index /tmp/results

python3 - <<PY
import struct
import numpy as np

rng = np.random.default_rng(0)

def write_bin(path, arr):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<II", arr.shape[0], arr.shape[1]))
        fh.write(arr.astype(np.float32).tobytes())

write_bin("$DATA/base.bin", rng.random(($N, $D)))
write_bin("$DATA/query.bin", rng.random(($NQ, $D)))
PY

./build/tools/generate_base_labels \
    --num_points "$N" --num_labels 12 --distribution_type zipf \
    --output_file "$DATA/base_labels.txt"

./build/tools/generate_query_labels \
    --num_points "$NQ" --distribution_type zipf --K "$K" --scenario containment \
    --input_file "$DATA/base_labels.txt" --output_file "$DATA/query_labels.txt"

./build/tools/compute_groundtruth \
    --data_type float --dist_fn L2 --scenario containment --K "$K" --num_threads 4 \
    --base_bin_file "$DATA/base.bin" --base_label_file "$DATA/base_labels.txt" \
    --query_bin_file "$DATA/query.bin" --query_label_file "$DATA/query_labels.txt" \
    --gt_file "$DATA/gt.bin"

./build/apps/build_UNG_index \
    --data_type float --dist_fn L2 --num_threads 4 --max_degree 32 --Lbuild 100 --alpha 1.2 \
    --base_bin_file "$DATA/base.bin" --base_label_file "$DATA/base_labels.txt" \
    --index_path_prefix /tmp/index/ --scenario general --num_cross_edges 6

./build/apps/search_UNG_index \
    --data_type float --dist_fn L2 --num_threads 4 --K "$K" \
    --base_bin_file "$DATA/base.bin" --base_label_file "$DATA/base_labels.txt" \
    --query_bin_file "$DATA/query.bin" --query_label_file "$DATA/query_labels.txt" \
    --gt_file "$DATA/gt.bin" --index_path_prefix /tmp/index/ \
    --result_path_prefix /tmp/results/ --scenario containment \
    --num_entry_points 16 --Lsearch 10 50 100

echo "--- search results ---"
cat /tmp/results/*.csv
echo "smoke ok"
