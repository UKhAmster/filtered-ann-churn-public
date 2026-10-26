#!/usr/bin/env bash
# Дымовой прогон Filtered-DiskANN на синтетических данных.
# Цепочка собрана из собственных утилит проекта: генерация векторов и меток,
# построение фильтрованного индекса, поиск с предикатом по метке.
set -euo pipefail
cd /src

N=5000
NQ=100
D=32
K=10
DATA=/tmp/smoke
mkdir -p "$DATA"

./build/apps/utils/rand_data_gen --data_type float --output_file "$DATA/base.bin" -D "$D" -N "$N"
./build/apps/utils/rand_data_gen --data_type float --output_file "$DATA/query.bin" -D "$D" -N "$NQ"

./build/apps/utils/generate_synthetic_labels \
    --num_labels 12 --num_points "$N" --output_file "$DATA/labels.txt"

# FilteredLbuild включает фильтрованное построение: без него метки игнорируются.
./build/apps/build_memory_index \
    --data_type float --dist_fn l2 \
    --data_path "$DATA/base.bin" --index_path_prefix "$DATA/index" \
    --label_file "$DATA/labels.txt" --FilteredLbuild 90 \
    -R 32 -L 100 --alpha 1.2 -T 4

./build/apps/search_memory_index \
    --data_type float --dist_fn l2 \
    --index_path_prefix "$DATA/index" --query_file "$DATA/query.bin" \
    --result_path "$DATA/result" --recall_at "$K" -L 20 40 -T 4 \
    --filter_label 1

echo "smoke ok"
