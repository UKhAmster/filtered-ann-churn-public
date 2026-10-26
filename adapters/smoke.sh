#!/usr/bin/env bash
# adapters/smoke.sh — собирает образ метода и прогоняет его собственный пример.
# Успех = ненулевой вывод примера и код возврата 0.
set -u
method="$1"
dir="$(dirname "$0")/$method"

if ! docker build -t "fanndyn-${method}:smoke" "$dir" >"/tmp/build-${method}.log" 2>&1; then
    echo "BUILD_FAIL ${method} (лог: /tmp/build-${method}.log)"
    exit 1
fi
if ! timeout 900 docker run --rm "fanndyn-${method}:smoke" >"/tmp/run-${method}.log" 2>&1; then
    echo "RUN_FAIL ${method} (лог: /tmp/run-${method}.log)"
    exit 2
fi
echo "OK ${method}"
