# Filtered-DiskANN.
#
# Важное уточнение: ветка main репозитория microsoft/DiskANN переписана на Rust,
# и её история не содержит C++-кода вовсе. Реализация, соответствующая статье
# Filtered-DiskANN (SIGMOD 2023) и сопоставимая с опубликованными результатами,
# живёт в ветке cpp_main. Коммит закреплён по SHA.
FROM ubuntu:22.04
ARG REPO_URL=https://github.com/microsoft/DiskANN.git
ARG COMMIT_SHA=78256bbab4685e1774e78d331e081a153be26823
ENV DEBIAN_FRONTEND=noninteractive
# libmkl-full-dev спрашивает про замену BLAS/LAPACK по умолчанию; без
# noninteractive сборка виснет на этом вопросе.
# CMakeLists требует Intel OpenMP и перебирает три фиксированных пути. В jammy
# пакета libiomp-dev нет, а libiomp5.so приходит с libomp-14-dev и лежит вне
# этих путей — отсюда явный -DOMP_PATH ниже.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git make cmake g++ ca-certificates \
        libaio-dev libgoogle-perftools-dev libboost-all-dev libmkl-full-dev \
        libomp-14-dev \
        python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "$REPO_URL" /src && cd /src && git checkout "$COMMIT_SHA" \
    && git submodule update --init --recursive
WORKDIR /src
RUN mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DOMP_PATH=/usr/lib/llvm-14/lib .. \
    && make -j"$(nproc)"
COPY smoke.sh /smoke.sh
CMD ["bash", "/smoke.sh"]
