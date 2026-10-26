# Filtered-DiskANN.
#
# Note: the main branch of microsoft/DiskANN was rewritten in Rust,
# and its history contains no C++ code at all. The implementation matching the
# Filtered-DiskANN paper (WWW 2023) and comparable with published results
# lives in the cpp_main branch. The commit is pinned by SHA.
FROM ubuntu:22.04
ARG REPO_URL=https://github.com/microsoft/DiskANN.git
ARG COMMIT_SHA=78256bbab4685e1774e78d331e081a153be26823
ENV DEBIAN_FRONTEND=noninteractive
# libmkl-full-dev asks about replacing the default BLAS/LAPACK; without
# noninteractive the build hangs on this question.
# CMakeLists requires Intel OpenMP and tries three fixed paths. In jammy
# there is no libiomp-dev package, and libiomp5.so comes with libomp-14-dev outside
# those paths, hence the explicit -DOMP_PATH below.
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
