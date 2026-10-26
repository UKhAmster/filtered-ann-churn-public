# ACORN is a FAISS fork; the commit is pinned by SHA: the repository has been frozen since
# March 2025, and the branch may still move.
FROM ubuntu:22.04
ARG REPO_URL=https://github.com/guestrin-lab/ACORN.git
ARG COMMIT_SHA=c259f11cdbec880671658eaa0d8747e2cc9de79d
ENV DEBIAN_FRONTEND=noninteractive
# FAISS needs zlib to serialize indexes; without it cmake fails on find_package(ZLIB).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake libomp-dev libopenblas-dev zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "$REPO_URL" /src && cd /src && git checkout "$COMMIT_SHA"
WORKDIR /src
# Two deviations from the project README, both forced.
#
# BUILD_TESTING=OFF: FAISS tests are unrelated to ACORN, and demos (where
# test_acorn lives) are included regardless of this flag.
#
# CMAKE_SKIP_INSTALL_RULES=ON: the root CMakeLists of ACORN pulls nlohmann_json via
# FetchContent and adds its path to the faiss target as PUBLIC, without a
# $<BUILD_INTERFACE:> wrapper. When generating the export set CMake rejects such a path
# ("prefixed in the build directory") and aborts configuration. We do not need to
# install the library system-wide, so disabling install rules removes the problem
# without touching the method's sources.
RUN cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
          -DCMAKE_SKIP_INSTALL_RULES=ON \
          -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
RUN make -C build -j"$(nproc)" faiss
# The project's own test: built to check compilation but not run,
# since it needs an external SIFT dataset with prepared metadata and ground truth.
RUN make -C build -j"$(nproc)" test_acorn
COPY smoke_acorn.cpp /smoke/smoke_acorn.cpp
RUN g++ -O2 -std=c++17 -fopenmp -I/src /smoke/smoke_acorn.cpp \
        -L/src/build/faiss -lfaiss -o /smoke/smoke_acorn
ENV LD_LIBRARY_PATH=/src/build/faiss
CMD ["/smoke/smoke_acorn"]
