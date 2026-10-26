// Minimal example: repeated calls to IndexACORNFlat::add crash (found 2026-09-29).
//
// On ACORN c259f11 with d = 128 the third add call (10000 + 2000 + 2000 points) ends
// in a segfault at any OMP_NUM_THREADS; with d = 32 and smaller batches it does not. Build:
//   docker run --rm -v $PWD:/r --entrypoint bash fanndyn-acorn:smoke -c \
//     "g++ -O1 -g -std=c++17 -fopenmp -I/src /r/repro_repeated_add.cpp \
//      -L/src/build/faiss -lfaiss -o /r/repro && LD_LIBRARY_PATH=/src/build/faiss /r/repro"

#include <faiss/IndexACORN.h>
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>

int main(int argc, char** argv) {
    const int d = 128, N = 20000, M = 32, gamma = 12, M_beta = 64, k = 10;
    const int batches[] = {10000, 2000, 2000, 2000, 2000};
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<float> xb(size_t(N) * d);
    for (auto& v : xb) v = u(rng);
    std::vector<int> metadata(N);
    for (int i = 0; i < N; ++i) metadata[i] = i % 4;
    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);
    size_t off = 0;
    for (int b : batches) {
        std::printf("add %d on top of %zu\n", b, off); std::fflush(stdout);
        index.add(b, xb.data() + off * d);
        off += b;
        std::vector<char> fmap(off);
        for (size_t i = 0; i < off; ++i) fmap[i] = metadata[i] == 1;
        std::vector<float> D(k); std::vector<faiss::idx_t> I(k);
        index.search(1, xb.data() + 5 * d, k, D.data(), I.data(), fmap.data());
        int bad = 0, found = 0;
        for (int j = 0; j < k; ++j) { if (I[j] >= 0) { found++; if (metadata[I[j]] != 1) bad++; } }
        std::printf("  ntotal=%lld found=%d violations=%d\n", (long long)index.ntotal, found, bad); std::fflush(stdout);
    }
    return 0;
}
