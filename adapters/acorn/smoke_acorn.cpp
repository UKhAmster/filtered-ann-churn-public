// Дымовой пример ACORN на синтетических данных.
//
// Собственный пример проекта (demos/test_acorn.cpp) не годится для дымовой
// проверки: он требует внешний датасет SIFT с заранее посчитанными метаданными,
// эталонами и файлами предикатов. Его сборка проверяется отдельной целью в
// Dockerfile, а работоспособность индекса — этим примером.
//
// Код следует разделу README «Example Usage», но по фактическому API: README
// показывает конструктор из четырёх аргументов, а IndexACORN.h требует пяти —
// метаданные передаются при построении.
#include <cstdio>
#include <random>
#include <vector>

#include <faiss/IndexACORN.h>

int main() {
    const int d = 32;
    const int N = 2000;
    const int nq = 5;
    const int k = 10;
    const int M = 32;
    const int gamma = 12;
    const int M_beta = 32;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::vector<float> xb(static_cast<size_t>(N) * d);
    for (auto& value : xb) {
        value = uniform(rng);
    }

    // Один атрибут на вектор, четыре значения — минимальная модель фильтра.
    std::vector<int> metadata(N);
    for (int i = 0; i < N; ++i) {
        metadata[i] = i % 4;
    }

    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);
    index.add(N, xb.data());

    std::vector<float> xq(xb.begin(), xb.begin() + static_cast<size_t>(nq) * d);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(k) * nq);
    std::vector<float> distances(static_cast<size_t>(k) * nq);

    // Предикат: пропускаем только векторы с атрибутом, равным атрибуту запроса.
    std::vector<char> filter_ids_map(static_cast<size_t>(nq) * N);
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < N; ++i) {
            filter_ids_map[static_cast<size_t>(q) * N + i] =
                    static_cast<char>(metadata[i] == metadata[q]);
        }
    }

    index.search(nq, xq.data(), k, distances.data(), labels.data(), filter_ids_map.data());

    int violations = 0;
    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            const faiss::idx_t id = labels[static_cast<size_t>(q) * k + j];
            if (id >= 0 && metadata[id] != metadata[q]) {
                ++violations;
            }
        }
    }

    std::printf("smoke ok nq=%d k=%d violations=%d\n", nq, k, violations);
    return violations == 0 ? 0 : 1;
}
