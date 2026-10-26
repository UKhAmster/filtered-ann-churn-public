// Драйвер ACORN для протокола fanndyn: тот же формат ввода-вывода, что у драйвера
// Filtered-DiskANN (см. adapters/filtered_diskann/fanndyn_driver.cpp).
//
// Удаления в ACORN нет (remove_ids не переопределён), поэтому они эмулируются так,
// как это делают системы без штатного удаления: удалённый объект исключается маской
// фильтра, а граф не чинится (режим tombstone). Режим rebuild дополнительно
// перестраивает индекс с нуля на живых объектах, когда доля удалённых достигает
// --rebuild-at от числа живых.
//
// Вставки буферизуются до ближайшей контрольной точки или перестройки: IndexACORN::add
// на каждый вызов заводит массив из ntotal блокировок, то есть одиночная вставка стоит
// O(N), а на одиночных вставках после построения драйвер падал. Удаления графа не
// касаются, поэтому набор и порядок вставленных точек сохраняются; меняется лишь то,
// что граф новых точек строится пакетом.
//
// ACORN хранит указатель на вектор метаданных, поэтому вектор выделяется сразу на все
// id и больше не меняет размер. Внутренний номер FAISS совпадает с порядком добавления;
// при перестройке ведётся таблица соответствия номеров и id.
#include <faiss/IndexACORN.h>
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::vector<uint32_t> read_u32(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("нет файла " + path);
    auto size = in.tellg();
    in.seekg(0);
    std::vector<uint32_t> out(size / sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(out.data()), size);
    return out;
}

static std::vector<float> read_bin(const std::string &path, uint32_t &n, uint32_t &d) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("нет файла " + path);
    in.read(reinterpret_cast<char *>(&n), 4);
    in.read(reinterpret_cast<char *>(&d), 4);
    std::vector<float> out(size_t(n) * d);
    in.read(reinterpret_cast<char *>(out.data()), out.size() * sizeof(float));
    return out;
}

template <typename V> static void write_vec(const std::string &path, const std::vector<V> &v) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(V));
}

struct State {
    uint32_t d;
    int M, gamma, M_beta, ef_search, build_threads;
    const std::vector<float> *base;
    std::vector<int> meta;              // метка по внутреннему номеру, размер фиксирован
    std::vector<uint32_t> slot_to_id;   // внутренний номер -> id объекта
    std::vector<char> alive_by_id;
    std::unique_ptr<faiss::IndexACORNFlat> index;

    void fresh() {
        slot_to_id.clear();
        index = std::make_unique<faiss::IndexACORNFlat>(int(d), M, gamma, meta, M_beta);
        index->acorn.efSearch = ef_search;
    }

    void add(const std::vector<uint32_t> &ids, const std::vector<uint32_t> &labels) {
        std::vector<float> buf(ids.size() * size_t(d));
        for (size_t i = 0; i < ids.size(); ++i) {
            size_t slot = slot_to_id.size() + i;
            if (slot >= meta.size()) throw std::runtime_error("нет места под метаданные");
            meta[slot] = int(labels[ids[i]]);
            std::copy_n(base->data() + size_t(ids[i]) * d, d, buf.data() + i * d);
        }
        omp_set_num_threads(build_threads);
        index->add(faiss::idx_t(ids.size()), buf.data());
        omp_set_num_threads(1);
        for (auto id : ids) slot_to_id.push_back(id);
    }
};

int main(int argc, char **argv) {
    std::string dir, mode = "tombstone", filters_arg = "-1";
    int M = 32, gamma = 12, M_beta = 64, ef = 64, threads = 8;
    uint32_t K = 10;
    double rebuild_at = 0.2;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--dir") dir = v;
        else if (k == "--mode") mode = v;
        else if (k == "--filters") filters_arg = v;
        else if (k == "--M") M = std::stoi(v);
        else if (k == "--gamma") gamma = std::stoi(v);
        else if (k == "--M-beta") M_beta = std::stoi(v);
        else if (k == "--ef-search") ef = std::stoi(v);
        else if (k == "--K") K = std::stoul(v);
        else if (k == "--threads") threads = std::stoi(v);
        else if (k == "--rebuild-at") rebuild_at = std::stod(v);
        else throw std::runtime_error("неизвестный параметр " + k);
    }
    if (mode != "tombstone" && mode != "rebuild") throw std::runtime_error("mode: tombstone | rebuild");
    std::vector<int> filters;
    {
        std::stringstream ss(filters_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) filters.push_back(std::stoi(tok));
    }

    uint32_t n, d, nq, dq;
    auto base = read_bin(dir + "/base.bin", n, d);
    auto queries = read_bin(dir + "/query.bin", nq, dq);
    auto labels = read_u32(dir + "/labels.u32");
    auto ops = read_u32(dir + "/ops.u32");

    State st{d, M, gamma, M_beta, ef, threads, &base, std::vector<int>(n, 0), {}, std::vector<char>(n, 0), nullptr};
    omp_set_num_threads(1);
    st.fresh();

    std::ofstream stats(dir + "/stats.tsv");
    stats << "checkpoint\tn_lazy_pending\tconsolidations\tconsolidate_s_total\tinsert_s_total\n";
    uint64_t dead = 0, rebuilds = 0;
    double rebuild_s = 0, insert_s = 0;
    uint32_t ck = 0;
    std::vector<uint32_t> pending_ins;
    auto flush = [&]() {
        if (pending_ins.empty()) return;
        auto t0 = Clock::now();
        st.add(pending_ins, labels);
        insert_s += std::chrono::duration<double>(Clock::now() - t0).count();
        pending_ins.clear();
    };

    for (size_t p = 0; p < ops.size();) {
        uint32_t kind = ops[p], cnt = ops[p + 1];
        std::vector<uint32_t> ids(ops.begin() + p + 2, ops.begin() + p + 2 + cnt);
        p += 2 + cnt;
        if (kind == 1) {
            pending_ins.insert(pending_ins.end(), ids.begin(), ids.end());
            for (auto id : ids) st.alive_by_id[id] = 1;
        } else if (kind == 2) {
            for (auto id : ids) st.alive_by_id[id] = 0;
            dead += cnt;
            size_t live = st.slot_to_id.size() + pending_ins.size() - dead;
            if (mode == "rebuild" && dead >= rebuild_at * live) {
                auto t0 = Clock::now();
                std::vector<uint32_t> keep;
                for (auto id : st.slot_to_id)
                    if (st.alive_by_id[id]) keep.push_back(id);
                for (auto id : pending_ins)
                    if (st.alive_by_id[id]) keep.push_back(id);
                pending_ins.clear();
                st.fresh();
                st.add(keep, labels);
                rebuild_s += std::chrono::duration<double>(Clock::now() - t0).count();
                rebuilds++;
                dead = 0;
            }
        } else if (kind == 3) {
            flush();
            size_t N = st.slot_to_id.size();
            for (size_t f = 0; f < filters.size(); ++f) {
                // Маска одна на все запросы: удалённые исключены, фильтр по метке.
                std::vector<char> row(N);
                for (size_t s = 0; s < N; ++s) {
                    uint32_t id = st.slot_to_id[s];
                    row[s] = st.alive_by_id[id] && (filters[f] < 0 || int(labels[id]) == filters[f]);
                }
                std::vector<uint32_t> res(size_t(nq) * K, 0xFFFFFFFFu);
                std::vector<float> lat(nq), dist(K);
                std::vector<faiss::idx_t> lab(K);
                for (uint32_t q = 0; q < nq; ++q) {
                    auto t0 = Clock::now();
                    st.index->search(1, &queries[size_t(q) * d], K, dist.data(), lab.data(), row.data());
                    lat[q] = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
                    for (uint32_t i = 0; i < K; ++i)
                        if (lab[i] >= 0) res[size_t(q) * K + i] = st.slot_to_id[size_t(lab[i])];
                }
                write_vec(dir + "/res_c" + std::to_string(ck) + "_f" + std::to_string(f) + ".u32", res);
                write_vec(dir + "/lat_c" + std::to_string(ck) + "_f" + std::to_string(f) + ".f32", lat);
            }
            stats << ck << "\t" << dead << "\t" << rebuilds << "\t" << rebuild_s << "\t" << insert_s << "\n";
            stats.flush();
            std::cerr << "checkpoint " << ck << " dead=" << dead << " rebuilds=" << rebuilds << std::endl;
            ck++;
        } else {
            throw std::runtime_error("неизвестный вид операции " + std::to_string(kind));
        }
    }
    return 0;
}
