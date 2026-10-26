// Общий ввод-вывод драйверов протокола fanndyn.
//
// Каталог прогона (--dir):
//   base.bin              векторы, формат DiskANN: uint32 n, uint32 d, float32 n*d
//   labels_ptr.u64        метки в CSR: n+1 смещений
//   labels_idx.u32        метки в CSR: номера меток
//   ops.u32               поток операций: вид, число id, id...
//                         (1 — вставка, 2 — удаление, 3 — контрольная точка)
//   query.bin             запросы по умолчанию
//   query_f<j>.bin        запросы для фильтра j, если есть (иначе query.bin)
// Выход для контрольной точки i и фильтра j:
//   res_c<i>_f<j>.u32     nq*K id, 0xFFFFFFFF — нет результата
//   lat_c<i>_f<j>.f32     мс на запрос
//   cmps_c<i>_f<j>.u32    вычислений расстояний на запрос
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fanndyn {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kMissing = 0xFFFFFFFFu;

template <typename V> inline std::vector<V> read_raw(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("нет файла " + path);
    auto size = in.tellg();
    in.seekg(0);
    std::vector<V> out(size / sizeof(V));
    in.read(reinterpret_cast<char *>(out.data()), size);
    return out;
}

inline std::vector<float> read_bin(const std::string &path, uint32_t &n, uint32_t &d) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("нет файла " + path);
    in.read(reinterpret_cast<char *>(&n), 4);
    in.read(reinterpret_cast<char *>(&d), 4);
    std::vector<float> out(size_t(n) * d);
    in.read(reinterpret_cast<char *>(out.data()), out.size() * sizeof(float));
    return out;
}

inline bool exists(const std::string &path) { return std::ifstream(path).good(); }

template <typename V> inline void write_vec(const std::string &path, const std::vector<V> &v) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(V));
}

inline std::string out_name(const std::string &dir, const char *kind, uint32_t ck, size_t f, const char *ext) {
    return dir + "/" + kind + "_c" + std::to_string(ck) + "_f" + std::to_string(f) + ext;
}

struct Labels {
    std::vector<uint64_t> ptr;
    std::vector<uint32_t> idx;
    std::vector<uint32_t> of(uint32_t id) const {
        return std::vector<uint32_t>(idx.begin() + ptr[id], idx.begin() + ptr[id + 1]);
    }
    bool has(uint32_t id, uint32_t label) const {
        for (uint64_t k = ptr[id]; k < ptr[id + 1]; ++k)
            if (idx[k] == label) return true;
        return false;
    }
    uint32_t max_label() const {
        uint32_t m = 0;
        for (auto l : idx) m = std::max(m, l);
        return m;
    }
};

inline Labels read_labels(const std::string &dir) {
    return Labels{read_raw<uint64_t>(dir + "/labels_ptr.u64"), read_raw<uint32_t>(dir + "/labels_idx.u32")};
}

struct Queries {
    uint32_t nq = 0, d = 0;
    std::vector<float> data;
    const float *row(uint32_t q) const { return data.data() + size_t(q) * d; }
};

inline Queries read_queries(const std::string &dir, size_t filter) {
    std::string own = dir + "/query_f" + std::to_string(filter) + ".bin";
    Queries q;
    q.data = read_bin(exists(own) ? own : dir + "/query.bin", q.nq, q.d);
    return q;
}

inline std::vector<int> parse_filters(const std::string &arg) {
    std::vector<int> out;
    std::stringstream ss(arg);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
    return out;
}

// Аргументы вида --ключ значение; неизвестный ключ — ошибка.
struct Args {
    std::map<std::string, std::string> kv;
    Args(int argc, char **argv, const std::map<std::string, std::string> &defaults) : kv(defaults) {
        for (int i = 1; i + 1 < argc; i += 2) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0 || !kv.count(k.substr(2))) throw std::runtime_error("неизвестный параметр " + k);
            kv[k.substr(2)] = argv[i + 1];
        }
    }
    const std::string &s(const std::string &k) const { return kv.at(k); }
    long long i(const std::string &k) const { return std::stoll(kv.at(k)); }
    double f(const std::string &k) const { return std::stod(kv.at(k)); }
};

}  // namespace fanndyn
