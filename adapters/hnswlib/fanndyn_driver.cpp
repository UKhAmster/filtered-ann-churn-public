// Драйвер hnswlib для протокола fanndyn (формат ввода-вывода — adapters/common/fanndyn_io.h).
//
// Заменяет пилотный прогон через Python-обвязку: фильтр вызывается из C++, поэтому задержка
// не содержит накладных расходов интерпретатора, а работа поиска считается точно — через
// пространство L2, увеличивающее счётчик потока при каждом вычислении расстояния.
//
// Режимы удаления: mark — markDelete, новые точки занимают новые места; replace — новые
// точки занимают места помеченных (allow_replace_deleted).
#include "hnswlib/hnswlib.h"

#include <memory>
#include <omp.h>

#include "fanndyn_io.h"

using namespace fanndyn;

static thread_local uint64_t g_cmps = 0;
static hnswlib::DISTFUNC<float> g_l2 = nullptr;

static float counting_l2(const void *a, const void *b, const void *param) {
    ++g_cmps;
    return g_l2(a, b, param);
}

class CountingL2Space : public hnswlib::SpaceInterface<float> {
    hnswlib::L2Space inner_;

  public:
    explicit CountingL2Space(size_t dim) : inner_(dim) { g_l2 = inner_.get_dist_func(); }
    size_t get_data_size() override { return inner_.get_data_size(); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return counting_l2; }
    void *get_dist_func_param() override { return inner_.get_dist_func_param(); }
};

struct MaskFilter : hnswlib::BaseFilterFunctor {
    const std::vector<char> &mask;
    explicit MaskFilter(const std::vector<char> &m) : mask(m) {}
    bool operator()(hnswlib::labeltype id) override { return mask[id]; }
};

int main(int argc, char **argv) {
    Args a(argc, argv,
           {{"dir", ""}, {"mode", "mark"}, {"filters", "-1"}, {"K", "10"}, {"M", "16"}, {"ef-construction", "200"},
            {"ef-search", "64"}, {"threads", "8"}, {"seed", "7"}, {"capacity", "0"}});
    const std::string dir = a.s("dir"), mode = a.s("mode");
    if (mode != "mark" && mode != "replace") throw std::runtime_error("mode: mark | replace");
    const uint32_t K = a.i("K");
    const int threads = a.i("threads");
    auto filters = parse_filters(a.s("filters"));

    uint32_t n, d;
    auto base = read_bin(dir + "/base.bin", n, d);
    auto labels = read_labels(dir);
    auto ops = read_raw<uint32_t>(dir + "/ops.u32");
    std::vector<Queries> queries;
    for (size_t f = 0; f < filters.size(); ++f) queries.push_back(read_queries(dir, f));

    CountingL2Space space(d);
    size_t capacity = a.i("capacity") > 0 ? size_t(a.i("capacity")) : size_t(ops[1]);
    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        &space, capacity, a.i("M"), a.i("ef-construction"), a.i("seed"), mode == "replace");
    index->setEf(a.i("ef-search"));

    std::vector<char> alive(n, 0);
    uint64_t marked = 0, deleted_total = 0;
    double insert_s = 0;

    auto ensure_room = [&](size_t add) {
        size_t reusable = mode == "replace" ? marked : 0;
        size_t free_slots = index->max_elements_ - index->cur_element_count + reusable;
        if (add > free_slots)
            index->resizeIndex(std::max(2 * index->max_elements_, index->max_elements_ + add - free_slots));
    };

    std::ofstream stats(dir + "/stats.tsv");
    stats << "checkpoint\tn_lazy_pending\tconsolidations\tconsolidate_s_total\tinsert_s_total\n";
    uint32_t ck = 0;
    for (size_t p = 0; p < ops.size();) {
        uint32_t kind = ops[p], cnt = ops[p + 1];
        const uint32_t *ids = ops.data() + p + 2;
        p += 2 + cnt;
        if (kind == 1) {
            auto t0 = Clock::now();
            ensure_room(cnt);
            bool replace = mode == "replace";
            if (cnt > 1) {
                omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic, 64)
                for (int64_t i = 0; i < int64_t(cnt); ++i)
                    index->addPoint(base.data() + size_t(ids[i]) * d, ids[i], replace);
            } else {
                index->addPoint(base.data() + size_t(ids[0]) * d, ids[0], replace);
            }
            if (replace) marked = marked > cnt ? marked - cnt : 0;
            for (uint32_t i = 0; i < cnt; ++i) alive[ids[i]] = 1;
            insert_s += std::chrono::duration<double>(Clock::now() - t0).count();
        } else if (kind == 2) {
            for (uint32_t i = 0; i < cnt; ++i) {
                index->markDelete(ids[i]);
                alive[ids[i]] = 0;
            }
            marked += cnt;
            deleted_total += cnt;
        } else if (kind == 3) {
            for (size_t f = 0; f < filters.size(); ++f) {
                std::vector<char> mask(n, 0);
                for (uint32_t id = 0; id < n; ++id) mask[id] = filters[f] < 0 || labels.has(id, uint32_t(filters[f]));
                MaskFilter flt(mask);
                const auto &qs = queries[f];
                std::vector<uint32_t> res(size_t(qs.nq) * K, kMissing), cmps(qs.nq);
                std::vector<float> lat(qs.nq);
                for (uint32_t q = 0; q < qs.nq; ++q) {
                    g_cmps = 0;
                    auto t0 = Clock::now();
                    auto top = index->searchKnn(qs.row(q), K, filters[f] < 0 ? nullptr : &flt);
                    lat[q] = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
                    cmps[q] = uint32_t(g_cmps);
                    for (size_t pos = top.size(); !top.empty(); top.pop())
                        res[size_t(q) * K + (--pos)] = uint32_t(top.top().second);
                }
                write_vec(out_name(dir, "res", ck, f, ".u32"), res);
                write_vec(out_name(dir, "lat", ck, f, ".f32"), lat);
                write_vec(out_name(dir, "cmps", ck, f, ".u32"), cmps);
            }
            stats << ck << "\t" << marked << "\t0\t0\t" << insert_s << "\n";
            stats.flush();
            std::cerr << "checkpoint " << ck << " deleted=" << deleted_total << std::endl;
            ck++;
        } else {
            throw std::runtime_error("неизвестный вид операции " + std::to_string(kind));
        }
    }
    return 0;
}
