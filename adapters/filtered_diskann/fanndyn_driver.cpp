// Драйвер Filtered-DiskANN для протокола fanndyn (формат ввода-вывода —
// adapters/common/fanndyn_io.h): проигрывает runbook на динамическом фильтрованном индексе.
//
// Режимы: lazy — ленивые удаления без уплотнения; consolidate — уплотнение графа каждые
// --consolidate-every удалений. С --probe-list 1 дополнительно пишется alive_c<i>_f<j>.u32:
// сколько живых вершин в списке из L лучших кандидатов (повторный поиск с K = L).
//
// Два обхода закрытых полей через явное инстанцирование шаблона (при нём проверка доступа
// не выполняется); исходники DiskANN не меняются:
//  * search_with_tags ищет фильтр в строковом словаре _label_map, который у динамического
//    индекса не заполняется, — словарь заполняется драйвером;
//  * для счёта работы поиска функция расстояния хранилища (_distance_fn) заменяется
//    наследником DistanceL2Float, увеличивающим счётчик потока при каждом вызове compare.
#include "index.h"
#include "index_factory.h"
#include "in_mem_data_store.h"

#include <cmath>
#include <memory>
#include <unordered_map>

#include "fanndyn_io.h"

using namespace fanndyn;
using TagT = uint32_t;
using LabelT = uint32_t;
using IndexT = diskann::Index<float, TagT, LabelT>;
using StoreT = diskann::InMemDataStore<float>;

template <typename Tag, typename Tag::type M> struct Expose {
    friend typename Tag::type member(Tag) { return M; }
};
struct LabelMapTag {
    using type = std::unordered_map<std::string, LabelT> IndexT::*;
    friend type member(LabelMapTag);
};
template struct Expose<LabelMapTag, &IndexT::_label_map>;
struct DataStoreTag {
    using type = std::shared_ptr<diskann::AbstractDataStore<float>> IndexT::*;
    friend type member(DataStoreTag);
};
template struct Expose<DataStoreTag, &IndexT::_data_store>;
struct DistanceTag {
    using type = std::unique_ptr<diskann::Distance<float>> StoreT::*;
    friend type member(DistanceTag);
};
template struct Expose<DistanceTag, &StoreT::_distance_fn>;

static thread_local uint64_t g_cmps = 0;

class CountingL2 : public diskann::DistanceL2Float {
  public:
    float compare(const float *a, const float *b, uint32_t size) const override {
        ++g_cmps;
        return diskann::DistanceL2Float::compare(a, b, size);
    }
};

int main(int argc, char **argv) {
    Args a(argc, argv,
           {{"dir", ""}, {"mode", "lazy"}, {"filters", "-1"}, {"R", "32"}, {"Lbuild", "100"}, {"Lsearch", "64"},
            {"K", "10"}, {"threads", "8"}, {"n-labels", "0"}, {"consolidate-every", "10000"}, {"alpha", "1.2"},
            {"probe-list", "0"}});
    const std::string dir = a.s("dir"), mode = a.s("mode");
    if (mode != "lazy" && mode != "consolidate") throw std::runtime_error("mode: lazy | consolidate");
    const uint32_t R = a.i("R"), Lb = a.i("Lbuild"), Ls = a.i("Lsearch"), K = a.i("K"), threads = a.i("threads");
    const uint64_t consolidate_every = a.i("consolidate-every");
    const bool probe = a.s("probe-list") == "1";
    auto filters = parse_filters(a.s("filters"));

    uint32_t n, d;
    auto base = read_bin(dir + "/base.bin", n, d);
    auto labels = read_labels(dir);
    auto ops = read_raw<uint32_t>(dir + "/ops.u32");
    const uint32_t n_labels = a.i("n-labels") > 0 ? uint32_t(a.i("n-labels")) : labels.max_label() + 1;
    std::vector<Queries> queries;
    for (size_t f = 0; f < filters.size(); ++f) queries.push_back(read_queries(dir, f));

    auto wparams = diskann::IndexWriteParametersBuilder(Lb, R)
                       .with_alpha(float(a.f("alpha")))
                       .with_max_occlusion_size(500)
                       .with_num_threads(threads)
                       .with_filter_list_size(Lb)
                       .build();
    auto search_params = diskann::IndexSearchParams(Ls, 1);
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(diskann::L2)
                      .with_dimension(d)
                      .with_max_points(n)
                      .is_dynamic_index(true)
                      .is_enable_tags(true)
                      .is_filtered(true)
                      .is_use_opq(false)
                      .with_num_pq_chunks(0)
                      .is_pq_dist_build(false)
                      .with_num_frozen_pts(n_labels)
                      .with_tag_type("uint32")
                      .with_label_type("uint")
                      .with_data_type("float")
                      .with_index_write_params(wparams)
                      .with_index_search_params(search_params)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .build();
    auto abstract = diskann::IndexFactory(config).create_instance();
    auto *index = dynamic_cast<IndexT *>(abstract.get());
    if (!index) throw std::runtime_error("неожиданный тип индекса");
    for (LabelT l = 0; l < n_labels; ++l) (index->*member(LabelMapTag{}))[std::to_string(l)] = l;
    auto *store = dynamic_cast<StoreT *>((index->*member(DataStoreTag{})).get());
    if (!store) throw std::runtime_error("неожиданный тип хранилища");
    (store->*member(DistanceTag{})) = std::make_unique<CountingL2>();

    double norm = 0;
    const uint32_t sample = std::min<uint32_t>(n, 1000);
    for (uint32_t i = 0; i < sample; ++i) {
        double s = 0;
        for (uint32_t j = 0; j < d; ++j) s += double(base[size_t(i) * d + j]) * base[size_t(i) * d + j];
        norm += std::sqrt(s);
    }
    index->set_start_points_at_random(float(norm / sample));

    std::ofstream stats(dir + "/stats.tsv");
    stats << "checkpoint\tn_lazy_pending\tconsolidations\tconsolidate_s_total\tinsert_s_total\n";
    uint64_t pending = 0, consolidations = 0;
    double consolidate_s = 0, insert_s = 0;
    uint32_t ck = 0;
    std::vector<TagT> tags(Ls);
    std::vector<float> dists(Ls);
    std::vector<float *> no_vectors;

    auto search = [&](const float *q, uint32_t k, int flt) -> size_t {
        return flt < 0 ? index->search_with_tags(q, k, Ls, tags.data(), dists.data(), no_vectors)
                       : index->search_with_tags(q, k, Ls, tags.data(), dists.data(), no_vectors, true,
                                                 std::to_string(flt));
    };

    for (size_t p = 0; p < ops.size();) {
        uint32_t kind = ops[p], cnt = ops[p + 1];
        const uint32_t *ids = ops.data() + p + 2;
        p += 2 + cnt;
        if (kind == 1) {
            auto t0 = Clock::now();
#pragma omp parallel for num_threads(cnt > 1 ? threads : 1) schedule(dynamic, 64)
            for (int64_t i = 0; i < int64_t(cnt); ++i) {
                uint32_t id = ids[i];
                auto ls = labels.of(id);
                // detect_common_filters пересекает метки методом двух указателей и требует
                // сортировки; insert_point её не проверяет.
                std::sort(ls.begin(), ls.end());
                if (index->insert_point(&base[size_t(id) * d], TagT(id + 1), ls) != 0)
                    throw std::runtime_error("insert_point не удался для id " + std::to_string(id));
            }
            insert_s += std::chrono::duration<double>(Clock::now() - t0).count();
        } else if (kind == 2) {
            for (uint32_t i = 0; i < cnt; ++i)
                if (index->lazy_delete(TagT(ids[i] + 1)) != 0)
                    throw std::runtime_error("lazy_delete не удался для id " + std::to_string(ids[i]));
            pending += cnt;
            if (mode == "consolidate" && pending >= consolidate_every) {
                auto t0 = Clock::now();
                index->consolidate_deletes(wparams);
                consolidate_s += std::chrono::duration<double>(Clock::now() - t0).count();
                consolidations++;
                pending = 0;
            }
        } else if (kind == 3) {
            for (size_t f = 0; f < filters.size(); ++f) {
                const auto &qs = queries[f];
                std::vector<uint32_t> res(size_t(qs.nq) * K, kMissing), cmps(qs.nq), alive(qs.nq);
                std::vector<float> lat(qs.nq);
                for (uint32_t q = 0; q < qs.nq; ++q) {
                    g_cmps = 0;
                    auto t0 = Clock::now();
                    size_t got = search(qs.row(q), K, filters[f]);
                    lat[q] = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
                    cmps[q] = uint32_t(g_cmps);
                    for (size_t i = 0; i < std::min<size_t>(got, K); ++i) res[size_t(q) * K + i] = tags[i] - 1;
                    if (probe) alive[q] = uint32_t(search(qs.row(q), Ls, filters[f]));
                }
                write_vec(out_name(dir, "res", ck, f, ".u32"), res);
                write_vec(out_name(dir, "lat", ck, f, ".f32"), lat);
                write_vec(out_name(dir, "cmps", ck, f, ".u32"), cmps);
                if (probe) write_vec(out_name(dir, "alive", ck, f, ".u32"), alive);
            }
            stats << ck << "\t" << pending << "\t" << consolidations << "\t" << consolidate_s << "\t" << insert_s
                  << "\n";
            stats.flush();
            std::cerr << "checkpoint " << ck << " pending=" << pending << " consolidations=" << consolidations
                      << std::endl;
            ck++;
        } else {
            throw std::runtime_error("неизвестный вид операции " + std::to_string(kind));
        }
    }
    return 0;
}
