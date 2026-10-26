// UNG driver for the fanndyn protocol (input/output format: adapters/common/fanndyn_io.h).
//
// UNG is static: UniNavGraph has only build, search, save and load. So its life
// under a stream of changes is modeled the way a static index lives in
// production: between rebuilds the index serves the previous collection, deleted
// objects are filtered out of answers (for that --fetch candidates are requested instead
// of K), new objects are invisible; when the number of deleted objects reaches --rebuild-at
// of the live count, the index is rebuilt on all live objects, including new ones.
//
// Label filter is the containment scenario (the object contains the label): an object may
// have several labels. Labels are shifted by +1: label 0 in UNG is taken by the trie root.
// UNG searches in a batch, so latency is the batch mean in one thread, the same for all
// queries; UNG returns the number of distance computations for every query.
#include "uni_nav_graph.h"

#include <memory>

#include "fanndyn_io.h"

using namespace fanndyn;

static void write_subset(const std::string &bin, const std::string &lab, const std::vector<float> &base, uint32_t d,
                         const std::vector<uint32_t> &ids, const Labels &labels) {
    std::ofstream b(bin, std::ios::binary);
    uint32_t n = ids.size();
    b.write(reinterpret_cast<const char *>(&n), 4);
    b.write(reinterpret_cast<const char *>(&d), 4);
    for (auto id : ids) b.write(reinterpret_cast<const char *>(base.data() + size_t(id) * d), d * sizeof(float));
    std::ofstream l(lab);
    for (auto id : ids) {
        auto ls = labels.of(id);
        for (size_t k = 0; k < ls.size(); ++k) l << (k ? "," : "") << ls[k] + 1;
        l << "\n";
    }
}

int main(int argc, char **argv) {
    Args a(argc, argv,
           {{"dir", ""}, {"mode", "rebuild"}, {"filters", "-1"}, {"K", "10"}, {"fetch", "50"}, {"Lsearch", "100"},
            {"entry-points", "16"}, {"threads", "8"}, {"R", "32"}, {"Lbuild", "100"}, {"cross-edges", "6"},
            {"alpha", "1.2"}, {"rebuild-at", "0.2"}});
    const std::string dir = a.s("dir");
    if (a.s("mode") != "rebuild") throw std::runtime_error("mode: rebuild");
    const uint32_t K = a.i("K"), fetch = a.i("fetch"), Ls = a.i("Lsearch"), entry = a.i("entry-points");
    const uint32_t threads = a.i("threads");
    const double rebuild_at = a.f("rebuild-at");
    auto filters = parse_filters(a.s("filters"));

    uint32_t n, d;
    auto base = read_bin(dir + "/base.bin", n, d);
    auto labels = read_labels(dir);
    auto ops = read_raw<uint32_t>(dir + "/ops.u32");
    std::shared_ptr<ANNS::DistanceHandler> dist(ANNS::get_distance_handler("float", "L2").release());

    // Query stores: one per filter (for unfiltered search the label does not matter).
    std::vector<std::shared_ptr<ANNS::IStorage>> qstorages;
    std::vector<uint32_t> nqs;
    for (size_t f = 0; f < filters.size(); ++f) {
        std::string own = dir + "/query_f" + std::to_string(f) + ".bin";
        std::string qbin = exists(own) ? own : dir + "/query.bin";
        uint32_t nq, dq;
        read_bin(qbin, nq, dq);
        std::string qlab = "/tmp/ung_qlabels_" + std::to_string(f) + ".txt";
        std::ofstream ql(qlab);
        for (uint32_t q = 0; q < nq; ++q) ql << (filters[f] < 0 ? 1 : filters[f] + 1) << "\n";
        ql.close();
        auto qs = ANNS::create_storage("float", false);
        qs->load_from_file(qbin, qlab);
        qstorages.push_back(qs);
        nqs.push_back(nq);
    }

    std::vector<char> alive(n, 0), in_index(n, 0);
    std::vector<uint32_t> index_ids;
    std::unique_ptr<ANNS::UniNavGraph> index;
    std::shared_ptr<ANNS::IStorage> bstorage;
    uint64_t dead_in_index = 0, rebuilds = 0, live = 0;
    double rebuild_s = 0;

    auto rebuild = [&]() {
        auto t0 = Clock::now();
        std::fill(in_index.begin(), in_index.end(), 0);
        index_ids.clear();
        for (uint32_t id = 0; id < n; ++id)
            if (alive[id]) index_ids.push_back(id);
        for (auto id : index_ids) in_index[id] = 1;
        write_subset("/tmp/ung_base.bin", "/tmp/ung_base_labels.txt", base, d, index_ids, labels);
        bstorage = ANNS::create_storage("float", false);
        bstorage->load_from_file("/tmp/ung_base.bin", "/tmp/ung_base_labels.txt");
        index = std::make_unique<ANNS::UniNavGraph>();
        index->build(bstorage, dist, "general", "Vamana", threads, a.i("cross-edges"), a.i("R"), a.i("Lbuild"),
                     float(a.f("alpha")));
        rebuild_s += std::chrono::duration<double>(Clock::now() - t0).count();
        rebuilds++;
        dead_in_index = 0;
    };

    std::ofstream stats(dir + "/stats.tsv");
    stats << "checkpoint\tn_lazy_pending\tconsolidations\tconsolidate_s_total\tinsert_s_total\n";
    uint32_t ck = 0;
    bool built = false;
    for (size_t p = 0; p < ops.size();) {
        uint32_t kind = ops[p], cnt = ops[p + 1];
        const uint32_t *ids = ops.data() + p + 2;
        p += 2 + cnt;
        if (kind == 1) {
            for (uint32_t i = 0; i < cnt; ++i) alive[ids[i]] = 1;
            live += cnt;
            if (!built) {
                rebuild();
                built = true;
            }
        } else if (kind == 2) {
            for (uint32_t i = 0; i < cnt; ++i) {
                alive[ids[i]] = 0;
                if (in_index[ids[i]]) dead_in_index++;
            }
            live -= cnt;
            if (dead_in_index >= rebuild_at * live) rebuild();
        } else if (kind == 3) {
            for (size_t f = 0; f < filters.size(); ++f) {
                const uint32_t nq = nqs[f];
                std::vector<std::pair<ANNS::IdxType, float>> results(size_t(nq) * fetch);
                std::vector<float> num_cmps(nq);
                auto t0 = Clock::now();
                index->search(qstorages[f], dist, 1, std::max(Ls, fetch), entry,
                              filters[f] < 0 ? "nofilter" : "containment", fetch, results.data(), num_cmps);
                float per_query = std::chrono::duration<float, std::milli>(Clock::now() - t0).count() / nq;
                std::vector<uint32_t> res(size_t(nq) * K, kMissing), cmps(nq);
                for (uint32_t q = 0; q < nq; ++q) {
                    cmps[q] = uint32_t(num_cmps[q]);
                    uint32_t pos = 0;
                    for (uint32_t j = 0; j < fetch && pos < K; ++j) {
                        auto slot = results[size_t(q) * fetch + j].first;
                        if (slot == ANNS::IdxType(-1) || slot >= index_ids.size()) continue;
                        uint32_t id = index_ids[slot];
                        if (alive[id]) res[size_t(q) * K + pos++] = id;
                    }
                }
                write_vec(out_name(dir, "res", ck, f, ".u32"), res);
                write_vec(out_name(dir, "lat", ck, f, ".f32"), std::vector<float>(nq, per_query));
                write_vec(out_name(dir, "cmps", ck, f, ".u32"), cmps);
            }
            stats << ck << "\t" << dead_in_index << "\t" << rebuilds << "\t" << rebuild_s << "\t0\n";
            stats.flush();
            std::cerr << "checkpoint " << ck << " dead_in_index=" << dead_in_index << " rebuilds=" << rebuilds
                      << std::endl;
            ck++;
        } else {
            throw std::runtime_error("unknown operation kind " + std::to_string(kind));
        }
    }
    return 0;
}
