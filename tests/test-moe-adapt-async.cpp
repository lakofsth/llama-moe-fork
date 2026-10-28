// fork-strata-ports ADAPT-ASYNC (tests/INVARIANTS.md): replay adaptive-tier rounds through a fake IO that models the
// compute stream, the copy stream and their two events, and check
//   (1) no copy writes a slot that a map published before its copy-free event still marks resident, and
//   (2) no map admits a copied slot before the compute stream has waited on that copy's done event, and
//   (3) every published map is consistent: each expert resident on exactly one side, an in-flight slot on neither.
// The checker is itself checked on two hand-misordered sequences that must be caught.
// ADAPT-PF: every map set_maps publishes goes through llama_moe_adapt_write_maps, and with LLAMA_MOE_PREFETCH_PP
// (map_cpu_pf present) it writes map_cpu_pf and the prefetch plan with map_cpu: the plan is exactly the experts the CPU
// serves (mask_cpu 1), ascending, and map_cpu_pf gives each its index in it, a resident one its sentinel.

#include "../src/llama-moe-adapt.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { n_fail++; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

static const int NEXP = 64;

struct fake_io {
    std::vector<llama_moe_adapt_layer> * L = nullptr;
    std::vector<llama_moe_adapt_maps> published;              // per layer, as of the latest set_maps (compute order)
    std::map<int, std::vector<llama_moe_adapt_maps>> at_free; // per dev: the maps as of begin_copies (copy-free event)
    std::map<int, bool> gated;                                // per dev: copy stream waiting on a recorded copy-free
    std::map<int, std::set<std::pair<int,int>>> pending, done_recorded;
    std::set<std::pair<int,int>> copied, visible;              // slots ever copied; copies the compute stream has waited on
    std::vector<std::string> violations;
    // ADAPT-PF: the tensors written by the latest publish, and the latest plan per layer
    int t_cpu = 0, t_pf = 0, t_gpu = 0, t_gpu_pp = 0, t_kc = 0, t_kg = 0; // stand-ins for the six tensors
    std::map<const void *, std::vector<uint8_t>> written;
    std::vector<std::vector<int32_t>> plan;
    int n_plans = 0;

    static int sentinel(int e) { return NEXP + 1 + e; }

    void wait_done(int dev) {
        for (const auto & s : done_recorded[dev]) visible.insert(s);
        done_recorded[dev].clear();
    }
    void set_maps(int l) {
        llama_moe_adapt_maps m;
        llama_moe_adapt_build_maps(m, *(*L)[l].cur, *(*L)[l].inflight, NEXP, sentinel);
        const auto & cur = *(*L)[l].cur;
        const auto & fl  = *(*L)[l].inflight;
        for (size_t g = 0; g < cur.size(); g++) {
            const bool resident = !(g < fl.size() && fl[g]);
            const std::pair<int,int> s = { l, (int) g };
            if (resident && copied.count(s) && !visible.count(s)) violations.push_back("admitted before copy-done was waited on: layer " + std::to_string(l) + " slot " + std::to_string(g));
        }
        for (int e = 0; e < NEXP; e++) {
            if ((m.kc[e] == 1.0f) == (m.kg[e] == 1.0f)) violations.push_back("expert resident on both or neither side");
            if (m.kg[e] == 1.0f && m.mc[e] != sentinel(e)) violations.push_back("GPU-resident expert not sentinelled in map_cpu");
        }
        for (size_t g = 0; g < cur.size(); g++) {
            if (g < fl.size() && fl[g] && m.kg[cur[g]] == 1.0f) violations.push_back("in-flight slot's expert marked GPU-resident");
        }
        // ADAPT-PF: publish through the shared writer, then check what it wrote against the masks
        written.clear();
        const llama_moe_adapt_map_tensors t = { &t_cpu, &t_pf, &t_gpu, &t_gpu_pp, &t_kc, &t_kg };
        bool planned = false;
        llama_moe_adapt_write_maps(m, t,
            [&](void * dst, const void * data, size_t size) { written[dst].assign((const uint8_t *) data, (const uint8_t *) data + size); },
            [&](const std::vector<int32_t> & list) { if ((size_t) l >= plan.size()) plan.resize(l + 1); plan[l] = list; planned = true; n_plans++; });
        if (written.size() != 6 || !planned) violations.push_back("publish did not write all six maps and the plan");
        std::vector<int32_t> want_list;
        for (int e = 0; e < NEXP; e++) if (m.kc[e] == 1.0f) want_list.push_back(e);
        if (planned && plan[l] != want_list) violations.push_back("plan is not the CPU-served experts of layer " + std::to_string(l));
        const auto & pfb = written[&t_pf];
        if (pfb.size() != NEXP*sizeof(int32_t)) { violations.push_back("map_cpu_pf has the wrong size"); }
        else {
            const int32_t * pf = (const int32_t *) pfb.data();
            for (int e = 0; e < NEXP; e++) {
                const auto it = std::find(want_list.begin(), want_list.end(), e);
                const int32_t want = it != want_list.end() ? (int32_t) (it - want_list.begin()) : m.mc[e];
                if (pf[e] != want) { violations.push_back("map_cpu_pf of expert " + std::to_string(e) + " disagrees with map_cpu"); break; }
            }
        }
        published[l] = m;
    }
    void begin_copies(int dev) {
        at_free[dev] = published;
        gated[dev] = true;
    }
    void copy(int l, int g, int e) {
        const int dev = (*L)[l].dev;
        if (!gated[dev]) { violations.push_back("copy issued without a copy-free wait"); return; }
        const auto & m = at_free[dev][l];
        for (int x = 0; x < NEXP; x++) {
            if (m.kg[x] == 1.0f && (m.mg[x] == g || m.mgpp[x] == g)) violations.push_back("copy into slot " + std::to_string(g) + " of layer " + std::to_string(l) + " while expert " + std::to_string(x) + " is published resident there");
        }
        (void) e;
        copied.insert({ l, g });
        visible.erase({ l, g });
        pending[dev].insert({ l, g });
    }
    void end_copies(int dev) {
        for (const auto & s : pending[dev]) done_recorded[dev].insert(s);
        pending[dev].clear();
        gated[dev] = false;
    }
};

struct world {
    std::vector<std::vector<int32_t>> cur;
    std::vector<std::vector<uint8_t>> fl;
    std::vector<std::vector<float>>   usage;
    std::vector<llama_moe_adapt_layer> L;
    fake_io io;
    world(const std::vector<int> & devs, int G) {
        const size_t n = devs.size();
        cur.resize(n); fl.resize(n); usage.assign(n, std::vector<float>(NEXP, 0.0f));
        for (size_t l = 0; l < n; l++) { for (int g = 0; g < G; g++) cur[l].push_back(g); fl[l].assign(G, 0); }
        for (size_t l = 0; l < n; l++) L.push_back({ &cur[l], &fl[l], devs[l] });
        io.L = &L; io.published.resize(n);
        for (size_t l = 0; l < n; l++) io.set_maps((int) l);
    }
    std::vector<llama_moe_swap> select(std::mt19937 & rng) {
        std::vector<const float *> up; std::vector<std::vector<uint8_t>> res(L.size(), std::vector<uint8_t>(NEXP, 0));
        std::vector<const uint8_t *> rp;
        for (size_t l = 0; l < L.size(); l++) {
            for (int e = 0; e < NEXP; e++) usage[l][e] = usage[l][e] * 0.7f + (float) (rng() % 7);
            for (int e : cur[l]) res[l][e] = 1;
            up.push_back(usage[l].data()); rp.push_back(res[l].data());
        }
        return llama_moe_adapt_select(up, rp, NEXP, 2.0f, 1.5f, 6);
    }
};

int main() {
    // (A) the real ordering over many rounds, three layers on two devices: no violation, and every round's admit
    // leaves nothing in flight before selection
    {
        world w({ 0, 0, 1 }, 16);
        std::mt19937 rng(1234);
        int total = 0;
        for (int round = 0; round < 200; round++) {
            llama_moe_adapt_admit(w.io, w.L);
            for (const auto & f : w.fl) for (uint8_t x : f) CHECK(x == 0, "slot still in flight after admit (round %d)", round);
            total += llama_moe_adapt_evict_and_copy(w.io, w.L, w.select(rng));
        }
        CHECK(total > 100, "too few swaps to exercise the ordering: %d", total);
        CHECK(w.io.violations.empty(), "real ordering: %zu violations, first: %s", w.io.violations.size(), w.io.violations.empty() ? "" : w.io.violations[0].c_str());
        CHECK(w.io.n_plans > 200, "too few plans published: %d", w.io.n_plans);
        std::printf("A: %d swaps over 200 rounds, %d plans, %zu violations\n", total, w.io.n_plans, w.io.violations.size());
    }
    // (B) checker power 1: copying before the eviction is published must be caught
    {
        world w({ 0 }, 8);
        w.io.begin_copies(0);
        w.cur[0][3] = 40; w.fl[0][3] = 1; // evict slot 3 without publishing the maps first
        w.io.copy(0, 3, 40); w.io.end_copies(0);
        CHECK(!w.io.violations.empty(), "checker missed a copy into a published-resident slot");
    }
    // (C) checker power 2: admitting without waiting on copy-done must be caught
    {
        world w({ 0 }, 8);
        w.cur[0][3] = 40; w.fl[0][3] = 1; w.io.set_maps(0);
        w.io.begin_copies(0); w.io.copy(0, 3, 40); w.io.end_copies(0);
        w.fl[0][3] = 0; w.io.set_maps(0); // admit with no wait_done
        CHECK(!w.io.violations.empty(), "checker missed an admission before copy-done");
    }
    // (D) the map builder: an in-flight slot's expert is CPU-served
    {
        std::vector<int32_t> cur = { 5, 6, 7 }; std::vector<uint8_t> fl = { 0, 1, 0 };
        llama_moe_adapt_maps m; llama_moe_adapt_build_maps(m, cur, fl, NEXP, fake_io::sentinel);
        CHECK(m.kg[6] == 0.0f && m.kc[6] == 1.0f && m.mc[6] == 6 && m.mgpp[6] == 3, "in-flight expert 6 not CPU-served");
        CHECK(m.kg[5] == 1.0f && m.mg[5] == 0 && m.mc[5] == fake_io::sentinel(5), "resident expert 5 wrong");
    }
    // (E) ADAPT-PF: without map_cpu_pf (no LLAMA_MOE_PREFETCH_PP) the writer writes the five and registers no plan
    {
        std::vector<int32_t> cur = { 5, 6, 7 }; std::vector<uint8_t> fl = { 0, 1, 0 };
        llama_moe_adapt_maps m; llama_moe_adapt_build_maps(m, cur, fl, NEXP, fake_io::sentinel);
        int a, b, c, d, e2; int n_set = 0, n_plan = 0;
        const llama_moe_adapt_map_tensors t = { &a, nullptr, &b, &c, &d, &e2 };
        llama_moe_adapt_write_maps(m, t, [&](void *, const void *, size_t) { n_set++; }, [&](const std::vector<int32_t> &) { n_plan++; });
        CHECK(n_set == 5 && n_plan == 0, "no map_cpu_pf: %d writes, %d plans", n_set, n_plan);
        // the in-flight expert 6 is CPU-served: in the plan at its rank among the host experts
        CHECK(m.pf[6] == 5 && m.pf_list.size() == NEXP - 2 && m.pf[5] == fake_io::sentinel(5), "pf of the in-flight expert");
    }
    if (n_fail) { std::fprintf(stderr, "test-moe-adapt-async: %d failures\n", n_fail); return 1; }
    std::printf("test-moe-adapt-async: OK\n");
    return 0;
}
