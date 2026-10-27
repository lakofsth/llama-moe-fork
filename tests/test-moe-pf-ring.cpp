// PREFILL-CHUNK-PF (tests/INVARIANTS.md): the chunk ring planner (ggml/src/ggml-moe-pf-ring.h) that compute_splits uses
// under LLAMA_PREFILL_CHUNK with LLAMA_MOE_PREFETCH_PP. The scheduler's order is replayed on the CPU: per graph, the
// slots the plan grows (content lost), the uploads at the start, then each entry's read followed by the hint uploads
// planned after it; one copy stream and one compute stream, each waiting on the other's latest event on a slot, make
// issue order the order that matters. Checked over chunked schedules (G=1 and G=4, R in 1..4, short last ubatches,
// plan changes between and inside chunks, per-graph fallback between chunks):
//   every read finds its slot holding its weight at the current plan generation;
//   with G=1 and R >= the entries of a layer, each host layer is uploaded once per chunk (per plan generation);
//   with R >= 2x a layer's entries, the next layer's upload is issued at the start of the current layer's first pass;
//   with G=1 and R a multiple of a layer's entries, the ring is the size the stock prefetch gives it (every host entry in
//   layer order, slot m % R); always at most R slots of the largest entry.
#include "../ggml/src/ggml-moe-pf-ring.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct weight { int id; size_t need; };

struct sim {
    ggml_moe_pf_ring ring;
    int R;
    std::vector<std::pair<int, uint64_t>> content; // per slot: (weight id, gen) last uploaded; (-1, 0) = garbage
    std::vector<size_t> size;
    long uploads = 0, reads = 0, bad = 0, early = 0;
    std::string first_bad;

    explicit sim(int R) : ring(R), R(R), content(R, { -1, 0 }), size(R, 0) {}

    void upload(int j, int w, uint64_t gen) { content[j] = { w, gen }; uploads++; }
    void read(int j, int w, uint64_t gen, const std::string & where) {
        reads++;
        if (content[j].first != w || content[j].second != gen) {
            if (bad++ == 0) first_bad = where;
        }
    }

    // one graph, as compute_splits runs it; chunk = the scheduler's chunk flag
    void graph(const std::vector<weight> & ents, const std::vector<weight> & hints, bool last, bool chunk, uint64_t gen,
               const std::string & where) {
        if (chunk && (int) ents.size() <= R) {
            std::vector<ggml_moe_pf_ring::item> ie, ih;
            for (auto & e : ents)  ie.push_back({ (const void *)(intptr_t)(e.id + 1), gen, e.need });
            for (auto & h : hints) ih.push_back({ (const void *)(intptr_t)(h.id + 1), gen, h.need });
            auto d = ring.plan(ie, ih, last);
            for (int j = 0; j < R; j++) if (d.grow[j] > size[j]) { size[j] = d.grow[j]; content[j] = { -1, 0 }; }
            for (size_t i = 0; i < ents.size(); i++) if (d.upload[i]) upload(d.slot[i], ents[i].id, gen);
            for (size_t t = 0; t < hints.size(); t++) if (d.hint_slot[t] >= 0 && d.hint_after[t] < 0) { upload(d.hint_slot[t], hints[t].id, gen); early++; }
            for (size_t i = 0; i < ents.size(); i++) {
                if (size[d.slot[i]] < ents[i].need && bad++ == 0) first_bad = where + " slot too small";
                read(d.slot[i], ents[i].id, gen, where + " entry " + std::to_string(i));
                for (size_t t = 0; t < hints.size(); t++) if (d.hint_slot[t] >= 0 && d.hint_after[t] == (int) i) upload(d.hint_slot[t], hints[t].id, gen);
            }
            return;
        }
        // the per-graph ring
        ring.invalidate();
        for (int j = 0; j < R; j++) {
            size_t need = 0;
            for (size_t i = j; i < ents.size(); i += R) need = std::max(need, ents[i].need);
            if (need > size[j]) { size[j] = need; content[j] = { -1, 0 }; ring.set_size(j, need); }
        }
        const int n = (int) ents.size();
        for (int i = 0; i < n && i < R; i++) upload(i % R, ents[i].id, gen);
        for (int i = 0; i < n; i++) {
            read(i % R, ents[i].id, gen, where + " entry " + std::to_string(i));
            if (i + R < n) upload((i + R) % R, ents[i + R].id, gen);
        }
    }
};

struct model {
    int n_layer, n_host, E; // E = entries per host layer
    std::vector<std::vector<weight>> w; // per layer
    model(int n_layer, int n_host, int E) : n_layer(n_layer), n_host(n_host), E(E), w(n_layer) {
        for (int il = 0; il < n_host; il++)
            for (int e = 0; e < E; e++)
                // the gate_up entry is twice the down entry; sizes vary a little per layer, as the plans do
                w[il].push_back({ il*E + e, (size_t) ((e + 1 < E ? 2000 : 1000) + 7*((il*13 + e*5) % 11)) });
    }
};

// one prompt: n_ub ubatches (the last `short_last` tokens-short: no compact ids), chunks of C, groups of G
struct result { long uploads, reads, bad, early; size_t ring_bytes, per_graph_bytes; std::string first_bad; };

static result run(const model & m, int R, int C, int G, int n_ub, bool short_last, int gen_bump_at_pass, bool chunk_on) {
    sim s(R);
    uint64_t gen = 1;
    int pass = 0;
    size_t per_graph_need[64] = { 0 };
    for (int c0 = 0; c0 < n_ub; c0 += C) {
        const int n_chunk = std::min(C, n_ub - c0);
        const int group = n_chunk > 1 ? G : m.n_layer;
        s.ring.invalidate(); // ggml_backend_sched_moe_pf_chunk(true)
        for (int il0 = 0; il0 < m.n_layer; il0 += group) {
            const int il1 = std::min(il0 + group, m.n_layer);
            for (int k = 0; k < n_chunk; k++) {
                const bool wide      = !(short_last && c0 + k == n_ub - 1);
                const bool next_wide = !(short_last && c0 == n_ub - 1);
                std::vector<weight> ents, hints;
                if (wide) for (int il = il0; il < il1; il++) for (auto & x : m.w[il]) ents.push_back(x);
                if (il1 < m.n_layer && next_wide) for (int il = il1; il < std::min(il1 + group, m.n_layer); il++) for (auto & x : m.w[il]) hints.push_back(x);
                if (pass++ == gen_bump_at_pass) gen++; // a repin or an adaptive swap: every plan re-registered
                s.graph(ents, hints, k + 1 == n_chunk, chunk_on, gen,
                        "chunk " + std::to_string(c0 / C) + " layers " + std::to_string(il0) + ".." + std::to_string(il1) + " k " + std::to_string(k));
            }
        }
        s.ring.invalidate(); // ggml_backend_sched_moe_pf_chunk(false)
    }
    // what the stock prefetch allocates for this model: one graph of every host entry in layer order, entry m in slot m % R
    {
        int mm = 0;
        for (int il = 0; il < m.n_layer; il++) for (auto & x : m.w[il]) { per_graph_need[mm % R] = std::max(per_graph_need[mm % R], x.need); mm++; }
    }
    result r{ s.uploads, s.reads, s.bad, s.early, 0, 0, s.first_bad };
    for (int j = 0; j < R; j++) { r.ring_bytes += s.size[j]; r.per_graph_bytes += per_graph_need[j]; }
    return r;
}

int main() {
    int n_fail = 0, n_check = 0;
    auto check = [&](bool ok, const std::string & what) {
        n_check++;
        if (!ok) { n_fail++; printf("FAIL %s\n", what.c_str()); }
    };

    for (int E : { 2, 3 }) {
        const model m(12, 7, E); // 7 host layers of 12 (the served model: 31 of 48)
        for (int R = 1; R <= 6; R++) {
            for (int C : { 2, 4, 8 }) {
                for (int G : { 1, 4 }) {
                    for (int short_last : { 0, 1 }) {
                        for (int bump : { -1, 5, 17 }) {
                            const int n_ub = 13;
                            const result r = run(m, R, C, G, n_ub, short_last, bump, true);
                            char tag[128];
                            snprintf(tag, sizeof(tag), "E=%d R=%d C=%d G=%d short=%d bump=%d", E, R, C, G, short_last, bump);
                            check(r.bad == 0, std::string(tag) + ": a read found the wrong content (" + r.first_bad + ")");
                            // a ring of whole layers (R a multiple of E) keeps one kind of tensor per slot: the stock size;
                            // otherwise, or after a plan change inside a chunk, at most R of the largest entry
                            size_t max_need = 0;
                            for (auto & lw : m.w) for (auto & x : lw) max_need = std::max(max_need, x.need);
                            const size_t bound = (R % E == 0 && bump < 0 && G == 1) ? r.per_graph_bytes : (size_t) R*max_need;
                            check(r.ring_bytes <= bound, std::string(tag) + ": ring grew to " + std::to_string(r.ring_bytes) +
                                  " > " + std::to_string(bound));
                            if (G == 1 && R >= E && bump < 0) {
                                // each host layer once per chunk, at each wide chunk; the short last ubatch reads nothing
                                const int n_chunks = (n_ub + C - 1) / C;
                                const int wide_chunks = n_chunks - ((short_last && (n_ub - 1) % C == 0) ? 1 : 0);
                                check(r.uploads == (long) wide_chunks * m.n_host * E, std::string(tag) + ": " + std::to_string(r.uploads) +
                                      " uploads, expected " + std::to_string((long) wide_chunks * m.n_host * E));
                            }
                            if (G == 1 && R >= 2*E && bump < 0) {
                                check(r.early > 0, std::string(tag) + ": no next-layer upload at a graph's start");
                            }
                            if (R < E) {
                                // the per-graph ring: every read was uploaded for its own graph
                                check(r.uploads == r.reads, std::string(tag) + ": per-graph ring uploads " + std::to_string(r.uploads) +
                                      " != reads " + std::to_string(r.reads));
                            }
                        }
                    }
                }
            }
        }
        // chunk mode off: the per-graph ring for every graph (the stock prefetch)
        const result r = run(m, 2, 4, 1, 13, false, -1, false);
        check(r.bad == 0 && r.uploads == r.reads, "chunk off: per-graph ring");
    }

    printf("%d of %d checks failed\n", n_fail, n_check);
    return n_fail == 0 ? 0 : 1;
}
