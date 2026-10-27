// PREFILL-CHUNK-PF (tests/INVARIANTS.md): the chunk ring planner (ggml/src/ggml-moe-pf-ring.h) that compute_splits uses
// under LLAMA_PREFILL_CHUNK with LLAMA_MOE_PREFETCH_PP. The scheduler's order is replayed on the CPU: per graph, the
// slots the plan grows (content lost), the uploads at the start, then each entry's read followed by the hint uploads
// planned after it; one copy stream and one compute stream, each waiting on the other's latest event on a slot, make
// issue order the order that matters. Checked over chunked schedules (G=1 and G=4, R in 1..4, short last ubatches,
// plan changes between and inside chunks, per-graph fallback between chunks):
//   every read finds its slot holding its weight at the current plan generation;
//   with G=1 and R >= the entries of a layer, each host layer is uploaded once per chunk (per plan generation);
//   with 2 <= R < a layer's entries (the served gate/up/down at R = 2), slots are still reused across passes;
//   with G=1 and R >= 2 a chunk graph is always planned (never handed to the per-graph ring);
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
    long uploads = 0, reads = 0, bad = 0, early = 0, fallback = 0;
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
        if (chunk) {
            std::vector<ggml_moe_pf_ring::item> ie, ih;
            for (auto & e : ents)  ie.push_back({ (const void *)(intptr_t)(e.id + 1), gen, e.need });
            for (auto & h : hints) ih.push_back({ (const void *)(intptr_t)(h.id + 1), gen, h.need });
            auto d = ring.plan(ie, ih, last);
            if (d.grow.size() == (size_t) R) {
                for (int j = 0; j < R; j++) if (d.grow[j] > size[j]) { size[j] = d.grow[j]; content[j] = { -1, 0 }; }
                // at the start: the entries uploaded before any read, then the hints into slots this graph does not read
                for (size_t i = 0; i < ents.size(); i++) if (d.upload[i] && d.after[i] < 0) upload(d.slot[i], ents[i].id, gen);
                for (size_t t = 0; t < hints.size(); t++) if (d.hint_slot[t] >= 0 && d.hint_after[t] < 0) { upload(d.hint_slot[t], hints[t].id, gen); early++; }
                for (size_t i = 0; i < ents.size(); i++) {
                    if (size[d.slot[i]] < ents[i].need && bad++ == 0) first_bad = where + " slot too small";
                    read(d.slot[i], ents[i].id, gen, where + " entry " + std::to_string(i));
                    // after entry i's read: the entries planned after it, then the hints
                    for (size_t k = 0; k < ents.size(); k++) if (d.upload[k] && d.after[k] == (int) i) upload(d.slot[k], ents[k].id, gen);
                    for (size_t t = 0; t < hints.size(); t++) if (d.hint_slot[t] >= 0 && d.hint_after[t] == (int) i) upload(d.hint_slot[t], hints[t].id, gen);
                }
                return;
            }
            fallback++;
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
        // a layer's entries share its non-resident count (one plan per layer), which varies across layers; per expert,
        // E = 2 is merged gate_up (2x) and down, E = 3 is the served up, gate (IQ3_S, 0.671 MiB) and down (IQ4_NL, 0.879)
        for (int il = 0; il < n_host; il++) {
            const size_t n_nonres = 200 + 37*((il*13) % 11);
            for (int e = 0; e < E; e++) {
                const size_t per = E == 2 ? (e == 0 ? 2000 : 1000) : (e < 2 ? 671 : 879);
                w[il].push_back({ il*E + e, n_nonres*per/100 + 512 });
            }
        }
    }
};

// one prompt: n_ub ubatches (the last `short_last` tokens-short: no compact ids), chunks of C, groups of G
struct result { long uploads, reads, bad, early, fallback; size_t ring_bytes, per_graph_bytes; std::string first_bad; std::vector<size_t> per_slot; };

// decodes: the ubatches of each llama_decode in turn (one scheduler, so one ring, across them); short_last: the last
// decode's last ubatch is narrower than LLAMA_MOE_PREFETCH_PP_MIN
static result run(const model & m, int R, int C, int G, const std::vector<int> & decodes, bool short_last_any, int gen_bump_at_pass, bool chunk_on) {
    sim s(R);
    uint64_t gen = 1;
    int pass = 0;
    size_t per_graph_need[64] = { 0 };
    for (size_t dec = 0; dec < decodes.size(); dec++) {
    const int  n_ub       = decodes[dec];
    const bool short_last = short_last_any && dec + 1 == decodes.size();
    for (int c0 = 0; c0 < n_ub; c0 += C) {
        const int n_chunk = std::min(C, n_ub - c0);
        const int group = n_chunk > 1 ? G : m.n_layer;
        // llama_context turns chunk mode on only for one-layer passes of a chunk of several ubatches
        const bool chunk = chunk_on && group == 1;
        s.ring.invalidate(); // ggml_backend_sched_moe_pf_chunk(true)
        for (int il0 = 0; il0 < m.n_layer; il0 += group) {
            const int il1 = std::min(il0 + group, m.n_layer);
            for (int k = 0; k < n_chunk; k++) {
                const bool wide      = !(short_last && c0 + k == n_ub - 1);
                const bool next_wide = !(short_last && c0 == n_ub - 1);
                std::vector<weight> ents, hints;
                if (wide) for (int il = il0; il < il1; il++) for (auto & x : m.w[il]) ents.push_back(x);
                if (chunk && il1 < m.n_layer && next_wide) for (int il = il1; il < std::min(il1 + group, m.n_layer); il++) for (auto & x : m.w[il]) hints.push_back(x);
                if (pass++ == gen_bump_at_pass) gen++; // a repin or an adaptive swap: every plan re-registered
                s.graph(ents, hints, k + 1 == n_chunk, chunk, gen,
                        "chunk " + std::to_string(c0 / C) + " layers " + std::to_string(il0) + ".." + std::to_string(il1) + " k " + std::to_string(k));
            }
        }
        s.ring.invalidate(); // ggml_backend_sched_moe_pf_chunk(false)
    }
    }
    // what the stock prefetch allocates for this model: one graph of every host entry in layer order, entry m in slot m % R
    {
        int mm = 0;
        for (int il = 0; il < m.n_layer; il++) for (auto & x : m.w[il]) { per_graph_need[mm % R] = std::max(per_graph_need[mm % R], x.need); mm++; }
    }
    result r{ s.uploads, s.reads, s.bad, s.early, s.fallback, 0, 0, s.first_bad };
    for (int j = 0; j < R; j++) { r.ring_bytes += s.size[j]; r.per_graph_bytes += per_graph_need[j]; r.per_slot.push_back(per_graph_need[j]); }
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
                            const result r = run(m, R, C, G, { n_ub }, short_last, bump, true);
                            char tag[128];
                            snprintf(tag, sizeof(tag), "E=%d R=%d C=%d G=%d short=%d bump=%d", E, R, C, G, short_last, bump);
                            check(r.bad == 0, std::string(tag) + ": a read found the wrong content (" + r.first_bad + ")");
                            // a ring of whole layers (R a multiple of E) keeps one kind of tensor per slot: the stock size;
                            // otherwise, or after a plan change inside a chunk, at most R of the largest entry
                            size_t max_need = 0;
                            for (auto & lw : m.w) for (auto & x : lw) max_need = std::max(max_need, x.need);
                            size_t bound = (R % E == 0 && bump < 0 && G == 1) ? r.per_graph_bytes : (size_t) R*max_need;
                            if (G == 1 && R >= 2 && R < E && bump < 0) {
                                // the wide layout: one slot per size rank 0..R-2, one streaming slot for the rest
                                std::vector<size_t> rank_max(R, 0);
                                for (int il = 0; il < m.n_host; il++) {
                                    std::vector<size_t> v; for (auto & x : m.w[il]) v.push_back(x.need);
                                    std::vector<int> ord(v.size()); for (size_t q = 0; q < ord.size(); q++) ord[q] = (int) q;
                                    std::stable_sort(ord.begin(), ord.end(), [&](int a2, int b2) { return v[a2] > v[b2]; });
                                    for (size_t q = 0; q < ord.size(); q++) {
                                        const int slot = q < (size_t) (R - 1) ? (int) q : R - 1;
                                        rank_max[slot] = std::max(rank_max[slot], v[ord[q]]);
                                    }
                                }
                                // a one-ubatch chunk runs the stock whole-model ring in the same slots, which never shrink
                                bound = 0; for (int j = 0; j < R; j++) bound += std::max(rank_max[j], r.per_slot[j]);
                            }
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
                            // never more uploads than the stock prefetch (one per read); a chunk graph is planned, not
                            // handed to the per-graph ring, unless the ring is one slot
                            if (R >= 2) {
                                check(r.uploads <= r.reads, std::string(tag) + ": uploads " + std::to_string(r.uploads) + " > reads " + std::to_string(r.reads));
                            }
                            if (R >= 2 && bump < 0 && G == 1) {
                                check(r.fallback == 0, std::string(tag) + ": " + std::to_string(r.fallback) + " chunk graphs fell back to the per-graph ring");
                            }
                            if (G == 1 && R >= 2 && R < E && bump < 0) {
                                // fewer slots than a layer's entries: the ring still keeps R - 1 of them across passes, so a
                                // layer costs at most E + (E - R + 1) per further pass, fewer than its E per pass
                                const long per_layer_max = (long) E + (long) (C - 1) * (E - R + 1);
                                const int n_chunks = (n_ub + C - 1) / C;
                                check(r.uploads <= (long) n_chunks * m.n_host * per_layer_max, std::string(tag) + ": " + std::to_string(r.uploads) +
                                      " uploads > " + std::to_string((long) n_chunks * m.n_host * per_layer_max));
                                check(r.uploads < r.reads, std::string(tag) + ": no slot reuse (uploads " + std::to_string(r.uploads) + " = reads)");
                            }
                        }
                    }
                }
            }
        }
        if (E == 3) {
            // the served shape: separate gate, up, down per host layer (blk.N.ffn_{gate,up,down}_exps), ring of 2
            // (LLAMA_MOE_PREFETCH_PP=2), C = 8, G = 1. The P2 gate of 27 Oct found this shape uploading every entry
            // on every pass (1767 uploads = 1767 reads): the first planner took a pass only if all its entries fit.
            const result rs = run(m, 2, 8, 1, { 16 }, false, -1, true);
            check(rs.bad == 0 && rs.fallback == 0 && rs.uploads * 10 <= rs.reads * 8,
                  "served shape E=3 R=2 C=8 G=1: uploads " + std::to_string(rs.uploads) + " of " + std::to_string(rs.reads) +
                  " reads, " + std::to_string(rs.fallback) + " fallbacks (want <= 80%, none)");
        }
        // chunk mode off: the per-graph ring for every graph (the stock prefetch)
        const result r = run(m, 2, 4, 1, { 13 }, false, -1, false);
        check(r.bad == 0 && r.uploads == r.reads, "chunk off: per-graph ring");
    }

    {
        // what the P2 gate should report for the served model (48 layers, 31 host, gate/up/down, R = 2, C = 8, G = 1)
        // on its 20,000-token prompt at -b 8192 -ub 1024: decodes of 8, 8 and 4 ubatches, the last ubatch 544 tokens
        const model served(48, 31, 3);
        for (int R : { 2, 3 }) {
            for (int C : { 8, 4 }) {
                const result r = run(served, R, C, 1, { 8, 8, 4 }, true, -1, true);
                check(r.bad == 0 && r.reads == 1767, "served prediction: a wrong read, or not the gate's 1767 slot reads");
                check(r.uploads < r.reads, "served prediction: no slot reuse");
                printf("served prediction (C=%d G=1 R=%d): %ld uploads for %ld slot reads (the stock prefetch uploads %ld)\n", C, R, r.uploads, r.reads, r.reads);
            }
        }
    }

    printf("%d of %d checks failed\n", n_fail, n_check);
    return n_fail == 0 ? 0 : 1;
}
