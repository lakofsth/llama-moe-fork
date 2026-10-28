#pragma once

// fork-strata-ports: the swap selection of Strata's adaptive VRAM tier (Strata
// program/generate.cpp, the `adapt` lambda, @36fa455), as a pure function so it can be tested
// without a model and checked against eval/two-card/cache-sim.py's replay of the same rule.
//
// Per layer: candidates are NON-resident experts with usage >= min_usage, sorted by usage down;
// victims are resident experts sorted by usage up; the i-th candidate is paired with the i-th
// victim while cand >= victim + margin (the first pair that fails ends the layer). All layers'
// pairs are ranked by gain (cand - victim) and the best max_swaps are kept. Usage is in ROUTINGS
// (one per token per routed slot), decayed by the caller after each round.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

struct llama_moe_swap {
    float   gain;
    int32_t layer; // index into the caller's layer list
    int32_t in;    // expert brought into VRAM
    int32_t out;   // resident expert it replaces
};

// usage[l][e], resident[l][e] (0/1), for each of the caller's layers; n_expert per layer.
inline std::vector<llama_moe_swap> llama_moe_adapt_select(
        const std::vector<const float *> & usage,
        const std::vector<const uint8_t *> & resident,
        int n_expert, float min_usage, float margin, int max_swaps) {
    std::vector<llama_moe_swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (size_t l = 0; l < usage.size(); l++) {
        cand.clear();
        vict.clear();
        const float   * u = usage[l];
        const uint8_t * r = resident[l];
        for (int32_t e = 0; e < n_expert; e++) {
            if (!r[e]) { if (u[e] >= min_usage) cand.emplace_back(u[e], e); }
            else       vict.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        // stable sorts, ties by expert id: deterministic, and the same order numpy's stable argsort gives
        std::stable_sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        std::stable_sort(vict.begin(), vict.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        for (size_t i = 0; i < nc; i++) {
            if (cand[i].first < vict[i].first + margin) break;
            swaps.push_back({ cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second });
        }
    }
    std::stable_sort(swaps.begin(), swaps.end(), [](const llama_moe_swap & a, const llama_moe_swap & b) { return a.gain > b.gain; });
    if (max_swaps >= 0 && (int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    return swaps;
}

// fork-strata-ports ADAPT-ASYNC (tests/INVARIANTS.md): the adaptive tier's map contents and round ordering, pure so
// they can be tested without a model or a GPU. A packed slot whose copy has not been admitted (inflight[g] != 0) is
// GPU-resident in no map; its expert, and the evicted one, are served by the CPU until the next round admits it.
struct llama_moe_adapt_maps {
    std::vector<int32_t> mc, mg, mgpp; // map_cpu, map_gpu, map_gpu_pp
    std::vector<float>   kc, kg;       // mask_cpu, mask_gpu
    // LLAMA_MOE_PREFETCH_PP (ADAPT-PF): map_cpu_pf, and the prefetch plan (the host-served experts, ascending)
    std::vector<int32_t> pf, pf_list;
};

// ADAPT-PF: from map_cpu's values mc (own id = host-served, else its sentinel), map_cpu_pf (a host-served e -> its index
// in the ascending list of host-served experts; a resident one keeps its sentinel) and that list, the scheduler's plan
inline void llama_moe_adapt_pf_maps(const std::vector<int32_t> & mc, std::vector<int32_t> & pf, std::vector<int32_t> & list) {
    pf.resize(mc.size());
    list.clear();
    for (size_t e = 0; e < mc.size(); e++) {
        if (mc[e] == (int32_t) e) { pf[e] = (int32_t) list.size(); list.push_back((int32_t) e); }
        else                      { pf[e] = mc[e]; }
    }
}

// ADAPT-PF: a layer's map tensors (opaque here); map_cpu_pf is null without LLAMA_MOE_PREFETCH_PP
struct llama_moe_adapt_map_tensors {
    void * map_cpu, * map_cpu_pf, * map_gpu, * map_gpu_pp, * mask_cpu, * mask_gpu;
};

// ADAPT-PF: publish a layer's maps. set(tensor, data, bytes) writes one tensor (async on the compute stream, or
// synchronously); plan(list) registers the prefetch plan. map_cpu_pf and the plan go with map_cpu, from the same mc,
// so the compact ids a graph reads and the experts the ring uploads never disagree.
template <class Set, class Plan>
inline void llama_moe_adapt_write_maps(const llama_moe_adapt_maps & m, const llama_moe_adapt_map_tensors & t, Set && set, Plan && plan) {
    set(t.map_cpu, m.mc.data(), m.mc.size()*sizeof(int32_t));
    if (t.map_cpu_pf) {
        set(t.map_cpu_pf, m.pf.data(), m.pf.size()*sizeof(int32_t));
        plan(m.pf_list);
    }
    set(t.map_gpu,    m.mg.data(),   m.mg.size()*sizeof(int32_t));
    set(t.map_gpu_pp, m.mgpp.data(), m.mgpp.size()*sizeof(int32_t));
    set(t.mask_cpu,   m.kc.data(),   m.kc.size()*sizeof(float));
    set(t.mask_gpu,   m.kg.data(),   m.kg.size()*sizeof(float));
}

// cpu_sentinel(e): map_cpu's value for a GPU-resident expert (the fork's per-expert sentinel encoding).
template <class SentinelFn>
inline void llama_moe_adapt_build_maps(llama_moe_adapt_maps & m, const std::vector<int32_t> & cur,
        const std::vector<uint8_t> & inflight, int n_expert, SentinelFn cpu_sentinel) {
    const size_t G = cur.size();
    m.mc.resize((size_t) n_expert);
    m.mg.assign((size_t) n_expert, 0);
    m.mgpp.assign((size_t) n_expert, (int32_t) G);
    m.kc.assign((size_t) n_expert, 1.0f);
    m.kg.assign((size_t) n_expert, 0.0f);
    for (int e = 0; e < n_expert; e++) m.mc[e] = e;
    for (size_t g = 0; g < G; g++) {
        if (g < inflight.size() && inflight[g]) continue;
        const int e = cur[g];
        m.mc[e] = cpu_sentinel(e); m.mg[e] = (int32_t) g; m.mgpp[e] = (int32_t) g;
        m.kc[e] = 0.0f; m.kg[e] = 1.0f;
    }
    llama_moe_adapt_pf_maps(m.mc, m.pf, m.pf_list);
}

struct llama_moe_adapt_layer {
    std::vector<int32_t> * cur;      // slot -> expert
    std::vector<uint8_t> * inflight; // slot -> copy not yet admitted
    int dev;                         // the device holding the layer's packed tensors (caller's numbering)
};

// IO, all enqueued (none blocks the host):
//   wait_done(dev)       the compute stream of dev waits for dev's last end_copies
//   set_maps(l)          rebuild layer l's maps from cur/inflight, async on its compute stream (the five, and with
//                        LLAMA_MOE_PREFETCH_PP map_cpu_pf and the plan: llama_moe_adapt_write_maps, ADAPT-PF)
//   begin_copies(dev)    record copy-free on dev's compute stream; dev's copy stream waits on it
//   copy(l, g, e)        expert e's tensors into layer l's slot g, on the copy stream
//   end_copies(dev)      record copy-done on dev's copy stream

// Step 1 of a round: admit every in-flight slot of the previous round.
template <class IO>
inline void llama_moe_adapt_admit(IO & io, std::vector<llama_moe_adapt_layer> & L) {
    std::vector<int> devs;
    for (const auto & l : L) {
        if (std::find(l.inflight->begin(), l.inflight->end(), (uint8_t) 1) != l.inflight->end() &&
            std::find(devs.begin(), devs.end(), l.dev) == devs.end()) devs.push_back(l.dev);
    }
    for (int d : devs) io.wait_done(d);
    for (size_t k = 0; k < L.size(); k++) {
        auto & fl = *L[k].inflight;
        if (std::find(fl.begin(), fl.end(), (uint8_t) 1) == fl.end()) continue;
        std::fill(fl.begin(), fl.end(), (uint8_t) 0);
        io.set_maps((int) k);
    }
}

// Step 3 of a round: evict each swap's victim at once (its slot leaves every map), then copy the incoming experts.
// sw.layer indexes L. Returns the number of swaps applied.
template <class IO>
inline int llama_moe_adapt_evict_and_copy(IO & io, std::vector<llama_moe_adapt_layer> & L, const std::vector<llama_moe_swap> & swaps) {
    struct pend { int l, g, e; };
    std::vector<pend> todo;
    std::vector<char> touched(L.size(), 0);
    for (const auto & sw : swaps) {
        auto & cur = *L[sw.layer].cur;
        const auto it = std::find(cur.begin(), cur.end(), sw.out);
        if (it == cur.end()) continue; // the victim is resident by construction; skip rather than corrupt
        const int g = (int) (it - cur.begin());
        if ((size_t) g >= L[sw.layer].inflight->size()) L[sw.layer].inflight->resize(cur.size(), 0);
        cur[g] = sw.in;
        (*L[sw.layer].inflight)[g] = 1;
        touched[sw.layer] = 1;
        todo.push_back({ sw.layer, g, sw.in });
    }
    for (size_t k = 0; k < L.size(); k++) if (touched[k]) io.set_maps((int) k);
    std::vector<int> devs;
    for (const auto & p : todo) if (std::find(devs.begin(), devs.end(), L[p.l].dev) == devs.end()) devs.push_back(L[p.l].dev);
    for (int d : devs) {
        io.begin_copies(d);
        for (const auto & p : todo) if (L[p.l].dev == d) io.copy(p.l, p.g, p.e);
        io.end_copies(d);
    }
    return (int) todo.size();
}
