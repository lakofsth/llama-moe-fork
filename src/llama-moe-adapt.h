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
