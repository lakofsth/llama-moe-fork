#pragma once

// fork-kernel-fusion (2026-10-06): the heat-split packer's expert SELECTION (llama-model.cpp
// llama_moe_heat_split_init), lifted out as a pure function so it can be tested without a model.
//
// Candidates are the (layer, expert) pairs of packable layers (ldev[il] >= 0), hottest first; each is taken while
// its layer holds fewer than its device's cap and its bytes fit the device's remaining budget — the first one that
// does not fit CLOSES that device (the single-device packer's `break`, per device), the others keep filling.
//
// Without `fill`, only experts with heat > 0 are candidates, so a layer whose map names fewer experts than the cap
// packs fewer slots than the budget pays for: on the deployed iq4xs map (2026-08-31), host layers 0-30 name
// 128..300 experts against a cap of 219, so a 15,144 MiB budget packed 14,672 MiB and no budget above ~16,000 MiB
// could be spent at all (pack-ab, vram-probe 2026-10-06). With `fill` (LLAMA_MOE_HEAT_FILL=1), the zero-heat
// experts follow every positive one, in (layer, expert) order, so every layer fills to its cap while the budget
// lasts and the adaptive tier (LLAMA_MOE_ADAPT) then swaps them for what the traffic routes — as Strata's tier
// starts full. The positive candidates are sorted exactly as without fill, so the fill selection is the no-fill
// selection plus slots that were empty.

#include <algorithm>
#include <cstddef>
#include <vector>

struct llama_moe_heat_pack {
    size_t budget = 0;     // bytes this device may pack
    int    cap    = 0;     // per-layer expert cap on this device
    size_t bytes  = 0;     // out: selected bytes
    int    n_sel  = 0;     // out: selected experts
    bool   full   = false; // in: closed from the start (budget 0); out: closed by a non-fit
};

// heat: [n_layer x n_expert]; ldev[il]: pack-device index of layer il, -1 = not packable; bpe[il]: bytes of one of
// layer il's experts. Returns sel[il]: the selected expert ids of each layer, in selection order.
static std::vector<std::vector<int>> llama_moe_heat_select(
        const std::vector<float> & heat, int n_layer, int n_expert,
        const std::vector<int> & ldev, const std::vector<size_t> & bpe,
        std::vector<llama_moe_heat_pack> & pd, bool fill) {
    struct cand { float h; int il; int e; };
    std::vector<cand> cands;
    std::vector<cand> zeros;
    cands.reserve(heat.size());
    for (int il = 0; il < n_layer; il++) {
        if (ldev[il] < 0) continue;
        for (int e = 0; e < n_expert; e++) {
            const float h = heat[(size_t) il*n_expert + e];
            if (h > 0.0f) {
                cands.push_back({h, il, e});
            } else if (fill) {
                zeros.push_back({h, il, e});
            }
        }
    }
    std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.h > b.h; });
    cands.insert(cands.end(), zeros.begin(), zeros.end());

    for (auto & p : pd) if (p.budget == 0) p.full = true;

    std::vector<std::vector<int>> sel(n_layer);
    int n_open = 0;
    for (const auto & p : pd) if (!p.full) n_open++;
    for (const auto & c : cands) {
        if (n_open == 0) break;
        auto & p = pd[ldev[c.il]];
        if (p.full) continue;                          // device closed — other devices keep filling
        if ((int) sel[c.il].size() >= p.cap) continue; // layer full — keep scanning, do NOT stop
        const size_t b = bpe[c.il];
        if (p.bytes + b > p.budget) {
            // the single-device packer's `break`, applied per device: this device takes nothing more.
            // Leaves at most one expert's bytes of its budget idle.
            p.full = true;
            n_open--;
            continue;
        }
        p.bytes += b;
        sel[c.il].push_back(c.e);
        p.n_sel++;
    }
    return sel;
}
