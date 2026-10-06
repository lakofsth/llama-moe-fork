// test-moe-heat-select — llama_moe_heat_select (src/llama-moe-heat-select.h), the heat-split packer's expert
// selection (fork-kernel-fusion, invariant H1 in tests/INVARIANTS.md).
#include "../src/llama-moe-heat-select.h"

#include <cstdio>
#include <set>
#include <vector>

static int fails = 0;
static void expect(const char * what, long long got, long long want) {
    printf("%s %s: %lld (want %lld)\n", got == want ? "ok  " : "FAIL", what, got, want);
    fails += got != want;
}

int main() {
    // two packable layers of 8 experts, one device, cap 5. Layer 0's map names 3 experts, layer 1's names 6.
    const int L = 2, E = 8;
    std::vector<float> heat(L*E, 0.0f);
    heat[0*E + 1] = 9; heat[0*E + 4] = 7; heat[0*E + 6] = 5;
    for (int e = 0; e < 6; e++) heat[1*E + e] = 1.0f + e;
    const std::vector<int>    ldev = {0, 0};
    const std::vector<size_t> bpe  = {10, 10};

    // 1. without fill: only heat > 0 is selectable, so layer 0 packs 3 of its 5 slots
    {
        std::vector<llama_moe_heat_pack> pd(1); pd[0].budget = 1000; pd[0].cap = 5;
        auto sel = llama_moe_heat_select(heat, L, E, ldev, bpe, pd, false);
        expect("no fill: layer 0 slots", (long long) sel[0].size(), 3);
        expect("no fill: layer 1 slots", (long long) sel[1].size(), 5);
        expect("no fill: bytes", (long long) pd[0].bytes, 80);
    }
    // 2. with fill: every layer reaches its cap, and the no-fill selection is kept whole
    {
        std::vector<llama_moe_heat_pack> pd0(1); pd0[0].budget = 1000; pd0[0].cap = 5;
        auto s0 = llama_moe_heat_select(heat, L, E, ldev, bpe, pd0, false);
        std::vector<llama_moe_heat_pack> pd(1); pd[0].budget = 1000; pd[0].cap = 5;
        auto sel = llama_moe_heat_select(heat, L, E, ldev, bpe, pd, true);
        expect("fill: layer 0 slots", (long long) sel[0].size(), 5);
        expect("fill: layer 1 slots", (long long) sel[1].size(), 5);
        expect("fill: bytes", (long long) pd[0].bytes, 100);
        for (int il = 0; il < L; il++) {
            std::set<int> s(sel[il].begin(), sel[il].end());
            int kept = 0; for (int e : s0[il]) kept += s.count(e);
            expect(il == 0 ? "fill keeps layer 0's no-fill selection" : "fill keeps layer 1's no-fill selection",
                   kept, (long long) s0[il].size());
        }
        // the positive-heat experts come first: layer 0's three named experts are its first three picks
        expect("fill: layer 0 first pick is the hottest", sel[0][0], 1);
    }
    // 3. the budget still binds under fill: 7 experts' bytes, so the device closes after 7
    {
        std::vector<llama_moe_heat_pack> pd(1); pd[0].budget = 75; pd[0].cap = 5;
        auto sel = llama_moe_heat_select(heat, L, E, ldev, bpe, pd, true);
        expect("fill under budget: experts", (long long) (sel[0].size() + sel[1].size()), 7);
        expect("fill under budget: device closed", pd[0].full, 1);
    }
    // 4. an unpackable layer (ldev -1) is never selected, fill or not
    {
        std::vector<int> ld = {-1, 0};
        std::vector<llama_moe_heat_pack> pd(1); pd[0].budget = 1000; pd[0].cap = 5;
        auto sel = llama_moe_heat_select(heat, L, E, ld, bpe, pd, true);
        expect("fill: unpackable layer slots", (long long) sel[0].size(), 0);
    }
    printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
