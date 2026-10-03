// test-moe-adapt-select — llama_moe_adapt_select (src/llama-moe-adapt.h), Strata's adaptive-tier
// swap rule (fork-strata-ports).
//
// No argument: hand cases with exact answers. With a CASES file (written by llm-config
// eval/two-card/adapt-select-crosscheck.py): prints each case's swaps as "case layer in out" lines,
// for that script to compare with cache-sim.py's replay of the same rule.
// CASES format: "n_cases" then per case "L E min_usage margin max_swaps", then L lines of E usage
// floats, then L lines of E 0/1 resident flags.
#include "../src/llama-moe-adapt.h"

#include <cstdio>
#include <fstream>
#include <vector>

static int fails = 0;
static void expect(const char * what, long long got, long long want) {
    printf("%s %s: %lld (want %lld)\n", got == want ? "ok  " : "FAIL", what, got, want);
    fails += got != want;
}

static std::vector<llama_moe_swap> sel(const std::vector<std::vector<float>> & u, const std::vector<std::vector<uint8_t>> & r,
                                        float mn, float mg, int cap) {
    std::vector<const float *> up; std::vector<const uint8_t *> rp;
    for (size_t l = 0; l < u.size(); l++) { up.push_back(u[l].data()); rp.push_back(r[l].data()); }
    return llama_moe_adapt_select(up, rp, (int) u[0].size(), mn, mg, cap);
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        std::ifstream f(argv[1]);
        int n; f >> n;
        for (int c = 0; c < n; c++) {
            int L, E, cap; float mn, mg; f >> L >> E >> mn >> mg >> cap;
            std::vector<std::vector<float>> u(L, std::vector<float>(E));
            std::vector<std::vector<uint8_t>> r(L, std::vector<uint8_t>(E));
            for (auto & row : u) for (auto & x : row) f >> x;
            for (auto & row : r) for (auto & x : row) { int v; f >> v; x = (uint8_t) v; }
            for (const auto & s : sel(u, r, mn, mg, cap)) printf("%d %d %d %d\n", c, s.layer, s.in, s.out);
        }
        return 0;
    }
    // 1. one layer, residents {0,1}: candidate 2 (usage 5) beats victim 0 (1) by 4 >= 1.5; candidate 3
    //    (usage 2.5) vs victim 1 (2): 2.5 < 3.5 -> the loop stops; candidate below min_usage ignored.
    {
        auto s = sel({{1, 2, 5, 2.5f}}, {{1, 1, 0, 0}}, 2.0f, 1.5f, 96);
        expect("1: swaps", (long long) s.size(), 1);
        if (s.size() == 1) { expect("1: in", s[0].in, 2); expect("1: out", s[0].out, 0); }
    }
    // 2. pairs stop at the first that fails the margin: candidates 9, 3; victims 0, 2.5 (pair 2:
    //    3 < 4) -> 1 swap. (cand falls and victim rises along the pairing, so no later pair could
    //    pass: Strata's `break` and a `continue` are the same rule.)
    {
        auto s = sel({{0, 2.5f, 9, 3}}, {{1, 1, 0, 0}}, 2.0f, 1.5f, 96);
        expect("2: swaps", (long long) s.size(), 1);
    }
    // 3. the cap keeps the best gains across layers: layer 0 gain 3, layer 1 gain 8, cap 1 -> layer 1.
    {
        auto s = sel({{0, 3}, {0, 8}}, {{1, 0}, {1, 0}}, 2.0f, 1.5f, 1);
        expect("3: swaps", (long long) s.size(), 1);
        if (s.size() == 1) expect("3: layer", s[0].layer, 1);
    }
    // 4. min_usage: a non-resident expert below it is never a candidate, however idle the victim.
    {
        auto s = sel({{0, 1.9f}}, {{1, 0}}, 2.0f, 0.0f, 96);
        expect("4: swaps", (long long) s.size(), 0);
    }
    printf("test-moe-adapt-select: %d failed\n", fails);
    return fails ? 1 : 0;
}
