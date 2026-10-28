// bench-kv-cells-copy.cpp - fork-chunk-loop review F3 (28 Oct 2026, Opus 5.5): what chunk_save's deep copy of a
// llama_kv_cells table costs at the served cache size. Build: see run line in the report; CPU only.
#include "llama-kv-cells.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    const uint32_t size = argc > 1 ? atoi(argv[1]) : 262144;
    for (uint32_t used : { 20000u, 40000u, 131072u, 262144u }) {
        if (used > size) continue;
        std::vector<llama_kv_cells> v(1);
        v[0].resize(size);
        for (uint32_t i = 0; i < used; i++) { v[0].pos_set(i, (llama_pos) i); v[0].seq_add(i, (llama_seq_id) (i % 3)); }
        std::vector<llama_kv_cells> c;
        const int reps = 10;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) { c = v; }
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        printf("cells %u used %6u: one copy %.2f ms; per chunk of C=8 (two caches) %.0f ms\n", size, used, ms, ms * 8 * 2);
    }
    return 0;
}
