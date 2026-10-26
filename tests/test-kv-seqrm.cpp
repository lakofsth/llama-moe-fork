// KV-SEQRM (tests/INVARIANTS.md): llama_kv_cells::seq_rm_range, which llama_kv_cache::seq_rm uses for one sequence,
// leaves the cells and returns the head that a scan of every cell does. Random caches built with every mutator the
// cache uses on cells (pos_set, seq_add, pos_add, rm, seq_keep, duplicate positions in one sequence), then both
// strategies on copies, compared cell by cell.
#include "../src/llama-kv-cells.h"

#include <cstdint>
#include <cstdio>
#include <limits>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t xr(uint32_t n) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng % n); }

static const int N_SEQ = 4;

// the loop llama_kv_cache::seq_rm ran before KV-SEQRM
static uint32_t scan_rm(llama_kv_cells & cells, llama_seq_id s, llama_pos p0, llama_pos p1) {
    uint32_t new_head = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }
        if (cells.seq_has(i, s) && cells.seq_rm(i, s)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }
    return new_head;
}

static bool same(const llama_kv_cells & a, const llama_kv_cells & b) {
    if (a.get_used() != b.get_used()) return false;
    if (a.get_used() && (a.used_min() != b.used_min() || a.used_max_p1() != b.used_max_p1())) return false;
    for (uint32_t i = 0; i < a.size(); ++i) {
        if (a.is_empty(i) != b.is_empty(i)) return false;
        if (a.is_empty(i)) continue;
        if (a.pos_get(i) != b.pos_get(i)) return false;
        for (int s = 0; s < N_SEQ; ++s) if (a.seq_has(i, s) != b.seq_has(i, s)) return false;
    }
    for (int s = 0; s < N_SEQ; ++s) {
        if (a.seq_pos_min(s) != b.seq_pos_min(s) || a.seq_pos_max(s) != b.seq_pos_max(s)) return false;
    }
    return true;
}

int main() {
    long checks = 0, bad = 0, removed = 0;
    for (int t = 0; t < 400; ++t) {
        llama_kv_cells cells;
        const uint32_t n = 64 + xr(512);
        cells.resize(n);
        for (int op = 0; op < 2000; ++op) {
            const uint32_t i = xr(n);
            switch (xr(10)) {
                case 0: case 1: case 2: case 3:  // fill an empty cell; positions repeat within a sequence
                    if (cells.is_empty(i)) {
                        cells.pos_set(i, (llama_pos)xr(300));
                        cells.seq_add(i, (llama_seq_id)xr(N_SEQ));
                    }
                    break;
                case 4:  // a second sequence shares the cell
                    if (!cells.is_empty(i)) { const llama_seq_id s = xr(N_SEQ); if (!cells.seq_has(i, s)) cells.seq_add(i, s); }
                    break;
                case 5:  // shift; a negative result empties the cell
                    if (!cells.is_empty(i)) cells.pos_add(i, (llama_pos)xr(40) - 20);
                    break;
                case 6:
                    if (!cells.is_empty(i)) cells.rm(i);
                    break;
                case 7:
                    if (!cells.is_empty(i)) cells.seq_keep(i, (llama_seq_id)xr(N_SEQ));
                    break;
                default: {  // compare the two strategies on copies of this state
                    const llama_seq_id s = xr(N_SEQ);
                    llama_pos p0 = (llama_pos)xr(320), p1 = p0 + (llama_pos)xr(60);
                    if (xr(4) == 0) p1 = std::numeric_limits<llama_pos>::max();  // seq_rm(s, p0, -1)
                    if (xr(8) == 0) p0 = 0;
                    llama_kv_cells a = cells, b = cells;
                    const uint32_t used0 = a.get_used();
                    const uint32_t ha = a.seq_rm_range(s, p0, p1);
                    const uint32_t hb = scan_rm(b, s, p0, p1);
                    ++checks;
                    removed += used0 - a.get_used();
                    if (ha != hb || !same(a, b)) {
                        if (bad++ < 5) fprintf(stderr, "t=%d op=%d seq %d [%d,%d): head %u vs scan %u\n", t, op, s, p0, p1, ha, hb);
                    }
                    if (xr(3) == 0) cells = a;  // carry the removal forward sometimes
                }
            }
        }
    }
    printf("test-kv-seqrm: %ld comparisons (%ld cells emptied), %ld differ from the scan\n", checks, removed, bad);
    return bad != 0 || removed == 0;
}
