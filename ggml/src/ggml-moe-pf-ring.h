#pragma once

// fork (fork-chunk-loop P2): slot planning for LLAMA_MOE_PREFETCH_PP under the layer-major prefill chunk loop
// (LLAMA_PREFILL_CHUNK). ggml-backend.cpp's compute_splits asks this planner, per graph, which ring slot each prefetch
// entry reads, whether that slot must be uploaded first and when; tests/test-moe-pf-ring.cpp drives it on the CPU.
//
// A chunk runs each layer group as C graphs (one per ubatch) that read the same host weights with the same plans, so
// a slot that already holds a weight's plan at the current plan generation is read again without an upload. The
// caller names the weights the next group reads (hints) so they go up as soon as the ring frees a slot.
// - A graph whose entries fit in the ring (n <= R) uses the stock layout: the chunk's m-th distinct weight in slot
//   m % R; the layer is uploaded once per chunk and the slots keep the stock prefetch's sizes.
// - A wider graph (the served gate/up/down at R = 2) keeps its R - 1 largest entries in slots 0..R-2, by size rank,
//   and streams the rest through slot R - 1 in read order, each after the previous one's read: at R = 2 the down
//   projection stays and gate and up take turns, 2 uploads a pass instead of 3, and each slot holds one kind.
// Uploads go on one copy stream and reads on one compute stream, each ordered after the other side's latest event on
// the slot, so a plan is correct when every read, in issue order, finds its slot holding its weight at the current
// generation. A slot grows (is reallocated, losing its content) only before the graph runs, and only if the graph
// does not read what it held; a plan that would need more returns empty and the caller runs the per-graph ring.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

struct ggml_moe_pf_ring {
    struct item {
        const void * w;    // the host weight
        uint64_t     gen;  // plan generation when the plan was read
        size_t       need; // slot bytes the plan needs
    };

    struct slot_state {
        const void * w    = nullptr; // what the slot holds (or is being uploaded with), nullptr = nothing usable
        uint64_t     gen  = 0;
        size_t       size = 0;       // allocated bytes
    };

    struct decision {
        std::vector<int>    slot;       // per entry
        std::vector<bool>   upload;     // per entry: upload into its slot before its read
        std::vector<int>    after;      // per entry with upload: after this entry's read, -1 = at the start of the graph
        std::vector<int>    hint_slot;  // per hint: -1 = not uploaded
        std::vector<int>    hint_after; // per hint: upload after this entry's read, -1 = at the start of the graph
        std::vector<size_t> grow;       // per slot: reallocate to this size before any upload (0 = keep)
    };

    int R = 0;
    std::vector<slot_state> s;
    uint64_t cursor = 0; // the next ring position (n <= R layout)

    explicit ggml_moe_pf_ring(int R = 0) : R(R), s(R) {}

    void invalidate() {
        for (auto & x : s) {
            x.w = nullptr;
        }
        cursor = 0;
    }

    // a slot reallocated outside plan() (the per-graph ring) keeps its size here
    void set_size(int j, size_t size) {
        s[j].size = size;
        s[j].w    = nullptr;
    }

    bool holds(int j, const item & it) const {
        return s[j].w == it.w && s[j].gen == it.gen && s[j].w != nullptr;
    }

    // entries: this graph's prefetch entries in read order; hints: what the next group reads; last: no later graph
    // reads this graph's entries before the chunk is invalidated. Empty result: use the per-graph ring.
    decision plan(const std::vector<item> & entries, const std::vector<item> & hints, bool last) {
        return (int) entries.size() <= R ? plan_fit(entries, hints, last) : plan_wide(entries, hints, last);
    }

    // entries: this graph's prefetch entries in read order (n <= R, else the result is empty); hints: what the next
    // graph reads; last: no later graph reads this graph's entries before the chunk is invalidated.
    // Slots follow the stock ring: the chunk's m-th distinct weight (in layer order) goes to slot m % R, so each slot
    // keeps holding one kind of tensor and the ring is the size the stock prefetch gives it.
    decision plan_fit(const std::vector<item> & entries, const std::vector<item> & hints, bool last) {
        const int n = (int) entries.size();
        decision d;
        d.slot.assign(n, -1);
        d.upload.assign(n, false);
        d.after.assign(n, -1);
        d.hint_slot.assign(hints.size(), -1);
        d.hint_after.assign(hints.size(), -1);
        d.grow.assign(R, 0);

        std::vector<int> read_by(R, -1); // the entry of this graph that reads slot j

        // 1. entries whose slot already holds them
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < R; ++j) {
                if (read_by[j] < 0 && holds(j, entries[i])) {
                    d.slot[i]  = j;
                    read_by[j] = i;
                    break;
                }
            }
        }

        // 2. the rest take the next ring slots this graph does not read, uploaded at the start
        for (int i = 0; i < n; ++i) {
            if (d.slot[i] >= 0) {
                continue;
            }
            int j = (int) (cursor % R);
            while (read_by[j] >= 0) {
                cursor++;
                j = (int) (cursor % R);
            }
            cursor++;
            if (s[j].size < entries[i].need) {
                d.grow[j]  = entries[i].need;
                s[j].size  = entries[i].need;
            }
            d.slot[i]   = j;
            d.upload[i] = true;
            read_by[j]  = i;
            s[j].w      = entries[i].w;
            s[j].gen    = entries[i].gen;
        }

        // 3. hints, in order, into the next ring slots. A slot this graph does not read is uploaded at the start (it may grow); one it
        //    reads, only if this graph is its last reader, right after that read (no growing). The first hint that
        //    cannot go yet stops the rest, so a later pass places them in the same slots.
        for (size_t t = 0; t < hints.size(); ++t) {
            const item & h = hints[t];
            bool done = false;
            for (int i = 0; i < n && !done; ++i) {
                done = entries[i].w == h.w && entries[i].gen == h.gen; // this graph reads it: it stays where it is
            }
            for (int j = 0; j < R && !done; ++j) {
                done = holds(j, h); // an earlier pass uploaded it
            }
            if (done) {
                continue;
            }
            const int j = (int) (cursor % R);
            if (read_by[j] < 0) {
                if (s[j].size < h.need) {
                    d.grow[j] = h.need;
                    s[j].size = h.need;
                }
                d.hint_after[t] = -1;
            } else if (last && s[j].size >= h.need && !hint_uses(d, j)) {
                d.hint_after[t] = read_by[j];
            } else {
                break;
            }
            cursor++;
            d.hint_slot[t] = j;
            s[j].w   = h.w;
            s[j].gen = h.gen;
        }
        return d;
    }

    // n > R: entries ranked by size (largest first, then read order); ranks 0..R-2 own slots 0..R-2, the rest stream
    // through slot R-1 in read order. Hints only on the last pass, by the same ranking (the first streamed one only).
    decision plan_wide(const std::vector<item> & entries, const std::vector<item> & hints, bool last) {
        const int n = (int) entries.size();
        if (R < 2) {
            return {};
        }
        decision d;
        d.slot.assign(n, -1);
        d.upload.assign(n, false);
        d.after.assign(n, -1);
        d.hint_slot.assign(hints.size(), -1);
        d.hint_after.assign(hints.size(), -1);
        d.grow.assign(R, 0);

        std::vector<slot_state> cur = s;
        std::vector<int>    last_read(R, -1);
        std::vector<bool>   keeps_old(R, false); // the graph reads what the slot held before it: no growing
        std::vector<bool>   written(R, false);
        std::vector<size_t> size(R);
        for (int j = 0; j < R; ++j) size[j] = s[j].size;

        auto same = [](const slot_state & c, const item & it) { return c.w != nullptr && c.w == it.w && c.gen == it.gen; };
        auto home = [&](const std::vector<item> & v) {
            std::vector<int> order(v.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return v[a].need > v[b].need; });
            std::vector<int> slot(v.size(), R - 1);
            for (int r = 0; r < R - 1 && r < (int) order.size(); ++r) slot[order[r]] = r;
            return slot;
        };
        // put item it into slot j after the slot's last read in this graph; false if it cannot fit there
        auto place = [&](int j, const item & it, int & after) {
            if (size[j] < it.need) {
                if (keeps_old[j]) return false;
                size[j] = it.need;
            }
            after       = last_read[j];
            cur[j]      = { it.w, it.gen, size[j] };
            written[j]  = true;
            return true;
        };

        const std::vector<int> eslot = home(entries);
        for (int i = 0; i < n; ++i) {
            const int j = eslot[i];
            if (same(cur[j], entries[i])) {
                if (!written[j]) keeps_old[j] = true;
            } else {
                if (!place(j, entries[i], d.after[i])) {
                    return {};
                }
                d.upload[i] = true;
            }
            d.slot[i]    = j;
            last_read[j] = i;
        }

        if (last && !hints.empty()) {
            const std::vector<int> hslot = home(hints);
            std::vector<bool> taken(R, false);
            for (size_t t = 0; t < hints.size(); ++t) {
                const int j = hslot[t];
                if (taken[j]) continue; // the streaming slot takes the first streamed hint only
                taken[j] = true;
                if (same(cur[j], hints[t])) continue;
                int after = -1;
                if (place(j, hints[t], after)) {
                    d.hint_slot[t]  = j;
                    d.hint_after[t] = after;
                }
            }
        }

        for (int j = 0; j < R; ++j) {
            if (size[j] > s[j].size) d.grow[j] = size[j];
            s[j] = cur[j];
            s[j].size = size[j];
        }
        return d;
    }

private:
    static bool hint_uses(const decision & d, int j) {
        for (int x : d.hint_slot) {
            if (x == j) {
                return true;
            }
        }
        return false;
    }
};
