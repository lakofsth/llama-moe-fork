#pragma once

// fork (fork-chunk-loop P2): slot planning for LLAMA_MOE_PREFETCH_PP under the layer-major prefill chunk loop
// (LLAMA_PREFILL_CHUNK). ggml-backend.cpp's compute_splits asks this planner, per graph, which ring slot each prefetch
// entry reads and whether that slot must be uploaded first; tests/test-moe-pf-ring.cpp drives it on the CPU.
//
// A chunk runs each layer group as C graphs (one per ubatch) that read the same host weights with the same plans, so
// a slot that already holds a weight's plan at the current plan generation is read again without an upload. The
// caller names the weights the next graph will read (hints); they are uploaded into slots this graph does not read
// at its start, and, when this graph is the last one to read its weights (last), also into this graph's own slots
// right after the entry that reads each one. Uploads go on one copy stream and reads on one compute stream, each
// ordered after the other side's latest event on the slot, so a plan is correct when every read, in issue order,
// finds its slot holding its weight at the current generation.
//
// Only a graph whose entries fit in the ring (n <= R) is planned this way; otherwise the caller keeps the per-graph
// ring (entry i in slot i % R, every entry uploaded) and calls invalidate().

#include <cstddef>
#include <cstdint>
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
        std::vector<bool>   upload;     // per entry: upload at the start of the graph
        std::vector<int>    hint_slot;  // per hint: -1 = not uploaded
        std::vector<int>    hint_after; // per hint: upload after this entry's read, -1 = at the start of the graph
        std::vector<size_t> grow;       // per slot: reallocate to this size before any upload (0 = keep)
    };

    int R = 0;
    std::vector<slot_state> s;
    uint64_t cursor = 0; // the next ring position

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

    // entries: this graph's prefetch entries in read order (n <= R, else the result is empty); hints: what the next
    // graph reads; last: no later graph reads this graph's entries before the chunk is invalidated.
    // Slots follow the stock ring: the chunk's m-th distinct weight (in layer order) goes to slot m % R, so each slot
    // keeps holding one kind of tensor and the ring is the size the stock prefetch gives it.
    decision plan(const std::vector<item> & entries, const std::vector<item> & hints, bool last) {
        const int n = (int) entries.size();
        if (n > R) {
            return {};
        }
        decision d;
        d.slot.assign(n, -1);
        d.upload.assign(n, false);
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

        // 3. hints, in order, into the next ring slots (none if they cannot all be in the ring: the next graph then
        //    takes the per-graph ring). A slot this graph does not read is uploaded at the start (it may grow); one it
        //    reads, only if this graph is its last reader, right after that read (no growing). The first hint that
        //    cannot go yet stops the rest, so a later pass places them in the same slots.
        if ((int) hints.size() > R) {
            return d;
        }
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
