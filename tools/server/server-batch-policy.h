#pragma once

// fork (fork-chunk-loop P3a): the server's batch size while slots generate, under LLAMA_PREFILL_CHUNK
// (tests/INVARIANTS.md PREFILL-CHUNK-MIXED). A layer-major chunk needs -b >= C x -ub, but a batch that carries a
// generating slot's token takes the micro-batch path and runs as long as all its prompt tokens: at -b 8192 each
// generating slot then waits ~4x the served step per token. Such a batch is held to the served -b, and so is the rest
// of a batch in which a slot reaches its prompt's tail: that slot's first token waits for it.

#include <algorithm>
#include <cstdint>
#include <cstdlib>

struct server_batch_policy {
    int32_t mixed_cap = 0; // most tokens in a batch that carries a generating token; 0 = no cap

    // LLAMA_PREFILL_CHUNK unset or < 2: no cap (today); else LLAMA_PREFILL_CHUNK_MIXED_BATCH, default 2048
    static server_batch_policy from_env() {
        server_batch_policy p;
        const char * c = getenv("LLAMA_PREFILL_CHUNK");
        if (c == nullptr || atoi(c) < 2) {
            return p;
        }
        const char * m = getenv("LLAMA_PREFILL_CHUNK_MIXED_BATCH");
        p.mixed_cap = m ? std::max(0, atoi(m)) : 2048;
        return p;
    }

    // the n_batch to fill to, given the generating tokens already in the batch
    int32_t cap(int32_t n_batch, int32_t n_generating_tokens) const {
        if (mixed_cap > 0 && n_generating_tokens > 0) {
            return std::min(n_batch, mixed_cap);
        }
        return n_batch;
    }

    // after a slot added prompt tokens: if it has at most n_ubatch + 4 left (the tail the server's checkpoints split
    // into their own batches, or none: its prompt ends in this batch), its first token waits for this batch and the
    // ones its tail needs, so the slots after it fill only to the cap
    int32_t cap_tail(int32_t n_batch, int32_t n_prompt_left, int32_t n_ubatch) const {
        if (mixed_cap > 0 && n_prompt_left <= n_ubatch + 4) {
            return std::min(n_batch, mixed_cap);
        }
        return n_batch;
    }
};
