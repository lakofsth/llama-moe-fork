#pragma once

// fork (fork-chunk-loop P3a, PREFILL-CHUNK-PF-ONLY): whether build_moe_ffn reads a split layer's host experts through
// the LLAMA_MOE_PREFETCH_PP ring (compact ids) for this graph.

#include <cstdint>
#include <cstdlib>

struct llama_moe_prefetch_policy {
    int  ring       = 0;     // LLAMA_MOE_PREFETCH_PP (0 = off)
    int  min_tokens = 1024;  // LLAMA_MOE_PREFETCH_PP_MIN, never below the 32-row op_offload bound
    bool chunk_only = false; // LLAMA_MOE_PREFETCH_CHUNK_ONLY: default on when LLAMA_PREFILL_CHUNK >= 2

    static llama_moe_prefetch_policy from_env() {
        llama_moe_prefetch_policy p;
        const char * r = getenv("LLAMA_MOE_PREFETCH_PP");
        const char * m = getenv("LLAMA_MOE_PREFETCH_PP_MIN");
        const char * c = getenv("LLAMA_PREFILL_CHUNK");
        const char * o = getenv("LLAMA_MOE_PREFETCH_CHUNK_ONLY");
        p.ring       = r && atoi(r) > 0 ? atoi(r) : 0;
        p.min_tokens = m ? (atoi(m) < 32 ? 32 : atoi(m)) : 1024;
        p.chunk_only = o ? atoi(o) != 0 : (c && atoi(c) >= 2);
        return p;
    }

    // has_pf_map: the layer has a compact-id map (a split layer under LLAMA_MOE_PREFETCH_PP); chunk_pass: the graph
    // is a pass of the chunk loop
    bool use(bool has_pf_map, bool op_offload, int64_t n_tokens, bool chunk_pass) const {
        return ring > 0 && has_pf_map && op_offload && n_tokens >= min_tokens && (!chunk_only || chunk_pass);
    }
};
