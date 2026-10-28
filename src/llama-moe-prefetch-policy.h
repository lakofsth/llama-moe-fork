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

    // PREFETCH-PP-MIN-SEQS (review F2): a ubatch split equally over n_seqs sequences holds n_seqs * floor(n_ubatch /
    // n_seqs) tokens (1023 for 3 sequences at -ub 1024, 1020 for 6), so the minimum is rounded down to a multiple of
    // n_seqs: the ubatch the default would let through for one sequence is let through for several
    bool wide(int64_t n_tokens, int64_t n_seqs) const {
        const int64_t m = n_seqs > 1 ? (min_tokens / n_seqs) * n_seqs : min_tokens;
        return n_tokens >= (m < 32 ? 32 : m);
    }

    // has_pf_map: the layer has a compact-id map (a split layer under LLAMA_MOE_PREFETCH_PP); n_seqs: the ubatch's
    // sequences when it is split equally (else 1); chunk_pass: the graph is a pass of the chunk loop
    bool use(bool has_pf_map, bool op_offload, int64_t n_tokens, int64_t n_seqs, bool chunk_pass) const {
        return ring > 0 && has_pf_map && op_offload && wide(n_tokens, n_seqs) && (!chunk_only || chunk_pass);
    }
};
