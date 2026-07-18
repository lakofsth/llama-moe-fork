#pragma once

// fork/phase-a — MoE router-lookahead prefetch (DeepSeek-V4).
//
// While layer L computes, run layer L+1's router on L's FFN-input state and
// issue MADV_WILLNEED for the predicted experts' weight slices (host/mmap
// layers only). Hash-routed layers (tid2eid) are prefetched from token ids at
// batch entry. Purely advisory: no compute-path change, logits bit-identical.
//
// Enabled via env LLAMA_MOE_PREFETCH="k[,depth]", e.g. "16" or "32,2".
// Measurement basis: 65.6% top-6 hit @k=6, 84.6% @k=16, 92.0% @k=32 (d=1);
// see ~/docs/hybrid-moe-fork-design-2026-07-18.md §2.2.

#include <memory>

struct llama_model;
struct ggml_tensor;

struct llama_moe_prefetch;

struct llama_moe_prefetch_deleter {
    void operator()(llama_moe_prefetch * p) const;
};

using llama_moe_prefetch_ptr = std::unique_ptr<llama_moe_prefetch, llama_moe_prefetch_deleter>;

// Returns null if the env var is unset, the arch is unsupported, or nothing is prefetchable.
llama_moe_prefetch_ptr llama_moe_prefetch_create(const llama_model & model);

// ggml_backend_sched_eval_callback; user_data = llama_moe_prefetch*.
bool llama_moe_prefetch_cb(struct ggml_tensor * t, bool ask, void * user_data);
