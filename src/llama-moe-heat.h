#pragma once

// fork: reading a profiled MoE heat map (LLAMA_MOE_HEAT_FILE / --moe-heat-label sidecar).
// Kept apart from llama-model.cpp so the length contract has a unit test (tests/test-moe-heat-map.cpp).

#include <cstdint>
#include <string>
#include <vector>

// Read a per-layer x per-expert f32 map into `heat` (sized n_layer*n_expert on return).
// Accepted lengths, and nothing else:
//   - exactly n_layer*n_expert floats;
//   - exactly n_layer_short*n_expert floats when n_layer_short < n_layer — the persisted form,
//     which excludes trailing NextN layers; the missing rows are left at zero heat and
//     `padded` (if given) is set.
// A file that is shorter, longer, or carries trailing bytes after the expected floats is refused
// (`err` says why) — a map of another model or shape must never be silently reinterpreted.
bool llama_moe_heat_read_map(
        const std::string & path,
        int n_layer,
        int n_expert,
        int n_layer_short,
        std::vector<float> & heat,
        std::string & err,
        bool * padded = nullptr);

// fork (posture L6): the online expert-heat counters live in the CPU backend
// (ggml_cpu_moe_online_*). With GGML_BACKEND_DL that backend is a separately loaded
// library, so libllama reaches them through the backend registry's proc-address table,
// never by direct symbol — the same route as ggml_backend_cpu_set_threadpool. Without a
// CPU backend loaded, counts returns nullptr and the others are no-ops.
const int64_t * llama_moe_online_counts(int32_t il, int64_t * sentinel, int64_t * total);
void            llama_moe_online_reset(void);
void            llama_moe_online_decay(float keep);
