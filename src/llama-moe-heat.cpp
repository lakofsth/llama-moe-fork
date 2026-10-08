#include "llama-moe-heat.h"

#include "ggml-backend.h"

#include <cstdio>

bool llama_moe_heat_read_map(
        const std::string & path,
        int n_layer,
        int n_expert,
        int n_layer_short,
        std::vector<float> & heat,
        std::string & err,
        bool * padded) {
    err.clear();
    if (padded) {
        *padded = false;
    }
    if (n_layer <= 0 || n_expert <= 0) {
        err = "non-positive map shape";
        return false;
    }
    heat.assign((size_t) n_layer * (size_t) n_expert, 0.0f);

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open";
        return false;
    }
    const size_t n_full  = heat.size();
    const size_t n_short = n_layer_short > 0 && n_layer_short < n_layer ? (size_t) n_layer_short * (size_t) n_expert : 0;
    const size_t n_read  = fread(heat.data(), sizeof(float), n_full, f);

    bool ok = false;
    if (n_read == n_full) {
        // fork (posture L4): the full form must END here — a longer file is another shape
        if (fgetc(f) == EOF) {
            ok = true;
        } else {
            err = "file is longer than " + std::to_string(n_full) + " floats";
        }
    } else if (n_short != 0 && n_read == n_short && feof(f)) {
        ok = true;
        if (padded) {
            *padded = true;
        }
    } else {
        err = "read " + std::to_string(n_read) + " of " + std::to_string(n_full) + " floats"
            + (n_short != 0 ? " (or " + std::to_string(n_short) + " in the persisted form)" : "");
    }
    fclose(f);
    if (!ok) {
        heat.assign(n_full, 0.0f);
    }
    return ok;
}

// ---- online counters via the CPU backend's proc-address table (posture L6)

namespace {
struct llama_moe_online_fns {
    const int64_t * (*counts)(int32_t, int64_t *, int64_t *) = nullptr;
    void (*reset)(void) = nullptr;
    void (*decay)(float) = nullptr;
    bool resolved = false;
};

llama_moe_online_fns & llama_moe_online_resolve() {
    static llama_moe_online_fns fns;
    if (!fns.resolved) {
        fns.resolved = true;
        if (ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU")) {
            fns.counts = (const int64_t * (*)(int32_t, int64_t *, int64_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_moe_online_counts");
            fns.reset  = (void (*)(void))  ggml_backend_reg_get_proc_address(reg, "ggml_cpu_moe_online_reset");
            fns.decay  = (void (*)(float)) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_moe_online_decay");
        }
    }
    return fns;
}
} // namespace

const int64_t * llama_moe_online_counts(int32_t il, int64_t * sentinel, int64_t * total) {
    auto & f = llama_moe_online_resolve();
    return f.counts ? f.counts(il, sentinel, total) : nullptr;
}

void llama_moe_online_reset(void) {
    auto & f = llama_moe_online_resolve();
    if (f.reset) f.reset();
}

void llama_moe_online_decay(float keep) {
    auto & f = llama_moe_online_resolve();
    if (f.decay) f.decay(keep);
}
