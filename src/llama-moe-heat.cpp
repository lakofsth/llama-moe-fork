#include "llama-moe-heat.h"

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
