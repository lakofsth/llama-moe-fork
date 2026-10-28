// test-moe-prefetch — the scheduler's copy-stream expert prefetch (LLAMA_MOE_PREFETCH_PP, fork-kernel-fusion, P1).
//
// A chain of L "layers", each two MUL_MAT_IDs (W1: K->M, W2: M->K) over host-resident WEIGHTS, offloaded to CUDA by
// op_offload (64 rows). Per layer a random subset of experts is "resident": its ids are the out-of-range sentinel on
// both runs, so neither run computes them. Reference run: original ids ("ffn_moe_ids_cpu-l"), the stock partial
// upload. Prefetch run: COMPACT ids ("ffn_moe_ids_pf-l": index in the ascending non-resident list) with that list
// registered as the plan — the scheduler streams each weight's plan into a slot ring on a second stream.
// The two must agree EXACTLY (each output row is one expert x one token on the same kernel). The prefetch graph is
// run three times — the third after the resident sets AND the plans change — so slot reuse across graphs, the
// read/upload events and plan updates are all exercised. A compact id read through the stock upload would select
// the wrong expert, so agreement also proves the slot path served the run; the log must name allocated slots.
// R = LLAMA_MOE_PREFETCH_PP (default 3 here: slots mix W1- and W2-sized entries). Skips (exit 0) without CUDA.
// PF-GROW-FALLBACK (review F5): run again with LLAMA_MOE_PF_GROW_FAIL_AFTER=N (every slot grow after the first N
// fails, as on a full device): a graph that needs a slot it cannot grow uploads its experts into its own weight copy
// at its split, and must still agree exactly; the run must log the fallback. N=0: no slot ever; N=1: a mixed ring.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int K = 256, M = 64, E = 16, U = 4, T = 64, L = 3;
static int slot_lines = 0;
static int fallback_lines = 0; // PF-GROW-FALLBACK

static void log_cb(ggml_log_level level, const char * text, void * ud) {
    (void) ud;
    if (strstr(text, "moe-prefetch: slot")) slot_lines++;
    if (strstr(text, "moe-prefetch: cannot grow slot")) fallback_lines++;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

struct layer_sets { std::vector<std::vector<int32_t>> ids_orig, ids_pf, plan; };

// per layer: U distinct experts per token; experts in `resident` -> sentinel E+1+e in both encodings
static layer_sets make_sets(std::mt19937 & rng) {
    layer_sets s;
    for (int l = 0; l < L; l++) {
        std::vector<int> perm(E); for (int e = 0; e < E; e++) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<bool> res(E, false); for (int r = 0; r < 6; r++) res[perm[r]] = true;
        std::vector<int32_t> plan, cidx(E, -1);
        for (int e = 0; e < E; e++) if (!res[e]) { cidx[e] = (int32_t) plan.size(); plan.push_back(e); }
        std::vector<int32_t> io(U*T), ip(U*T);
        for (int t = 0; t < T; t++) {
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int u = 0; u < U; u++) {
                const int e = perm[u];
                io[t*U + u] = res[e] ? E + 1 + e : e;
                ip[t*U + u] = res[e] ? E + 1 + e : cidx[e];
            }
        }
        s.ids_orig.push_back(io); s.ids_pf.push_back(ip); s.plan.push_back(plan);
    }
    return s;
}

struct run_graph { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * x; std::vector<ggml_tensor *> ids; ggml_tensor * out; };

static run_graph build(const std::vector<ggml_tensor *> & w1, const std::vector<ggml_tensor *> & w2, bool pf) {
    run_graph g;
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.x = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, K, 1, T); ggml_set_input(g.x); ggml_set_name(g.x, "x");
    ggml_tensor * cur = g.x;
    for (int l = 0; l < L; l++) {
        ggml_tensor * ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, U, T); ggml_set_input(ids);
        ggml_format_name(ids, pf ? "ffn_moe_ids_pf-%d" : "ffn_moe_ids_cpu-%d", l);
        g.ids.push_back(ids);
        ggml_tensor * h = ggml_mul_mat_id(g.ctx, w1[l], cur, ids);           // [M, U, T]
        h = ggml_scale(g.ctx, ggml_tanh(g.ctx, h), 0.5f);
        ggml_tensor * y = ggml_mul_mat_id(g.ctx, w2[l], h, ids);             // [K, U, T]
        cur = ggml_cont(g.ctx, ggml_view_3d(g.ctx, y, K, 1, T, y->nb[1], y->nb[2], 0)); // slot 0's row feeds the next layer
        cur = ggml_tanh(g.ctx, cur);
    }
    g.out = cur;
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

// reuse = compute the graph the scheduler already split and allocated again, as llama.cpp's graph reuse does (a
// cgraph is split in place, so a NEW split needs a freshly built graph)
static std::vector<float> run(ggml_backend_sched_t sched, run_graph & g, const std::vector<float> & x, const std::vector<std::vector<int32_t>> & ids, bool reuse = false) {
    if (!reuse) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, g.gf)) { fprintf(stderr, "alloc failed\n"); exit(1); }
    }
    ggml_backend_tensor_set(g.x, x.data(), 0, x.size()*sizeof(float));
    for (int l = 0; l < L; l++) ggml_backend_tensor_set(g.ids[l], ids[l].data(), 0, ids[l].size()*sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sched, g.gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, out.size()*sizeof(float));
    return out;
}

static int fails = 0;
static void same(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    size_t diff = 0; double mx = 0, mag = 0;
    for (size_t i = 0; i < a.size(); i++) { if (a[i] != b[i]) diff++; mx = std::max(mx, (double) std::fabs(a[i]-b[i])); mag += std::fabs(b[i]); }
    const bool ok = diff == 0 && mag > 0;
    printf("%s %s: %zu of %zu values differ (max |d| %.3g, mean |ref| %.3g)\n", ok ? "ok  " : "FAIL", what, diff, a.size(), mx, mag / a.size());
    fails += !ok;
}

int main() {
    if (!getenv("LLAMA_MOE_PREFETCH_PP")) setenv("LLAMA_MOE_PREFETCH_PP", "3", 1);
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) { printf("skip: no GPU device\n"); return 0; }
    ggml_backend_t be_gpu = ggml_backend_dev_init(gpu, nullptr);
    ggml_backend_t be_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    // host WEIGHTS
    ggml_init_params wp = { 4*L*ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    std::vector<ggml_tensor *> w1, w2;
    for (int l = 0; l < L; l++) {
        w1.push_back(ggml_new_tensor_3d(wctx, GGML_TYPE_Q8_0, K, M, E)); ggml_format_name(w1.back(), "blk.%d.w1_exps", l);
        w2.push_back(ggml_new_tensor_3d(wctx, GGML_TYPE_Q8_0, M, K, E)); ggml_format_name(w2.back(), "blk.%d.w2_exps", l);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto * w : { &w1, &w2 }) for (ggml_tensor * t : *w) {
        std::vector<float> f(ggml_nelements(t)); for (auto & v : f) v = uni(rng) / std::sqrt((float) t->ne[0]);
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, t->ne[1]*t->ne[2], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
    std::vector<float> x(K*T); for (auto & v : x) v = uni(rng);

    ggml_backend_t bes[2] = { be_gpu, be_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, 2, 2048, false, true);

    for (int round = 0; round < 2; round++) {
        run_graph gr = build(w1, w2, false), gp = build(w1, w2, true);
        layer_sets s = make_sets(rng);
        for (int l = 0; l < L; l++) {
            ggml_backend_sched_moe_pf_plan(w1[l], s.plan[l].data(), (int) s.plan[l].size());
            ggml_backend_sched_moe_pf_plan(w2[l], s.plan[l].data(), (int) s.plan[l].size());
        }
        const std::vector<float> ref = run(sched, gr, x, s.ids_orig);
        for (int rep = 0; rep < (round == 0 ? 2 : 1); rep++) {
            const std::vector<float> got = run(sched, gp, x, s.ids_pf, rep > 0);
            const std::string what = "round " + std::to_string(round) + (rep ? " reused graph" : " new graph") + " (prefetch vs stock upload)";
            same(what.c_str(), got, ref);
        }
        ggml_free(gr.ctx); ggml_free(gp.ctx);
    }
    const char * fail_after = getenv("LLAMA_MOE_PF_GROW_FAIL_AFTER");
    if (fail_after) {
        if (fallback_lines == 0) { printf("FAIL LLAMA_MOE_PF_GROW_FAIL_AFTER=%s but no grow failed (the fallback never ran)\n", fail_after); fails++; }
        else                     { printf("ok   the fallback ran (%d slot allocation line(s) before it)\n", slot_lines); }
    } else if (slot_lines == 0) { printf("FAIL no prefetch slot was allocated (the slot path never ran)\n"); fails++; }
    else                 { printf("ok   %d prefetch slot allocation line(s)\n", slot_lines); }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    ggml_backend_free(be_gpu); ggml_backend_free(be_cpu);
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
