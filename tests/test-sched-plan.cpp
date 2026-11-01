// test-sched-plan — PLAN-PER-ARENA (tests/INVARIANTS.md): a graph whose scheduler plan was kept aside
// (ggml_backend_sched_plan_set / _use) computes exactly what it computes when it is split and allocated afresh, and a
// kept plan is never restored once the allocator's buffers have been freed or replaced, or once it was forgotten.
//
// llama_context's MTP drafter alternates two graphs on one scheduler (catch-up: 3 tokens, no output; draft: 1 token,
// 1 output). This test does the same with two graphs, each in its own ggml context as llama's two arenas are, with new
// input values every round. Each graph runs a weight that lives in host memory through a GPU split (an input copy the
// scheduler makes, which is part of the kept state) and a CPU split, when a GPU is present; on the CPU alone it is one
// split. Every round's output is compared with the same arithmetic done on the host (relative error < 1e-2: the GPU
// matmul of the 128-token graph is off by ~2e-3; a stale or mis-swapped plan is off by ~1).
// Checks:
//   (1) alternation: from the third round on every round is a restore (plan_use true), and every round is right;
//   (2) a graph that needs larger buffers (the allocator replaces them): no kept plan is restored afterwards;
//   (3) the caller forgets the current graph's plan before rebuilding it: the other graph's aside plan survives;
//   (4) plan_forget of the aside plan: it is not restored.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int K = 256, M = 64;
static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

struct graph_t {
    std::vector<uint8_t> mem;
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x   = nullptr;
    ggml_tensor  * out = nullptr;
    ggml_tensor  * mm  = nullptr; // runs on the GPU when there is one
    ggml_tensor  * mid = nullptr; // runs on the CPU
    int n = 0;
    uint64_t key = 0;
};

// out = (2 * (w x)) ^ 2 + 1, built fresh into g's own context (llama: res->reset() + build_graph)
static void build(graph_t & g, ggml_tensor * w, int n) {
    if (g.ctx) ggml_free(g.ctx);
    g.mem.resize(32 * ggml_tensor_overhead() + ggml_graph_overhead());
    ggml_init_params ip = { g.mem.size(), g.mem.data(), true };
    g.ctx = ggml_init(ip);
    g.n   = n;
    g.x   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, n);
    ggml_set_input(g.x);
    g.mm  = ggml_mul_mat(g.ctx, w, g.x);
    g.mid = ggml_scale(g.ctx, ggml_add(g.ctx, g.mm, g.mm), 1.0f);
    g.out = ggml_add(g.ctx, ggml_sqr(g.ctx, g.mid), ggml_arange(g.ctx, 1.0f, 2.0f, 1.0f));
    ggml_set_output(g.out);
    g.gf  = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
}

static uint64_t g_serial = 0;

// llama's rebuild path: forget this graph's old plan, rebuild, reset (which keeps the current named state aside),
// assign backends, allocate, name the new state
static void rebuild(ggml_backend_sched_t sched, graph_t & g, ggml_tensor * w, int n, ggml_backend_t gpu, ggml_backend_t cpu) {
    ggml_backend_sched_plan_forget(sched, g.key);
    g.key = 0;
    build(g, w, n);
    ggml_backend_sched_reset(sched);
    if (gpu) {
        ggml_backend_sched_set_tensor_backend(sched, g.mm, gpu);
        ggml_backend_sched_set_tensor_backend(sched, g.mid, cpu);
    }
    if (!ggml_backend_sched_alloc_graph(sched, g.gf)) { fprintf(stderr, "alloc failed\n"); exit(2); }
    g.key = ++g_serial;
    ggml_backend_sched_plan_set(sched, g.key);
}

static void run_check(ggml_backend_sched_t sched, graph_t & g, const std::vector<float> & wf, unsigned seed, const char * what) {
    std::vector<float> xv((size_t) K * g.n);
    srand(seed);
    for (float & v : xv) v = (float) rand() / RAND_MAX - 0.5f;
    ggml_backend_tensor_set(g.x, xv.data(), 0, ggml_nbytes(g.x));
    if (ggml_backend_sched_graph_compute(sched, g.gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }
    std::vector<float> o(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, o.data(), 0, ggml_nbytes(g.out));
    double maxerr = 0;
    for (int t = 0; t < g.n; ++t) {
        for (int m = 0; m < M; ++m) {
            double d = 0;
            for (int k = 0; k < K; ++k) d += (double) wf[(size_t) m * K + k] * xv[(size_t) t * K + k];
            const double want = 4.0 * d * d + 1.0;
            maxerr = std::fmax(maxerr, std::fabs(want - o[(size_t) t * M + m]) / (1.0 + std::fabs(want)));
        }
    }
    CHECK(maxerr < 1e-2, "%s (n=%d): relative error %g", what, g.n, maxerr);
}

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    printf("backends: %s%s\n", gpu ? ggml_backend_name(gpu) : "", gpu ? " + CPU" : "CPU only");

    // the weight lives in host memory, as a model's host-resident tensors do
    std::vector<float> wf((size_t) K * M);
    srand(11);
    for (float & v : wf) v = (float) rand() / RAND_MAX - 0.5f;
    ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, wf.data(), 0, ggml_nbytes(w));

    ggml_backend_t bes[2]; int nb = 0;
    if (gpu) bes[nb++] = gpu;
    bes[nb++] = cpu;
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, nb, 2048, false, true);

    // reserve for the larger of the two shapes first, as llama reserves its worst case
    graph_t g[2];
    rebuild(sched, g[0], w, 3, gpu, cpu);
    ggml_backend_sched_plan_clear(sched);
    ggml_backend_sched_reset(sched);
    g[0].key = 0;

    // (1) alternation
    int restored = 0, rounds = 16;
    for (int r = 0; r < rounds; ++r) {
        graph_t & cur = g[r % 2];
        const int n = r % 2 ? 1 : 3;
        if (cur.ctx && cur.n == n && ggml_backend_sched_plan_use(sched, cur.key)) {
            restored++;
        } else {
            rebuild(sched, cur, w, n, gpu, cpu);
        }
        run_check(sched, cur, wf, 100 + r, "alternation");
    }
    printf("(1) alternation: %d of %d rounds restored a kept plan\n", restored, rounds);
    CHECK(restored == rounds - 2, "(1) expected %d restores, got %d", rounds - 2, restored);

    // (2) a graph that needs larger buffers: the allocator replaces them, and no kept plan may be restored
    {
        graph_t big;
        rebuild(sched, big, w, 128, gpu, cpu);
        run_check(sched, big, wf, 300, "large graph");
        CHECK(!ggml_backend_sched_plan_use(sched, g[1].key), "(2) the plan of the last alternating graph was restored after the buffers were replaced");
        CHECK(!ggml_backend_sched_plan_use(sched, g[0].key), "(2) the plan of the first alternating graph was restored after the buffers were replaced");
        ggml_backend_sched_plan_forget(sched, big.key);
        ggml_free(big.ctx);
    }
    // both rebuild and run right again; the next switch restores
    rebuild(sched, g[0], w, 3, gpu, cpu); run_check(sched, g[0], wf, 400, "after (2), first");
    rebuild(sched, g[1], w, 1, gpu, cpu); run_check(sched, g[1], wf, 401, "after (2), second");
    CHECK(ggml_backend_sched_plan_use(sched, g[0].key), "(2) no restore after both graphs were rebuilt");
    run_check(sched, g[0], wf, 402, "after (2), restored");

    // (3) g[0] is current: the caller rebuilds it (forget, then reset); g[1]'s aside plan must survive that
    rebuild(sched, g[0], w, 3, gpu, cpu);
    run_check(sched, g[0], wf, 500, "(3) rebuilt current");
    CHECK(ggml_backend_sched_plan_use(sched, g[1].key), "(3) the aside plan was lost when the current graph was rebuilt");
    run_check(sched, g[1], wf, 501, "(3) restored aside");

    // (4) g[0] is now aside: forgotten, it is not restored
    ggml_backend_sched_plan_forget(sched, g[0].key);
    CHECK(!ggml_backend_sched_plan_use(sched, g[0].key), "(4) a forgotten plan was restored");

    ggml_backend_sched_free(sched);
    for (auto & x : g) if (x.ctx) ggml_free(x.ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("%s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
