// test-cuda-q8-1-cache — the CUDA backend's shared q8_1 copy of a matvec input (fork-kernel-fusion).
//
// mul_mat_vec_q keeps the q8_1 copy of its F32 input for the next matvec on the same activation
// (ggml_cuda_q8_1_cache). The copy must never outlive the bytes it was made from:
//   (a) a node that writes over the input in the same graph drops it (invalidate) — y2 below reads x
//       after an in-place scale and must see the scaled values, not the copy y1 left;
//   (b) a new evaluation starts empty (clear) — the second run sets x from the host between graphs,
//       which no node of the graph sees.
// And it must be used when it holds: y3 reads the scaled x again (a hit) and must match too.
// Reference: the same graph on the CPU backend. The two quantize the activation differently (q8_1 vs
// q8_0), so values are compared at a tolerance far below the error a stale copy makes (a sign flip
// of every output here). Skips (exit 0) without a CUDA device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int fails = 0;

static void check(const char * what, const std::vector<float> & got, const std::vector<float> & want) {
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        num += (got[i] - want[i]) * (got[i] - want[i]);
        den += want[i] * want[i];
    }
    const double nmse = den > 0 ? num / den : num;
    const bool ok = nmse < 1e-3;
    printf("%s %s: nmse %.3g\n", ok ? "ok  " : "FAIL", what, nmse);
    fails += !ok;
}

struct graph_run {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x = nullptr, * w1 = nullptr, * w2 = nullptr, * y1 = nullptr, * y2 = nullptr, * y3 = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

static const int K = 512, M = 96;

static graph_run build(ggml_backend_t be) {
    graph_run g;
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.x  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32,  K, 1);
    g.w1 = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, K, M);
    g.w2 = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, K, M);
    g.y1 = ggml_mul_mat(g.ctx, g.w1, g.x);                    // quantizes x, keeps the copy
    ggml_tensor * xs = ggml_scale_inplace(g.ctx, g.x, -1.5f); // writes over x
    g.y2 = ggml_mul_mat(g.ctx, g.w2, xs);                     // must quantize again
    g.y3 = ggml_mul_mat(g.ctx, g.w1, xs);                     // may reuse y2's copy
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.y1);
    ggml_build_forward_expand(g.gf, g.y2);
    ggml_build_forward_expand(g.gf, g.y3);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static void set_q8_0(ggml_tensor * t, unsigned seed) {
    std::vector<float> f(K*M);
    srand(seed);
    for (float & v : f) v = (float) rand() / RAND_MAX - 0.5f;
    std::vector<char> q(ggml_row_size(GGML_TYPE_Q8_0, K) * M);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, M, K, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void set_x(ggml_tensor * t, unsigned seed) {
    std::vector<float> f(K);
    srand(seed);
    for (float & v : f) v = (float) rand() / RAND_MAX - 0.25f;
    ggml_backend_tensor_set(t, f.data(), 0, f.size()*sizeof(float));
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu == nullptr || std::string(ggml_backend_dev_name(gpu)).rfind("CUDA", 0) != 0) {
        printf("skip: no CUDA device\n");
        return 0;
    }
    ggml_backend_t cuda = ggml_backend_dev_init(gpu, nullptr);
    ggml_backend_t cpu  = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    graph_run gc = build(cuda), gr = build(cpu);
    set_q8_0(gc.w1, 1); set_q8_0(gr.w1, 1);
    set_q8_0(gc.w2, 2); set_q8_0(gr.w2, 2);

    for (int run = 0; run < 3; ++run) { // a new x from the host each evaluation
        set_x(gc.x, 10 + run); set_x(gr.x, 10 + run);
        ggml_backend_graph_compute(cuda, gc.gf);
        ggml_backend_graph_compute(cpu,  gr.gf);
        char name[64];
        snprintf(name, sizeof(name), "run %d y1 (x from the host)", run);       check(name, get(gc.y1), get(gr.y1));
        snprintf(name, sizeof(name), "run %d y2 (x scaled in place)", run);     check(name, get(gc.y2), get(gr.y2));
        snprintf(name, sizeof(name), "run %d y3 (same input as y2)", run);      check(name, get(gc.y3), get(gr.y3));
    }

    ggml_backend_buffer_free(gc.buf); ggml_backend_buffer_free(gr.buf);
    ggml_free(gc.ctx); ggml_free(gr.ctx);
    ggml_backend_free(cuda); ggml_backend_free(cpu);
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
