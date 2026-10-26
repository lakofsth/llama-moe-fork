// test-cuda-graph-key — GRAPH-KEY (tests/INVARIANTS.md): GGML_CUDA_GRAPH_SHAPE_KEY=1 keeps one CUDA graph per split
// shape, and the key never changes a result.
//
// llama builds every batch in one reused ggml context, so graphs of different batch sizes start at the same node
// address. This test does the same: one context buffer, rebuilt for n = 1 or 3 tokens in an alternating order, with
// new input values every round, allocated by one of two graph allocators that change every four rounds — so the same
// shape at the same node addresses also arrives with different data buffers, which a graph reused without
// ggml_cuda_graph_update_required's comparison would read stale. Each round's CUDA result is compared with the CPU backend's.
// The parent runs itself twice, shape key off and on, with GGML_CUDA_GRAPH_STATS=1 and a pause before the last round
// so the stats line covers the run, and requires (a) every round of both runs to match the CPU, and (b) more rounds
// served by a launched graph with the key on than off (off, each change of shape resets the one cached graph's
// warmup, so alternating shapes run eagerly). Skips (exit 0) without a CUDA device.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const int K = 512, M = 96;
static std::string g_log;

static void log_cb(ggml_log_level, const char * text, void *) {
    g_log += text;
    fputs(text, stderr);
}

static std::vector<float> run_graph(ggml_backend_t be, ggml_gallocr_t galloc, std::vector<uint8_t> & mem,
                                    ggml_tensor * w, const std::vector<float> & xv, int n) {
    ggml_init_params ip = { mem.size(), mem.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_tensor * z = ggml_scale(ctx, ggml_add(ctx, y, y), 0.5f);
    ggml_set_output(z);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, z);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "alloc failed\n"); exit(2); }
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out(ggml_nelements(z));
    ggml_backend_tensor_get(z, out.data(), 0, ggml_nbytes(z));
    ggml_free(ctx);
    return out;
}

static int child() {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { printf("no GPU device: skipped\n"); return 0; }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();

    // the weight lives in its own context on each backend, as a model's do
    std::vector<float> wf(K * M);
    srand(7);
    for (float & v : wf) v = (float) rand() / RAND_MAX - 0.5f;
    std::vector<char> wq(ggml_row_size(GGML_TYPE_Q8_0, K) * M);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, M, K, nullptr);
    ggml_tensor * w[2]; ggml_context * wctx[2]; ggml_backend_t bes[2] = { gpu, cpu };
    for (int b = 0; b < 2; ++b) {
        ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
        wctx[b] = ggml_init(ip);
        w[b] = ggml_new_tensor_2d(wctx[b], GGML_TYPE_Q8_0, K, M);
        ggml_backend_alloc_ctx_tensors(wctx[b], bes[b]);
        ggml_backend_tensor_set(w[b], wq.data(), 0, wq.size());
    }
    std::vector<uint8_t> mem[2];
    ggml_gallocr_t galloc[2][2];
    for (int b = 0; b < 2; ++b) {
        mem[b].resize(16 * ggml_tensor_overhead() + ggml_graph_overhead());
        for (int a = 0; a < 2; ++a) galloc[b][a] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bes[b]));
    }

    const int shapes[] = { 1, 1, 1, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3 };
    const int R = (int) (sizeof(shapes) / sizeof(shapes[0]));
    int bad = 0;
    for (int r = 0; r < R; ++r) {
        if (r == R - 1) std::this_thread::sleep_for(std::chrono::milliseconds(1100));  // the stats period ends here
        const int n = shapes[r];
        std::vector<float> xv(K * n);
        for (float & v : xv) v = (float) rand() / RAND_MAX - 0.5f;
        const int a = (r / 4) % 2;
        const std::vector<float> g = run_graph(gpu, galloc[0][a], mem[0], w[0], xv, n);
        const std::vector<float> c = run_graph(cpu, galloc[1][a], mem[1], w[1], xv, n);
        double num = 0, den = 0;
        for (size_t i = 0; i < g.size(); ++i) { num += (g[i] - c[i]) * (g[i] - c[i]); den += c[i] * c[i]; }
        if (num / den >= 1e-3) { ++bad; printf("round %d n=%d: nmse %.3g against the CPU\n", r, n, num / den); }
    }
    // the stats line is printed by the first compute after the period; one more GPU round triggers it if needed
    if (g_log.find("cuda-graph-stats:") == std::string::npos) {
        std::vector<float> xv(K, 0.25f);
        run_graph(gpu, galloc[0][0], mem[0], w[0], xv, 1);
    }
    const size_t p = g_log.find("launches ");
    const long launches = p == std::string::npos ? -1 : atol(g_log.c_str() + p + 9);
    printf("RESULT bad=%d launches=%ld\n", bad, launches);
    return 0;
}

static bool run_child(const char * self, const char * shape_key, int & bad, long & launches) {
    std::string cmd = std::string("GGML_CUDA_GRAPH_STATS=1 GGML_CUDA_GRAPH_SHAPE_KEY=") + shape_key + " '" + self + "' --child 2>/dev/null";
    FILE * f = popen(cmd.c_str(), "r");
    if (!f) return false;
    char line[512]; bool got = false, skipped = false;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "RESULT bad=%d launches=%ld", &bad, &launches) == 2) got = true;
        if (strstr(line, "skipped")) skipped = true;
    }
    pclose(f);
    if (skipped) { printf("no GPU device: skipped\n"); exit(0); }
    return got;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) return child();
    int bad_off = -1, bad_on = -1; long cap_off = -1, cap_on = -1;
    if (!run_child(argv[0], "0", bad_off, cap_off) || !run_child(argv[0], "1", bad_on, cap_on)) {
        printf("FAIL: a child run did not report\n"); return 1;
    }
    printf("shape key off: %d rounds differ from the CPU, %ld graph launches\n", bad_off, cap_off);
    printf("shape key on:  %d rounds differ from the CPU, %ld graph launches\n", bad_on, cap_on);
    const bool ok = bad_off == 0 && bad_on == 0 && cap_off >= 0 && cap_on > cap_off;
    printf("%s\n", ok ? "test-cuda-graph-key: ok" : "test-cuda-graph-key: FAIL");
    return ok ? 0 : 1;
}
