// test-moe-online-hits — the CPU mul_mat_id ids sweep's online counters (fork-strata-ports).
//
// The split graph's CPU branch gets ids where a GPU-resident expert appears as a SENTINEL (any id
// outside [0, n_as)). A resident expert e has its own sentinel, n_as + 1 + e (llama_moe_cpu_sentinel),
// so its hits are counted per expert (ggml_cpu_moe_online_hits); the shared sentinel n_as (kept for
// the openai-moe bias copies) only adds to the layer total. Cold ids are counted per expert as
// before, and every sentinel slot's output is zero. One mul_mat_id, n_as = 4 experts, 3 used per
// token, 2 tokens, layer 0:
//     token 0: 7 (= 4+1+2, resident 2), 1 (cold), 4 (shared sentinel)
//     token 1: 5 (= 4+1+0, resident 0), 3 (cold), 2 (cold)
// expected: hits[0]=1 hits[2]=1 (others 0), sentinel 3, counts[1]=counts[2]=counts[3]=1, total 6.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int fails = 0;
static void expect(const char * what, long long got, long long want) {
    const bool ok = got == want;
    printf("%s %s: %lld (want %lld)\n", ok ? "ok  " : "FAIL", what, got, want);
    fails += !ok;
}

static void run(int n_threads) {
    ggml_init_params ip = { 16*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const int K = 4, M = 2, n_as = 4, n_used = 3, n_tok = 2;
    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, n_as);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_set_name(ids, "ffn_moe_ids_cpu-0");
    for (int i = 0; i < K*M*n_as; i++)       ((float *) a->data)[i] = 1.0f;
    for (int i = 0; i < K*n_used*n_tok; i++) ((float *) b->data)[i] = 1.0f;
    const int32_t v[6] = { n_as + 1 + 2, 1, n_as,   n_as + 1 + 0, 3, 2 };
    memcpy(ids->data, v, sizeof(v));
    ggml_tensor * out = ggml_mul_mat_id(ctx, a, b, ids); // [M, n_used, n_tok]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_cpu_moe_online_reset();
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    char tag[64];
    int64_t sent = 0, tot = 0;
    const int64_t * c = ggml_cpu_moe_online_counts(0, &sent, &tot);
    const int64_t * h = ggml_cpu_moe_online_hits(0);
    snprintf(tag, sizeof tag, "[%d thr] total", n_threads);    expect(tag, tot, 6);
    snprintf(tag, sizeof tag, "[%d thr] sentinel", n_threads); expect(tag, sent, 3);
    const long long want_c[4] = { 0, 1, 1, 1 }, want_h[4] = { 1, 0, 1, 0 };
    for (int e = 0; e < 4; e++) {
        snprintf(tag, sizeof tag, "[%d thr] counts[%d]", n_threads, e); expect(tag, c[e], want_c[e]);
        snprintf(tag, sizeof tag, "[%d thr] hits[%d]", n_threads, e);   expect(tag, h[e], want_h[e]);
    }
    // outputs: sentinel slots (token 0 slots 0 and 2, token 1 slot 0) are zero; cold slots are K
    const float * o = (const float *) out->data;
    const int sentinel_slot[6] = { 1, 0, 1,  1, 0, 0 };
    for (int t = 0; t < n_tok; t++) for (int s = 0; s < n_used; s++) {
        const float want = sentinel_slot[t*n_used + s] ? 0.0f : (float) K;
        snprintf(tag, sizeof tag, "[%d thr] out[t%d,s%d]", n_threads, t, s);
        expect(tag, (long long) o[(t*n_used + s)*M], (long long) want);
    }
    // decay keeps the per-expert hits in step with the totals
    ggml_cpu_moe_online_decay(0.5f);
    snprintf(tag, sizeof tag, "[%d thr] hits[2] after decay 0.5", n_threads); expect(tag, h[2], 0);
    ggml_free(ctx);
}

int main() {
    setenv("LLAMA_MOE_HEAT_ONLINE", "1", 1);
    run(1);
    run(4);
    printf("test-moe-online-hits: %d failed\n", fails);
    return fails ? 1 : 0;
}
