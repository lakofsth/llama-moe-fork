// fork-strata-ports TAIL-SKIP (tests/INVARIANTS.md): a non-resident expert routed only at rank >= R by EVERY token of the
// pass becomes -1-e on the CPU branch; one any token routes at rank < R is kept everywhere; sentinels are untouched; rank
// is by router weight, not by slot position; and the CPU mul_mat_id zeroes a -1-e row while the online heat counter
// still counts it as cold demand for e (not as a resident hit).
#include "ggml.h"
#include "ggml-cpu.h"
#include "../src/llama-graph.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct llama_moe_tail_skip_params { int32_t rank; int32_t n_expert; };

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { n_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void test_rule() {
    ggml_init_params ip = { 16*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const int k = 4, T = 3, NE = 8, R = 2;
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, T);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, T);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, T);
    // slot weights are NOT in descending order, so position != rank:
    //   token 0: ids {1, 2, 3, 9}   w {0.1, 0.4, 0.3, 0.2} -> ranks {3, 0, 1, 2}
    //   token 1: ids {4, 1, 10, 5}  w {0.5, 0.05, 0.3, 0.15} -> ranks {0, 3, 1, 2}
    //   token 2: ids {6, 3, 4, 11}  w {0.1, 0.2, 0.3, 0.4}  -> ranks {3, 2, 1, 0}
    // non-resident (< 8): 1 ranks {3, 3} -> skip; 2 rank 0 -> keep; 3 ranks {1, 2} -> keep (head in token 0, tail in
    // token 2: a per-pair rule would skip the token-2 row); 4 ranks {0, 1} -> keep; 5 rank 2 -> skip (slot 3 of token 1,
    // position 3 >= R would skip it too); 6 rank 3 -> skip; sentinels 9, 10, 11 untouched (9 sits at rank 2).
    const int32_t vid[T][k] = { {1, 2, 3, 9}, {4, 1, 10, 5}, {6, 3, 4, 11} };
    const float   vw [T][k] = { {0.1f, 0.4f, 0.3f, 0.2f}, {0.5f, 0.05f, 0.3f, 0.15f}, {0.1f, 0.2f, 0.3f, 0.4f} };
    for (int t = 0; t < T; t++) for (int q = 0; q < k; q++) {
        ((int32_t *) ids->data)[t*k + q] = vid[t][q];
        ((float   *) w->data)[t*k + q]   = vw[t][q];
    }
    llama_moe_tail_skip_params p = { R, NE };
    llama_moe_tail_skip_apply(dst, ids, w, 0, 1, &p);
    const int32_t want[T][k] = { {-2, 2, 3, 9}, {4, -2, 10, -6}, {-7, 3, 4, 11} };
    for (int t = 0; t < T; t++) for (int q = 0; q < k; q++) {
        const int32_t got = ((int32_t *) dst->data)[t*k + q];
        CHECK(got == want[t][q], "rule: token %d slot %d: got %d want %d", t, q, got, want[t][q]);
    }
    // a thread other than 0 writes nothing
    memset(dst->data, 0x7f, ggml_nbytes(dst));
    llama_moe_tail_skip_apply(dst, ids, w, 1, 2, &p);
    CHECK(((int32_t *) dst->data)[0] == 0x7f7f7f7f, "rule: ith 1 wrote");
    ggml_free(ctx);
}

static void test_cpu_count() {
    // the CPU mul_mat_id: -1-e zeroes the row and counts e as cold demand; n_as is a plain sentinel
    ggml_init_params ip = { 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const int n_embd = 32, n_ff = 4, n_as = 6, k = 3, T = 2, il = 5;
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_as);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, T);
    ggml_set_name(ids, "ffn_moe_ids_cpu-5");
    for (int64_t i = 0; i < ggml_nelements(as); i++) ((float *) as->data)[i] = 0.01f * (float) ((i * 7) % 13 + 1);
    for (int64_t i = 0; i < ggml_nelements(b); i++)  ((float *) b->data)[i]  = 1.0f;
    // token 0: {2, -1-4, n_as}; token 1: {-1-4, 3, -1-2}
    const int32_t v[T][k] = { {2, -5, n_as}, {-5, 3, -3} };
    for (int t = 0; t < T; t++) for (int q = 0; q < k; q++) ((int32_t *) ids->data)[t*k + q] = v[t][q];
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_cpu_moe_online_reset();
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    int64_t sentinel = -1, total = -1;
    const int64_t * counts = ggml_cpu_moe_online_counts(il, &sentinel, &total);
    CHECK(counts != nullptr, "count: no counters");
    if (counts) {
        CHECK(counts[2] == 2, "count: expert 2 (computed once, skipped once) got %lld want 2", (long long) counts[2]);
        CHECK(counts[3] == 1, "count: expert 3 got %lld want 1", (long long) counts[3]);
        CHECK(counts[4] == 2, "count: expert 4 (skipped twice) got %lld want 2", (long long) counts[4]);
        CHECK(sentinel == 1, "count: sentinel got %lld want 1 (only the n_as slot)", (long long) sentinel);
        CHECK(total == 6, "count: total got %lld want 6", (long long) total);
    }
    for (int t = 0; t < T; t++) for (int q = 0; q < k; q++) {
        const float * row = (const float *) ((const char *) out->data + q*out->nb[1] + t*out->nb[2]);
        bool zero = true;
        for (int i = 0; i < n_ff; i++) zero = zero && row[i] == 0.0f;
        const bool want_zero = v[t][q] < 0 || v[t][q] >= n_as;
        CHECK(zero == want_zero, "mmid: token %d slot %d zero=%d want %d", t, q, (int) zero, (int) want_zero);
    }
    ggml_free(ctx);
}

int main() {
    setenv("LLAMA_MOE_HEAT_ONLINE", "1", 1);
    unsetenv("LLAMA_MOE_TAIL_SKIP");
    CHECK(llama_moe_tail_skip_rank() == 0, "default: rank %d, want 0 (off)", llama_moe_tail_skip_rank());
    test_rule();
    test_cpu_count();
    if (n_fail) { fprintf(stderr, "test-moe-tail-skip: %d FAILED\n", n_fail); return 1; }
    printf("test-moe-tail-skip: OK\n");
    return 0;
}
