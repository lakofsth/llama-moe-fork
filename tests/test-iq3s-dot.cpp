// IQ3S-512 (tests/INVARIANTS.md): the CPU iq3_s x q8_K dot product returns bit for bit the value of the AVX2
// kernel's arithmetic, whichever x86 SIMD path ggml-cpu compiled. The reference below is that arithmetic written
// out in scalar: exact int32 lane sums, one fused multiply-add of d per block into 8 float lanes, hsum_float_8's
// order. Without an x86 AVX2 kernel they are free to sum differently, so the test then only reports and passes.
#include "ggml.h"
#include "ggml-cpu.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../ggml/src/ggml-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t rng = 88172645463325252ull;
static uint64_t xr() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static float ref_dot(int n, const block_iq3_s * x, const block_q8_K * y) {
    float acc[8] = {0};
    for (int i = 0; i < n / QK_K; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        int32_t lane[8] = {0};
        for (int b = 0; b < QK_K/32; ++b) {
            const int ls = 2*((x[i].scales[b/2] >> (4*(b % 2))) & 0xf) + 1;
            for (int l = 0; l < 8; ++l) {
                const int g = x[i].qs[8*b + l] | (((x[i].qh[b] >> l) & 1) << 8);
                const uint8_t * grid = (const uint8_t *)(iq3s_grid + g);
                int32_t sum = 0;
                for (int k = 0; k < 4; ++k) {
                    const int e = 32*b + 4*l + k;
                    int8_t q = y[i].qs[e];
                    if ((x[i].signs[e/8] >> (e % 8)) & 1) q = (int8_t)(-q);  // wraps -128 as the SIMD negate does
                    sum += grid[k] * q;
                }
                lane[l] += ls * sum;
            }
        }
        for (int l = 0; l < 8; ++l) acc[l] = std::fma(d, (float)lane[l], acc[l]);
    }
    const float r0 = acc[4] + acc[0], r1 = acc[5] + acc[1], r2 = acc[6] + acc[2], r3 = acc[7] + acc[3];
    return (r0 + r2) + (r1 + r3);
}

int main() {
    ggml_cpu_init();
    const auto * tt = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S);
    if (tt->vec_dot_type != GGML_TYPE_Q8_K) { fprintf(stderr, "iq3_s vec_dot_type is not q8_K\n"); return 1; }
    const int ns[] = {256, 512, 2560, 4096};
    long trials = 0, bad = 0;
    for (int t = 0; t < 20000; ++t) {
        const int n = ns[t % 4], nb = n / QK_K;
        std::vector<block_iq3_s> x(nb);
        std::vector<block_q8_K>  y(nb);
        uint8_t * px = (uint8_t *)x.data();
        for (size_t k = 0; k < sizeof(block_iq3_s) * nb; ++k) px[k] = (uint8_t)xr();
        for (int b = 0; b < nb; ++b) {
            x[b].d = ggml_fp32_to_fp16((float)((xr() % 2000) / 1000.0 + 0.001));
            y[b].d = (float)((xr() % 1000) / 997.0 + 0.01);
            for (int k = 0; k < QK_K; ++k) y[b].qs[k] = (t % 7 == 0 && k % 5 == 0) ? (int8_t)-128 : (int8_t)xr();
        }
        float got = 0.0f;
        tt->vec_dot(n, &got, 0, x.data(), 0, y.data(), 0, 1);
        const float want = ref_dot(n, x.data(), y.data());
        ++trials;
        if (memcmp(&got, &want, sizeof(float)) != 0 && bad++ < 5)
            fprintf(stderr, "n=%d: kernel %.9g, reference %.9g\n", n, got, want);
    }
#if defined(__x86_64__)
    // The x86 kernels are chosen when ggml-cpu is compiled; these report what that build has.
    if (ggml_cpu_has_avx2()) {
        printf("test-iq3s-dot: %ld trials, %ld not bit-identical (ggml-cpu AVX2=%d AVX512=%d AVX512_VNNI=%d)\n", trials, bad,
               ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni());
        return bad != 0;
    }
#endif
    printf("test-iq3s-dot: %ld trials, %ld differ from the AVX2 order (no x86 AVX2 kernel: reported, not gated)\n", trials, bad);
    return 0;
}
