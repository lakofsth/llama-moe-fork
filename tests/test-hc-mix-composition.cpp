// test-hc-mix-composition — GGML_OP_HC_MIX against the ops it replaces, composed exactly as
// src/models/qwen4exp.cpp builds them (fork-kernel-fusion).
//
// test-backend-ops checks the CUDA kernels against the CPU reference, but both were written from one reading
// of the model; a misreading would pass it. This test builds the UNFUSED graph — rms_norm, mul, mul_mat,
// scale, silu, mul_mat, dsv4_hc_pre_gated, mul_mat (inject), scale, sigmoid, scale — and ggml_hc_mix on the
// CPU backend from the same data, and compares mixed and w. The unfused path quantizes its activations
// (q8_0 vec_dot on the CPU), so the two differ by that rounding: a reading error is orders of magnitude
// larger. Real Flash-Next sizes; n_tokens 1 to 8 (a decode step of up to 8 slots, or a speculative window). HC_MIX_TEST_DEVICE=CUDA0 runs both graphs on that device
// instead (the CUDA kernels mirror the unfused arithmetic, so there the two should agree to summation order).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;

static double nmse(const float * a, const float * b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += (double) (a[i] - b[i])*(a[i] - b[i]);
        den += (double) b[i]*b[i];
    }
    return den > 0 ? num/den : num;
}

static std::vector<char> make_data(ggml_type type, int64_t ne0, int64_t nrows, float lo, float hi, unsigned seed) {
    srand(seed);
    std::vector<float> f(ne0*nrows);
    for (float & v : f) v = lo + (hi - lo)*((float) rand()/RAND_MAX);
    std::vector<char> out(ggml_row_size(type, ne0)*nrows);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), f.data(), out.size());
    } else {
        ggml_quantize_chunk(type, f.data(), out.data(), 0, nrows, ne0, nullptr);
    }
    return out;
}

static void run(ggml_backend_t be, int64_t nt, double tol) {
    const int64_t n_embd = 2560, hc = 4, r = 320, hd = n_embd*hc;
    const float eps = 1e-6f;
    ggml_init_params ip = { 128*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  n_embd, hc, nt);
    ggml_tensor * w_norm   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  n_embd, hc);
    ggml_tensor * w_down   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hd, r);
    ggml_tensor * w_up     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, r, hd);
    ggml_tensor * w_inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  hd, hc);

    // unfused, as qwen4exp.cpp build_hc_mix (fused_dsv4_hc_pre) + build_hc_combine's weights
    ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w_norm);
    xn = ggml_reshape_2d(ctx, xn, hd, nt);
    ggml_tensor * lo = ggml_mul_mat(ctx, w_down, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f/hc));
    ggml_tensor * gate = ggml_mul_mat(ctx, w_up, lo);
    ggml_tensor * mixed = ggml_dsv4_hc_pre_gated(ctx, ggml_reshape_3d(ctx, xn, n_embd, hc, nt),
            ggml_reshape_3d(ctx, gate, n_embd, hc, nt), 1.0f/hc);
    ggml_tensor * inject = ggml_mul_mat(ctx, w_inject, xn);
    ggml_tensor * w = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f/hc)), 2.0f);

    ggml_tensor * fused = ggml_hc_mix(ctx, x, w_norm, w_down, w_up, w_inject, eps, 1.0f/hc);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mixed);
    ggml_build_forward_expand(gf, w);
    ggml_build_forward_expand(gf, fused);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    auto set = [](ggml_tensor * t, const std::vector<char> & d) { ggml_backend_tensor_set(t, d.data(), 0, d.size()); };
    // magnitudes as in the model: unit-scale residual streams, gains near 1, small projections
    set(x,        make_data(GGML_TYPE_F32,  n_embd, hc*nt, -2.0f, 2.0f, 1));
    set(w_norm,   make_data(GGML_TYPE_F32,  n_embd, hc, 0.5f, 1.5f, 2));
    set(w_down,   make_data(GGML_TYPE_Q8_0, hd, r, -0.05f, 0.05f, 3));
    set(w_up,     make_data(GGML_TYPE_Q8_0, r, hd, -0.2f, 0.2f, 4));
    set(w_inject, make_data(GGML_TYPE_F32,  hd, hc, -0.05f, 0.05f, 5));
    ggml_backend_graph_compute(be, gf);

    std::vector<float> f(ggml_nelements(fused)), m(ggml_nelements(mixed)), ww(ggml_nelements(w));
    ggml_backend_tensor_get(fused, f.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(mixed, m.data(), 0, ggml_nbytes(mixed));
    ggml_backend_tensor_get(w,     ww.data(), 0, ggml_nbytes(w));
    const double e_mixed = nmse(f.data(),             m.data(),  n_embd*nt);
    const double e_w     = nmse(f.data() + n_embd*nt, ww.data(), hc*nt);
    const bool ok = e_mixed < tol && e_w < tol;
    printf("%s %s n_tokens=%lld: nmse mixed %.3g, w %.3g (tol %.0e)\n", ok ? "ok  " : "FAIL",
           ggml_backend_name(be), (long long) nt, e_mixed, e_w, tol);
    fails += !ok;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    const char * dev_name = getenv("HC_MIX_TEST_DEVICE");
    ggml_backend_t be = nullptr;
    double tol = 1e-4; // CPU: the fused reference (F32 activations) vs the unfused path's q8_0 rounding
    if (dev_name && *dev_name) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
        if (!dev) {
            printf("skip: no device %s\n", dev_name);
            return 0;
        }
        be = ggml_backend_dev_init(dev, nullptr);
        tol = 1e-8; // the CUDA kernels mirror the unfused arithmetic (q8_1 activations, the same dot)
    } else {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    for (int64_t nt : {1, 2, 3, 4, 6, 8}) {
        run(be, nt, tol);
    }
    ggml_backend_free(be);
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
