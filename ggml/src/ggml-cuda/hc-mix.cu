// hc-mix.cu (fork, fork-kernel-fusion): GGML_OP_HC_MIX — a Qwen3.8-Flash-Next layer's hyper-connection read
// in three kernels instead of nine (rms_norm+mul, quantize, down mmvq, scale, silu, quantize, up mmvq,
// hc_pre, inject mmvf) plus hc_combine's scale/sigmoid/scale. Definition: ggml.h ggml_hc_mix.
//
//   hc_mix_prep  xn = x*w_norm/rms per stream (rms recomputed per block), F32 and q8_1 (as quantize_q8_1)
//   hc_mix_down  row k of w_down . xn with vec_dot_q8_0_q8_1 in mmvq's layout; lo = silu(scale*dot);
//                plus F32 inject partials (as mul_mat_vec_f)
//   hc_mix_up    lo -> q8_1 in shared memory, a thread per (gate row, Q8_0 block), mixed = scale*sum_h xn*sigmoid(gate)
//                (as hc_pre); block 0 finishes w = 2*sigmoid(scale*inject)
// The arithmetic mirrors the unfused path (q8_1 activations, the same dot), so the result differs from it only in
// summation order; v2 kept F32 activations and moved KLD 3x past the adaptive tier's (gate-20261005T0827*).
// Output (ggml.h ggml_hc_mix): mixed [n_embd, nt] then w [hc, nt], one buffer.
//
// Every kernel takes up to HC_MIX_MAX_T token columns at once, so a weight is read once per step for all of
// them (multi-slot decode, a speculative window); more columns run in chunks. The activations stay F32 —
// unlike the unfused path, nothing is quantized to q8_1 — so results differ from it in rounding only.
#include "hc-mix.cuh"
#include "vecdotq.cuh"

#define HC_MIX_MAX_T      8
#define HC_MIX_MAX_HC     8
#define HC_MIX_PREP_SLICE 256 // xn elements a prep block quantizes
#define HC_MIX_ROW_THREADS 128 // threads per down row (mmvq's layout: 4 per Q8_0 block, 32 blocks per pass)
#define HC_MIX_INJ_PARTS  16  // blocks per inject row; hc_mix_up sums their partials
#define HC_MIX_UP_I       8   // outputs i per up block

static __device__ __forceinline__ float hc_mix_block_sum(float v, float * sh) {
    v = warp_reduce_sum(v);
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    __syncthreads();
    if (lane == 0) {
        sh[warp] = v;
    }
    __syncthreads();
    v = lane < (int) (blockDim.x / WARP_SIZE) ? sh[lane] : 0.0f;
    return warp_reduce_sum(v);
}

// q8_1 of one 32-value block held one value per lane, exactly as quantize.cu quantize_q8_1
static __device__ __forceinline__ void hc_mix_quantize_lane(const float xi, block_q8_1 * y) {
    float amax = fabsf(xi), sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y->qs[threadIdx.x % WARP_SIZE] = q;
    if (threadIdx.x % WARP_SIZE == 0) {
        y->ds = make_half2(d, sum);
    }
}

// grid (hc*n_embd/SLICE, NT): xn = x*w_norm/rms(stream) for a slice — the rms recomputed per block from L2 —
// written as F32 (inject, hc_pre) and as q8_1 (the down rows)
static __global__ void hc_mix_prep(const float * __restrict__ x, const float * __restrict__ w_norm,
        float * __restrict__ xn, block_q8_1 * __restrict__ xq, const int n_embd, const int hd, const int64_t sx_t, const float eps) {
    __shared__ float sh[32];
    const int t  = blockIdx.y;
    const int e0 = blockIdx.x*HC_MIX_PREP_SLICE;
    const int h  = e0/n_embd;
    const float * xs = x + t*sx_t + (int64_t) h*n_embd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        ss += xs[i]*xs[i];
    }
    ss = hc_mix_block_sum(ss, sh);
    const float rs = rsqrtf(ss/n_embd + eps);
    const int e = e0 + threadIdx.x;
    const float v = x[t*sx_t + e]*rs*w_norm[e];
    xn[(int64_t) t*hd + e] = v;
    hc_mix_quantize_lane(v, xq + (int64_t) t*(hd/QK8_1) + e/QK8_1);
}

// blocks [0, r): row k of w_down against q8_1 xn with vec_dot_q8_0_q8_1, mmvq's thread layout; lo = silu(scale*dot).
// blocks [r, r + hc*PARTS): F32 inject partials (w_inject[h] . xn over 1/PARTS of the row), as mul_mat_vec_f
template <int NT>
static __global__ void hc_mix_down(const block_q8_0 * __restrict__ w_down, const float * __restrict__ w_inject,
        const block_q8_1 * __restrict__ xq, const float * __restrict__ xn, float * __restrict__ lo, float * __restrict__ inj_part,
        const int hd, const int r, const int hc, const float scale) {
    __shared__ float sh[32];
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
    }
    const int nb = hd/QK8_0;
    if ((int) blockIdx.x < r) {
        constexpr int tpb = QI8_0/VDR_Q8_0_Q8_1_MMVQ;      // threads per Q8_0 block
        constexpr int bpi = HC_MIX_ROW_THREADS/tpb;        // blocks per pass
        const block_q8_0 * row = w_down + (int64_t) blockIdx.x*nb;
        const int iqs = VDR_Q8_0_Q8_1_MMVQ*(threadIdx.x % tpb);
        for (int kbx = threadIdx.x/tpb; kbx < nb; kbx += bpi) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] += vec_dot_q8_0_q8_1(row, xq + (int64_t) t*nb + kbx, kbx, iqs);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = scale*hc_mix_block_sum(acc[t], sh);
            if (threadIdx.x == 0) {
                lo[t*r + blockIdx.x] = v/(1.0f + expf(-v));
            }
        }
    } else {
        const int q = blockIdx.x - r, h = q/HC_MIX_INJ_PARTS, part = q%HC_MIX_INJ_PARTS;
        const int j0 = (int) ((int64_t) hd*part/HC_MIX_INJ_PARTS), j1 = (int) ((int64_t) hd*(part + 1)/HC_MIX_INJ_PARTS);
        const float * wi = w_inject + (int64_t) h*hd;
        for (int j = j0 + (int) threadIdx.x; j < j1; j += blockDim.x) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] += wi[j]*xn[(int64_t) t*hd + j];
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = hc_mix_block_sum(acc[t], sh);
            if (threadIdx.x == 0) {
                inj_part[((int64_t) t*hc + h)*HC_MIX_INJ_PARTS + part] = v;
            }
        }
    }
}

// a block per HC_MIX_UP_I outputs i: its HC_MIX_UP_I*hc gate rows (i + h*n_embd) times nb = r/32 Q8_0 blocks, a thread
// per (row, block) — coalesced, and n_embd/HC_MIX_UP_I blocks to fill the card. lo is quantized to q8_1 in shared
// memory by every block (r values); the per-block dots are summed per row in a fixed order, then
// mixed[i] = scale*sum_h xn*sigmoid(gate), as hc_pre. Block 0 also finishes w = 2*sigmoid(scale*sum of the inject partials).
template <int NT>
static __global__ void hc_mix_up(const block_q8_0 * __restrict__ w_up, const float * __restrict__ lo,
        const float * __restrict__ xn, const float * __restrict__ inj_part, float * __restrict__ mixed, float * __restrict__ w_out,
        const int n_embd, const int hd, const int hc, const int r, const float scale) {
    extern __shared__ char smem[];
    const int nb   = r/QK8_1;
    const int rows = HC_MIX_UP_I*hc;
    block_q8_1 * loq  = (block_q8_1 *) smem;                           // [NT][nb]
    float      * part = (float *) (smem + NT*nb*sizeof(block_q8_1));   // [NT][rows][nb]
    for (int b = threadIdx.x/WARP_SIZE; b < NT*nb; b += blockDim.x/WARP_SIZE) {
        hc_mix_quantize_lane(lo[(b/nb)*r + (b%nb)*QK8_1 + threadIdx.x % WARP_SIZE], loq + b);
    }
    for (int q = threadIdx.x; blockIdx.x == 0 && q < NT*hc; q += blockDim.x) {
        float v = 0.0f;
        for (int p = 0; p < HC_MIX_INJ_PARTS; ++p) {
            v += inj_part[(int64_t) q*HC_MIX_INJ_PARTS + p];
        }
        const float u = scale*v;
        w_out[q] = 2.0f*(1.0f/(1.0f + expf(-u)));
    }
    __syncthreads();
    const int rl  = threadIdx.x/nb;   // row in this block: il*hc + h
    const int kbx = threadIdx.x%nb;
    const int i0  = blockIdx.x*HC_MIX_UP_I;
    {
        const int il = rl/hc, h = rl%hc, i = i0 + il;
        float s[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            s[t] = 0.0f;
        }
        if (i < n_embd) {
            const block_q8_0 * row = w_up + (int64_t) (i + h*n_embd)*nb;
#pragma unroll
            for (int iqs = 0; iqs < QI8_0; iqs += VDR_Q8_0_Q8_1_MMVQ) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    s[t] += vec_dot_q8_0_q8_1(row, loq + t*nb + kbx, kbx, iqs);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            part[(t*rows + rl)*nb + kbx] = s[t];
        }
    }
    __syncthreads();
    for (int q = threadIdx.x; q < NT*HC_MIX_UP_I; q += blockDim.x) {
        const int t = q/HC_MIX_UP_I, il = q%HC_MIX_UP_I, i = i0 + il;
        if (i < n_embd) {
            float v = 0.0f;
            for (int h = 0; h < hc; ++h) {
                const float * pr = part + (t*rows + il*hc + h)*nb;
                float g = 0.0f;
                for (int b = 0; b < nb; ++b) {
                    g += pr[b];
                }
                v += xn[(int64_t) t*hd + h*n_embd + i]*(1.0f/(1.0f + expf(-g)));
            }
            mixed[t*n_embd + i] = scale*v;
        }
    }
}

template <int NT>
static void hc_mix_launch(const float * x, const float * w_norm, const block_q8_0 * w_down, const block_q8_0 * w_up,
        const float * w_inject, float * xn, block_q8_1 * xq, float * lo, float * inj_part, float * mixed, float * w_out,
        int n_embd, int hc, int r, int64_t sx_t, float eps, float scale, cudaStream_t stream) {
    const int hd = n_embd*hc;
    hc_mix_prep<<<dim3(hd/HC_MIX_PREP_SLICE, NT), HC_MIX_PREP_SLICE, 0, stream>>>(x, w_norm, xn, xq, n_embd, hd, sx_t, eps);
    hc_mix_down<NT><<<r + hc*HC_MIX_INJ_PARTS, HC_MIX_ROW_THREADS, 0, stream>>>(w_down, w_inject, xq, xn, lo, inj_part, hd, r, hc, scale);
    const int nb = r/QK8_1, threads = HC_MIX_UP_I*hc*nb;
    const size_t smem = NT*nb*sizeof(block_q8_1) + (size_t) NT*HC_MIX_UP_I*hc*nb*sizeof(float);
    hc_mix_up<NT><<<(n_embd + HC_MIX_UP_I - 1)/HC_MIX_UP_I, threads, smem, stream>>>(
        w_up, lo, xn, inj_part, mixed, w_out, n_embd, hd, hc, r, scale);
}

bool ggml_cuda_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * w_down = op->src[2];
    const ggml_tensor * w_up   = op->src[3];
    const int64_t n_embd = op->src[0]->ne[0], hc = op->src[0]->ne[1];
    return w_down->type == GGML_TYPE_Q8_0 && w_up->type == GGML_TYPE_Q8_0 &&
           w_down->ne[1] % QK8_0 == 0 && (size_t) HC_MIX_MAX_T*w_down->ne[1]*sizeof(float) <= 48*1024 &&
           hc <= HC_MIX_MAX_HC && n_embd % HC_MIX_PREP_SLICE == 0 &&
           (HC_MIX_UP_I*hc*(w_down->ne[1]/QK8_0)) % WARP_SIZE == 0 && HC_MIX_UP_I*hc*(w_down->ne[1]/QK8_0) <= 1024;
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * w_norm   = dst->src[1];
    const ggml_tensor * w_down   = dst->src[2];
    const ggml_tensor * w_up     = dst->src[3];
    const ggml_tensor * w_inject = dst->src[4];

    const int n_embd = x->ne[0];
    const int hc     = x->ne[1];
    const int nt     = x->ne[2];
    const int r      = w_down->ne[1];

    const float eps   = ggml_get_op_params_f32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);

    const int64_t sx_t = x->nb[2]/sizeof(float);

    cudaStream_t stream = ctx.stream();
    const int chunk = nt < HC_MIX_MAX_T ? nt : HC_MIX_MAX_T;
    ggml_cuda_pool_alloc<float> xn(ctx.pool(), (size_t) chunk*n_embd*hc);
    ggml_cuda_pool_alloc<char>  xq(ctx.pool(), (size_t) chunk*(n_embd*hc/QK8_1)*sizeof(block_q8_1));
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) chunk*r);
    ggml_cuda_pool_alloc<float> inj(ctx.pool(), (size_t) chunk*hc*HC_MIX_INJ_PARTS);

    for (int t0 = 0; t0 < nt; t0 += HC_MIX_MAX_T) {
        const int n = nt - t0 < HC_MIX_MAX_T ? nt - t0 : HC_MIX_MAX_T;
        const float * xp = (const float *) x->data + t0*sx_t;
        float       * mp = (float *) dst->data + (int64_t) t0*n_embd;                    // mixed [n_embd, nt]
        float       * wp = (float *) dst->data + (int64_t) nt*n_embd + (int64_t) t0*hc;  // w [hc, nt]
        const float * wn = (const float *) w_norm->data;
        const block_q8_0 * wd = (const block_q8_0 *) w_down->data;
        const block_q8_0 * wu = (const block_q8_0 *) w_up->data;
        const float * wi = (const float *) w_inject->data;
        switch (n) {
            case 1: hc_mix_launch<1>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            case 2: hc_mix_launch<2>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            case 3: hc_mix_launch<3>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            case 4: hc_mix_launch<4>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            case 5: hc_mix_launch<5>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            case 6: hc_mix_launch<6>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            case 7: hc_mix_launch<7>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
            default: hc_mix_launch<8>(xp, wn, wd, wu, wi, xn.get(), (block_q8_1 *) xq.get(), lo.get(), inj.get(), mp, wp, n_embd, hc, r, sx_t, eps, scale, stream); break;
        }
    }
}
