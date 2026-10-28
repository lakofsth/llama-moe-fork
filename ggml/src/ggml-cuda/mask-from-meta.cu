// fork (fork-chunk-loop, PREFILL-CHUNK-GPU-MASK): GGML_OP_MASK_FROM_META on CUDA. The rule is the CPU reference's
// (ggml-cpu/ops.cpp ggml_mask_meta_keep), element for element; the values are exactly 0 or -INFINITY in F16 or F32.
#include "mask-from-meta.cuh"

static __device__ __forceinline__ bool mask_meta_keep(const int4 c, const int4 t, const int mode) {
    if (((((unsigned int) c.y) >> (t.y & 31)) & 1u) == 0) {
        return false;
    }
    switch (mode) {
        case GGML_MASK_META_ANY:       return true;
        case GGML_MASK_META_CAUSAL:    return c.x <= t.x;
        case GGML_MASK_META_CAUSAL_2D: return c.x != t.x ? c.x < t.x : !(c.w > t.w || (c.w == t.w && c.z > t.z));
        case GGML_MASK_META_STRICT:    return c.x < t.x;
    }
    return false;
}

template <typename T>
static __global__ void mask_from_meta_kernel(
        const int4 * __restrict__ cells, const int4 * __restrict__ toks, T * __restrict__ dst,
        const int64_t n_cells, const int64_t n_tps, const int64_t nb1, const int64_t nb3, const int mode, const T keep, const T drop) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (j >= n_cells) {
        return;
    }
    const int4 t = toks[s*n_tps + i];
    const int4 c = cells[s*n_cells + j];
    T * row = (T *) ((char *) dst + s*nb3 + i*nb1);
    row[j] = mask_meta_keep(c, t, mode) ? keep : drop;
}

void ggml_cuda_op_mask_from_meta(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cells = dst->src[0];
    const ggml_tensor * toks  = dst->src[1];

    GGML_ASSERT(cells->type == GGML_TYPE_I32 && toks->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(cells) && ggml_is_contiguous(toks));

    const int     mode    = ggml_get_op_params_i32(dst, 0);
    const int64_t n_cells = dst->ne[0];
    const int64_t n_tps   = dst->ne[1];
    const int64_t n_s     = dst->ne[3];

    GGML_ASSERT(n_tps <= 65535 && n_s <= 65535);

    cudaStream_t stream = ctx.stream();

    const int    nt = 256;
    const dim3   grid((unsigned) ((n_cells + nt - 1)/nt), (unsigned) n_tps, (unsigned) n_s);

    if (n_cells == 0 || n_tps == 0 || n_s == 0) {
        return;
    }

    if (dst->type == GGML_TYPE_F16) {
        mask_from_meta_kernel<half><<<grid, nt, 0, stream>>>(
            (const int4 *) cells->data, (const int4 *) toks->data, (half *) dst->data, n_cells, n_tps, dst->nb[1], dst->nb[3], mode,
            __float2half(0.0f), __float2half(-INFINITY));
    } else {
        mask_from_meta_kernel<float><<<grid, nt, 0, stream>>>(
            (const int4 *) cells->data, (const int4 *) toks->data, (float *) dst->data, n_cells, n_tps, dst->nb[1], dst->nb[3], mode,
            0.0f, -INFINITY);
    }
    CUDA_CHECK(cudaGetLastError());
}
