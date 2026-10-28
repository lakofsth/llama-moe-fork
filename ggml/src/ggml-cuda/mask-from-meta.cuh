#pragma once

#include "common.cuh"

// fork (fork-chunk-loop, PREFILL-CHUNK-GPU-MASK): GGML_OP_MASK_FROM_META, ggml.h ggml_mask_from_meta
void ggml_cuda_op_mask_from_meta(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
