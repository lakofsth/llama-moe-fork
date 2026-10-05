# Fork invariants (llama-moe-fork)

One sentence per invariant; each names the test that pins it.

- **Q1 (fork-kernel-fusion, 2026-10-05).** A q8_1 copy that mul_mat_vec_q reuses (ggml_cuda_q8_1_cache) is
  never read after any byte of its source has been written — by a graph node of the same evaluation
  (invalidate) or by anything between evaluations (clear) — so reuse never changes a result.
  Pinned by `tests/test-cuda-q8-1-cache.cpp` (red with either mechanism removed, CUDA graphs on and off).
