# Fork invariants (llama-moe-fork)

One sentence per invariant; each names the test that pins it.

- **Q1 (fork-kernel-fusion, 2026-10-05).** A q8_1 copy that mul_mat_vec_q reuses (ggml_cuda_q8_1_cache) is
  never read after any byte of its source has been written — by a graph node of the same evaluation
  (invalidate) or by anything between evaluations (clear) — so reuse never changes a result.
  Pinned by `tests/test-cuda-q8-1-cache.cpp` (red with either mechanism removed, CUDA graphs on and off).

- **Q2 (fork-kernel-fusion, 2026-10-05).** GGML_OP_HC_MIX computes what the unfused qwen4exp hyper-connection
  read computes (rms_norm, mul, the down/up/inject matvecs, scale, silu, hc_pre, and hc_combine's
  2*sigmoid(inject/hc)) with the same arithmetic on CUDA — q8_1 activations, vec_dot_q8_0_q8_1 — so the two
  differ in summation order only. Pinned by `tests/test-hc-mix-composition.cpp` (CUDA nmse <= 1e-8 for 1..8
  tokens; red on a kernel without its sigmoid gate) and `test-backend-ops -o HC_MIX` (CUDA vs the CPU reference).

- **H1 (fork-kernel-fusion, 2026-10-06).** With LLAMA_MOE_HEAT_FILL=1 the heat-split packer fills every packable
  layer to its per-layer cap while the device budget lasts — zero-heat experts after every positive one — and
  its selection contains the selection made without it; without the variable the selection is the packer's
  previous one. Pinned by `tests/test-moe-heat-select.cpp` (red with the zero-heat candidates dropped under fill).
