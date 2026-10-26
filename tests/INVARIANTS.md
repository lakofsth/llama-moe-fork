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
- **S1 (fork-kernel-fusion, 2026-10-08).** The cells a qwen4exp QSA query attends are a function of its block scores
  alone: its top indexer_top_k/ratio whole blocks among the whole blocks wholly before its tail, plus every cell of
  its tail (positions tail_start..q), and nothing else. No block other than the tail is ever attended in part, so no
  tie among the cells of one block can change what is attended; blocks whose scores tie at the cut (the indexer
  score passes a relu, so many sit at exactly 0) are taken lowest block first by the CUDA radix-select top-k
  (top_k_radix_ties). Test: tests/test-qsa-select.cpp (llama_qsa_select_cells; case 6, CUDA only, the tie at the
  cut); end to end, prompt passes of 1000-35000 tokens at temperature 0 repeat bit-exact under the deterministic config.
- **M1 (fork-kernel-fusion, 2026-10-09).** With LLAMA_MLOCK_TENSORS set to a comma-separated list of name substrings,
  every host-resident model tensor whose name contains one of them has its page-aligned byte range mlock'ed from load
  until the model is freed, so the kernel never evicts those pages to read them back from disk at prompt time; a
  tensor matching none, and every tensor when the variable is unset, is left unlocked; a range that fails to lock is
  reported with its errno, never silently dropped. Pinned by `tests/test-mlock-tensors.cpp` (VmLck rises by exactly
  the matched ranges; red with the mlock call removed or with the name match inverted).

- **MMQ-PAD (fork-strata-ports, 2026-10-11; widened 2026-10-22).** Every buffer an MMQ launch's last column tile reads
  J entries wide is padded by at least the largest column tile J that `mul_mat_q_switch_J` can select
  (`MMQ_J_SELECT_MAX`), because that tile reads J entries from its first column whatever the real column count.
  Two buffers are read that way. (a) The quantized src1, padded in `block_q8_1_mmq`. (b) The MUL_MAT_ID `ids_dst`
  map, padded by `MMQ_J_SELECT_MAX - 1` int32. (b) is loaded into `ids_dst_shared` at the three
  `ids_dst_shared[j] = ids_dst[col_low + jt*J + j]` sites in mmq.cuh, guarded only by j < J; the last expert's last
  tile runs past `ne_get_rows`. (b) was found 2026-10-22 by comparing against upstream fc9ce6b9d (#29953), which
  pads ids_dst by J_best-1. Sites derived by `grep -n 'ids_dst\[' ggml/src/ggml-cuda/mmq.cuh` (the three loads;
  the write-backs skip j > j_max = col_diff - jt*J - 1, so no store uses a padded entry) and its single allocation
  in mmq.cu. mmf.cuh reads its compact ids_dst under global_j < ncols_expert and is outside the class. (b) is pinned
  by compute-sanitizer memcheck over `test-backend-ops test -o MUL_MAT_ID` on a GGML_CUDA_NO_VMM=ON build (the
  legacy pool makes each buffer its own allocation, which memcheck tracks; VMM sub-allocations it cannot see):
  llm-config eval/fork-kernel-fusion/diag/memcheck-ids.sh + build-novmm.sh. Red on the parent f24965e3c: 1,235
  errors, invalid 4-byte reads in mul_mat_q 1-5 bytes past a 256-byte allocation. Green on fbd98f26e: 929/929
  MUL_MAT_ID OK, 0 invalid reads (/data/two-card-bench/mmq-pad/memcheck-20261022T130106Z). For (a), both
  allocation sites (`ggml_cuda_mul_mat_q`, its plain
  and MUL_MAT_ID paths) pad by `ggml_cuda_mmq_get_J_pad`, derived from `MMQ_J_SELECT_MAX`, the same bound the
  selection loop uses. Sites derived by `grep -n "sizeof(block_q8_1_mmq)" ggml/src/ggml-cuda/mmq.cu` (the two
  `nbytes_src1_q8_1` terms). Pinned end to end, not by a unit test: crash-repro2/repro-order.sh with order 120
  faults (Xid 31 FAULT_PDE at the VMM pool's mapping boundary) on the unpadded build and must not on this one;
  residual — no unit-level pin, because the fault needs the buffer at the top of the pool within J blocks of a
  2 MiB mapping boundary.

- **ADAPT-ASYNC (fork-strata-ports, 2026-10-22).** With LLAMA_MOE_ADAPT_ASYNC=1 the adaptive tier's expert copies
  run on a copy stream, and a packed slot is GPU-resident in NO map (map_cpu, map_gpu, map_gpu_pp, mask_cpu,
  mask_gpu) from the round that evicts its occupant until the round that admits its new occupant. Each round runs in
  three steps, in this order:
  1. **Admit.** The compute stream waits on the previous round's copy-done event, then that round's in-flight slots
     are cleared and their maps rebuilt.
  2. **Select.**
  3. **Evict + copy.** The new swaps' slots are marked in flight and their maps rebuilt. The copy-free event is
     recorded on the compute stream, the copy stream waits on it, then the copies run and the copy-done event is
     recorded.
  So no GPU kernel reads a slot while it is being written, and an expert is never resident in one map and absent
  from the other. Rounds run only in the trunk context (LLAMA_CONTEXT_TYPE_DEFAULT), whose compute stream orders
  every packed-slot read; the MTP drafter (graph_mtp) reads none, and a round run from it would order nothing. This
  holds for the synchronous tier too: its drain is the calling context's synchronize(), which does not reach the trunk.
  Found by the first A/B (async-ab run-20261022T133456Z), which logged two copy streams on CUDA0, one per context. Design: ~/docs/fork-async-adapt-swaps-design-2026-10-22.md. Pinned by
  `tests/test-moe-adapt-async.cpp`, which replays rounds through a fake IO that checks this ordering and the maps'
  contents.

- **IQ3S-512 (fork-strata-ports, 2026-10-26).** On x86, `ggml_vec_dot_iq3_s_q8_K` returns bit for bit the value of
  the AVX2 kernel's arithmetic — exact int32 lane sums (lane l of sub-block b = (2·scale_b+1) · Σ of its 4 signed
  grid·q8 products), one fused multiply-add of `d` per block into 8 float lanes, and `hsum_float_8`'s order — whether
  it is compiled for AVX2 or for AVX-512 (F+BW+VNNI), so the AVX-512 kernel changes speed and never a result.
  Pinned by `tests/test-iq3s-dot.cpp` (that lane-structured scalar reference over random blocks, q8 = -128 included;
  red on a wrong scale nibble, high index bit or sign mask in the AVX-512 kernel).
