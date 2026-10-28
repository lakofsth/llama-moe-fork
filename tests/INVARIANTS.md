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
  <local instrument> + build-novmm.sh. Red on the parent f24965e3c: 1,235
  errors, invalid 4-byte reads in mul_mat_q 1-5 bytes past a 256-byte allocation. Green on fbd98f26e: 929/929
  MUL_MAT_ID OK, 0 invalid reads (<bench>/mmq-pad/memcheck-20261022T130106Z). For (a), both
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
  Found by the first A/B (async-ab run-20261022T133456Z), which logged two copy streams on CUDA0, one per context. Design: <design note>. Pinned by
  `tests/test-moe-adapt-async.cpp`, which replays rounds through a fake IO that checks this ordering and the maps'
  contents.

- **IQ3S-512 (fork-strata-ports, 2026-10-26).** On x86, `ggml_vec_dot_iq3_s_q8_K` returns bit for bit the value of
  the AVX2 kernel's arithmetic — exact int32 lane sums (lane l of sub-block b = (2·scale_b+1) · Σ of its 4 signed
  grid·q8 products), one fused multiply-add of `d` per block into 8 float lanes, and `hsum_float_8`'s order — whether
  it is compiled for AVX2 or for AVX-512 (F+BW+VNNI), so the AVX-512 kernel changes speed and never a result.
  Pinned by `tests/test-iq3s-dot.cpp` (that lane-structured scalar reference over random blocks, q8 = -128 included;
  red on a wrong scale nibble, high index bit or sign mask in the AVX-512 kernel).

- **KV-SEQRM (fork-strata-ports, 2026-10-26).** `llama_kv_cache::seq_rm` for one sequence removes it from exactly
  the cells a scan of every cell would (those holding it at a position in [p0, p1)) and moves the head to the same
  cell, found through the sequence's (pos, cell) index `seq_pos` in time proportional to the cells removed, not to
  the cache's size (262,144 cells on the served recipe, scanned per MTP rollback: 1.3% of the decode thread in
  decode-pace-20261024T233039Z). Pinned by `tests/test-kv-seqrm.cpp` (random caches under every cell mutator,
  duplicate positions included, against the old scan; red on an inclusive p1, a p0 off by one, or the last
  emptied cell in position order taken as the head).

- **PREFILL-CHUNK (fork-chunk-loop, 2026-10-27).** With LLAMA_PREFILL_CHUNK=C (C >= 2) a prompt-only batch of more than
  one ubatch runs layer-major over chunks of C ubatches (LLAMA_PREFILL_CHUNK_GROUP=G layers per pass, G = 1 or a
  multiple of the layer-type period), and every pass over ubatch k sees the memory metadata apply() left for k
  (attention and indexer cells and heads, recurrent cells, head, n, rs_z, used and the rollback index rs_idx), with
  each ubatch applied exactly once; so its logits, h_nextn rows and memory state are bit for bit those of the
  micro-batch-major path. Unset or C < 2 is the micro-batch path unchanged; a batch in which a sequence's first token
  requests output (a generating sequence), or with pooling, embeddings or per-layer input embeddings, takes the
  micro-batch path. Backend samplers take the chunk path since P2 (28 Oct; amended): the last group's pass builds the
  sampler graph once per ubatch in token order and the dist sampler draws in set_input only for a sequence with an
  output in the ubatch, so the sampled tokens are the micro-batch path's (scenario `sampling`; red with the old
  refusal put back, and with one extra set_inputs on the last group, 7/63 on the sampled tokens). Pinned by `tests/test-prefill-chunk.cpp` (a random 12-layer qwen4exp MoE
  with PLE and QSA on the CPU: 7 (C, G) configurations x 8 scenarios, incl. 3 sequences per batch, an MTP-style
  rollback with rs_idx != 0 on entry, and a reused recurrent cell; it compares the logits, h_nextn, the
  llama_state_get_data bytes, the apply() count and every named graph input each pass saw; 8 scenarios incl. one
  KV stream, 56 checks). Each mechanism reverted alone turns it red (tests/prefill-chunk-mutate.py, 27 Oct, on
  25225562c): KV cells/heads swap 49/56 (inputs only: on the CPU the logits do not see a wider n_kv, so the input
  trace is what pins this seam), recurrent cells/head/n/used/rs_z swap 28/56, rs_idx copied in rather than swapped
  18/56, rs_idx advanced in chunk_save 21/56, apply() per pass instead of the swap 49/56 (apply count 74 != 19),
  n_kv recomputed in chunk_swap_in 12/56, the residual write 42/56, the generating-batch policy 7/56, and the
  mem_hybrid kq_mask ->buffer guard aborts (GGML_ASSERT(buffer)). GPU bit identity is gated separately by
  <bench>/chunk-loop/gate-bits.sh (copy tests/prefill-chunk-gate-bits.sh; dumps with tools/prefill-dump).

- **P1 (fork-kernel-fusion, 2026-10-08).** Under LLAMA_MOE_PREFETCH_PP=R a MUL_MAT_ID whose ids are compact
  (`ffn_moe_ids_pf`, built from map_cpu_pf) reads exactly the experts the stock partial upload would give it for
  the original ids: the scheduler's slot holds the registered plan in plan order, the slot is uploaded only after the
  matmul R entries earlier has finished reading it, and the matmul runs only after its own upload — so outputs equal
  the stock path's bit for bit, across new graphs, reused graphs and plan changes. Every site that writes map_cpu
  writes map_cpu_pf and the plan from the same values (llama_moe_split_set_pf). Pinned by `tests/test-moe-prefetch.cpp`
  (R = 1..4; red with the upload's source offset taken from the slot index, with the compute stream's wait on the
  upload removed, and with the copy stream's wait on the earlier read removed).

- **PREFILL-CHUNK-PF (fork-chunk-loop P2, 2026-10-28; amended 28 Oct after the first GPU gate).** Under
  LLAMA_PREFILL_CHUNK with LLAMA_MOE_PREFETCH_PP=R, every compact-id MUL_MAT_ID reads a ring slot holding its weight's
  plan at the current plan generation (bumped by every ggml_backend_sched_moe_pf_plan: load, repin, adaptive swap),
  and a slot is written only after the last read of what it held, so the outputs are the stock prefetch's (P1) bit
  for bit. Chunk mode is on only for one-layer passes (G = 1) of chunks of several ubatches; across a layer's passes:
  with R >= its entries the layer is uploaded once per chunk in the stock layout (the chunk's m-th weight in slot
  m % R, the stock slot sizes); with fewer (the served separate gate/up/down at R = 2) its R - 1 largest entries keep
  slots 0..R-2 by size rank and the rest stream through slot R-1, so it is uploaded at most E + (E - R + 1) times per
  chunk of passes, never more than the stock count, and each slot is its size rank's largest entry. The next layer's
  entries go up as soon as the ring frees a slot. Pinned by `tests/test-moe-pf-ring.cpp` (the planner,
  ggml/src/ggml-moe-pf-ring.h, replayed in the scheduler's issue order over E = 2, 3 (sizes by kind and per-layer plan),
  R = 1..6, C = 2, 4, 8, G = 1, 4, short last ubatches, plan changes inside chunks and several decodes in one ring,
  no upload overwritten before it is read: 1449 checks; it prints the gate's expected counters, 1767 slot reads (the
  measured stock count) and 1271 uploads at C = 8, R = 2). Each mechanism reverted alone is red
  (tests/prefill-chunk-mutate.py, on the commit after 8eeee074a; of 1449): the n <= R layout's slot reuse 288,
  generation in the match 90, after-read hints only on the last pass 114, start-of-pass hints only into unread slots
  246, its hint cap (hints for a wider graph would sit where the wide layout does not look: 75 wasted uploads in the
  served prediction) 3, the m % R layout 17; the wide layout's slot reuse 41, no growing a slot the graph reads 21,
  hints only on the last pass 41, uploads after the slot's last read 29, slots by size rank 15. Found by the GPU gate of 27 Oct (gate-p2-20261027T221645Z: uploads 1767 = reads 1767): the first planner took a
  pass only if all its entries fit, and its test asserted that E = 3, R = 2 falls back; the served-shape check (C = 8,
  R = 2, E = 3: uploads <= 80% of reads, no fallback) is the one it fails. Residual: the planner-to-compute_splits
  seam (events, the slot pointer swap, the copy stream) runs only on CUDA; gated by
  <bench>/chunk-loop/gate-bits-p2.sh (copy tests/prefill-chunk-gate-bits-p2.sh), with a
  LLAMA_MOE_PF_TRACE=1 arm.

- **PREFILL-CHUNK-MIXED (fork-chunk-loop P3a, 2026-10-28).** With LLAMA_PREFILL_CHUNK set (C >= 2), a server batch that
  carries any generating slot's token (sampled or draft) holds at most LLAMA_PREFILL_CHUNK_MIXED_BATCH tokens in all
  (default 2048, the served -b; 0 = no cap), so while a slot generates, every batch is the one the served recipe
  builds and its step time, hence the generating slots' per-token latency, is the served one. Amended 28 Oct after the
  server gate (server-20261028T000433Z, peak-a +14 s TTFT): likewise, once a slot's tokens bring it to its prompt's
  last n_ubatch + 4 tokens (the tail the server's checkpoints split off into their own batches) or to its end, the
  slots after it add only up to the cap, since that slot's first token waits for those batches. Any other batch fills
  to n_batch (the -b 8192 that lets a chunk hold 8 ubatches). Unset (or C < 2), the batch is filled as before. Pinned
  by `tests/test-server-batch-policy.cpp` (tools/server/server-batch-policy.h, the caps the server's update_slots
  applies; red with either cap reverted or the default off: tests/prefill-chunk-mutate.py B1-B3).

- **PREFILL-CHUNK-PF-ONLY (fork-chunk-loop P3a, 2026-10-28).** With LLAMA_PREFILL_CHUNK set (C >= 2) and
  LLAMA_MOE_PREFETCH_PP=R, a graph reads its host experts through the prefetch ring (compact ids) only if it is a pass
  of the chunk loop; every other graph (a batch carrying a generating token, a single-ubatch batch) uploads the routed
  experts as the stock path does, unless LLAMA_MOE_PREFETCH_CHUNK_ONLY=0. Without LLAMA_PREFILL_CHUNK the prefetch
  applies to every wide graph as before (LLAMA_MOE_PREFETCH_CHUNK_ONLY=1 restricts it anyway). Found by the server
  gate's trace arm (server-20261028T005245Z, XT): each capped 2048-token mixed step streamed all 93 non-resident host
  expert tensors through the per-graph ring. Pinned by `tests/test-moe-prefetch-policy.cpp`
  (src/llama-moe-prefetch-policy.h, the decision build_moe_ffn takes); the graph flag it reads is part of
  allow_reuse.

- **PREFILL-CHUNK-INPUTS (fork-chunk-loop P3a, 2026-10-28).** Within a chunk, every pass over ubatch k sees the same
  memory metadata (PREFILL-CHUNK), so the inputs computed from it are the same: the attention KQ mask (mem_hybrid) and
  the qwen4exp QSA tables (k_idxs, cell_blk, blk_cells, blk_pos, bias, tail_idxs) are filled once per ubatch of the
  chunk, on its first pass that reads them, and copied into the later passes' input tensors; the cache is keyed by
  (k, input kind) and checked by byte size, and is cleared at every chunk's start and end, so no fill outlives its
  chunk. The recurrent s_copy is not cached: it is cheap, and it is what advances rs_idx (chunk_save emulates that
  once per ubatch; caching must not add or remove an advance). Outside the chunk loop nothing is cached. Found by
  the server gate's trace arm (server-20261028T005245Z, XT): set_inputs took 1.7-1.85 s of a's 6.3 s 8192-token
  chunk batches, against 0.14 s for the same batch with one sequence in the cache: the mask and QSA fills are
  O(n_kv x n_tokens) and ran on each of a ubatch's 12 attention-layer passes. Pinned by `tests/test-prefill-chunk.cpp`
  (the existing identity checks on logits, h_nextn, state and every named input a pass saw, plus: each chunk ubatch
  fills its mask and QSA set exactly once, and later passes reuse them). Each mechanism reverted alone is red
  (tests/prefill-chunk-mutate.py on 798434b6e, of 63): the mask cache 48, the QSA cache 48, the key without the chunk
  slot 56 (wrong masks), both clears removed 56 (a chunk reuses the previous chunk's fills); one clear alone is enough,
  so the mutation removes both. CPU proxy (the fixture, n_kv <= 512, 119 chunk decodes): set_inputs 90 -> 31 ms.
