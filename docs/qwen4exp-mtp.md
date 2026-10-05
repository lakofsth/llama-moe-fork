# qwen4exp MTP (NextN) speculative decoding: design note

Status, 2026-10-05, branch `fork-mtp` (cut from `dca6b25be`, the deployed rev):

- **Converter side: done.** `convert_hf_to_gguf.py --mtp` is accepted for qwen4exp, and the MTP GGUF has been
  built from the fetched tensors.
- **Graph side: not implemented.** This note specifies it.

Nothing here has run on a GPU.

Conventions: `file:function` cites code in this tree (fork) or in Strata (`/data/two-card-bench/ext/Strata`,
read-only). **INFERENCE** marks a conclusion of mine that is not stated in either.

## 1. What exists

| artifact | where |
|---|---|
| fetched `mtp.*` tensors (31, 5.214 GB BF16, sha256-verified, pinned rev `de4b8e4d…`) | `/data/models/qwen3.8-flash-next-mtp/tensors/`, `mtp-manifest.json` |
| checkpoint `config.json` (same revision) | `/data/models/qwen3.8-flash-next-mtp/hf/config.json` |
| MTP GGUF | `/data/models/qwen3.8-flash-next-mtp/mtp-qwen3.8-flash-next-Q8_0-expQ4_0.gguf` |
| tensor inventory | `docs/qwen4exp-mtp-inventory.md` |
| builder / checker | `scripts/qwen4exp-mtp-gguf.py`, `scripts/qwen4exp-mtp-gguf-check.py` |

Converter (`conversion/qwen4exp.py:Qwen4ExpTextModel`):

- `supports_mtp_export = True`.
- A plain conversion still leaves the block out. `no_mtp` defaults to True, so the target GGUF is unchanged.
- `--mtp` keeps only `mtp.*` plus the shared embedding and head.
- The top-level `mtp.*` names go onto `blk.{n_layer}` (`_MTP_TOP_LEVEL`, `filter_tensors`), and
  `_QwenMtpMixin.filter_tensors` remaps `mtp.layers.0.*`. The mixin also extends `block_count` and writes
  `nextn_predict_layers`.
- `fc_embedding` and `fc_hidden` are joined into one `nextn.eh_proj` = [fc_embedding | fc_hidden] (`modify_tensors`).
- `compress_ratios` covers the MTP layer (`set_gguf_parameters`).
- gguf-py gains `NEXTN_HC_HEAD_{NORM,DOWN,UP}` (`blk.N.nextn.hc_head_*`) for the block's own final mixer.
- Under `--mtp`, the PLE hash constants and the PLE table shards still pass the filter, so the draft's PLE
  metadata matches the target's. The shards contribute only their row width; no table is written.
- The full-checkpoint `--mtp` run is untested here because we hold no checkpoint. The filtering (default drops
  all 31 `mtp.*`; `--mtp` keeps them plus embed/head/PLE inputs) and the shard path were smoke-tested on names.

Why there is a builder script rather than a plain `convert_hf_to_gguf.py --mtp` run: we hold only the 31 MTP
tensors, not the 360 GB checkpoint. The converter's `set_gguf_parameters` reads the int64 PLE constants from the
checkpoint, and `set_vocab` needs the tokenizer files. `scripts/qwen4exp-mtp-gguf.py` therefore:

- instantiates the converter's own class, with `mtp_only` set and the fetched files as its tensor index;
- writes what `filter_tensors` / `modify_tensors` yield, so the names and transforms are the converter's;
- copies the metadata and tokenizer from the deployed target GGUF, so the draft's vocabulary is the target's byte
  for byte and `common_speculative_are_compatible` cannot trip on a re-derived tokenizer (INFERENCE: it would
  otherwise be identical too, but this removes the question);
- copies `token_embd` (Q8_0) and `output` (Q6_K) from the target as they are.

Quantization:

- **Dense parts** use the target's type for each role: Q8_0 matrices, F32 router/inject/1-D, BF16 indexer.
  `build_hc_mix`'s fused `ggml_hc_mix` path requires Q8_0 down/up and an F32 inject
  (`src/models/qwen4exp.cpp:llama_model_qwen4exp::graph::build_hc_mix`).
- **Routed experts are Q4_0**: 1.42 GB, against 2.67 GB at Q8_0 and 5.03 GB at BF16. Free VRAM on 2026-10-05 was
  about 4.0 and 2.8 GiB on the two cards (`nvidia-smi`, a read-only query), so a Q8_0 expert set would not fit
  beside the resident at all. The target itself runs these roles at IQ3_S (gate/up) and Q8_0 (down), and Strata
  drafts with Q2_0 experts (`mtp/mtp-q2_0.gguf`, `tools/mtp_pack.py`). Expert precision moves only the acceptance
  rate, never the output.
- The script still offers `--experts q8_0`.

File size: 2.725 GB. That is 1.53 GB for the block, plus the copied token_embd at 0.675 GB and output at 0.521 GB.
**INFERENCE:** in the same-model packaging of §5 the embedding and head are the target's own, so the block adds
about 1.53 GB of weights plus its KV.

## 2. What the MTP layer computes

Source: Strata `include/strata/core/mtp.hpp` (header comment) and `src/core/mtp.cpp:MtpDrafter::record_forward`.
Strata transcribed this from vLLM 0.30.0's `qwen4_exp` MTP and reports 0.89/0.86/0.85 greedy acceptance at
draft steps 1–3 (mtp.hpp comment).

Notation: n = n_embd = 2560 and hc = 4. R is a residual of hc streams, [n, hc] per token, which is the
10240-wide row the fork already exposes as `n_embd_out` (`llama_model_qwen4exp::load_arch_hparams`:
`n_embd_out_impl = dsv4_hc_mult * n_embd`).

Per draft cell, with input token x and hidden residual R_in:

1. **Input fusion.**
   - `e = W_e · (rms_n(embed(x)) ⊙ enorm)`
   - `hn = rms_{hc·n}(R_in) ⊙ hnorm`. This is one RMS over all 10240 values (`record_forward`:
     `native_qsa_rms_norm_weighted(Rin_, …, HC*N, T)`), *not* the grouped per-stream norm of the HC modules.
   - `R_s = W_h · hn_s + e` for each stream s (the `add_streams_broadcast` call).
   - In the GGUF, `W_e` and `W_h` are the two halves of `nextn.eh_proj` (columns 0..n-1 and n..2n-1).
   - ggml form: two `ggml_view_2d` halves of eh_proj, `mul_mat(W_h, reshape(hn, [n, hc·T]))`, and
     `mul_mat(W_e, en)` broadcast-added across streams. The halves are views on a Q8_0 row; 2560 is a multiple of
     the 32-element block, so both views are valid.
2. **Attention HC read**: `mixed, inject = build_hc_mix(R, blk.48.hc_attn_*)`, identical to a trunk layer.
3. **QSA attention** with the MTP layer's own K/V: q|gate interleaved, q/k RMS norm, IMRoPE (sections), sigmoid
   output gate, `wo`. This is `llama_model_qwen4exp::graph::build_layer_attn` unchanged at `il = 48`. Strata runs
   it **dense** over a window of the last 32768 cells (`MtpDrafter::load`, `window` default; `first_needed()`).
   mtp.hpp says that equals the model's sparse selection below 2,051 cells. **INFERENCE:** `build_layer_attn`
   already falls back to dense when the context has no indexer cache (`qsa = mctx_hyb->get_idx() != nullptr && ratio > 0`),
   so dense comes free if the MTP context's memory carries no indexer.
4. `R = build_hc_combine(R, attn_out, inject)`.
5. **FFN HC read**: `mixed2, inject2 = build_hc_mix(R, blk.48.hc_ffn_*)`.
6. **MoE + shared expert**: `build_layer_ffn(mixed2, 48)` unchanged: 512 experts, top-10 softmax, sigmoid-gated
   shared expert. Strata's equivalent: `moe_grouped_s2` + `shared_expert` + `moe_combine` in `record_forward`.
7. `R = build_hc_combine(R, ffn_out, inject2)`. **This R is the cell's `h_nextn`.** It feeds the next draft step
   ("its own residual feeds the next draft step", mtp.hpp).
8. **Head**: `y = build_hc_mix(R, nextn.hc_head_*, no inject)`, then `logits = output · y` using the **target's**
   head (`gr_read(… hyper_connection_mixer …)` then `native_mmvq(head_->type(), …)` in `record_forward`;
   config `mtp_use_dedicated_embeddings: false`).

Position pairing:

- **Strata**: "Cell i … pairs the main model's final multi-stream residual at position i with the token at
  position i+1, at rope position i" (mtp.hpp).
- **The fork's draft-mtp**: cell p pairs the target's `h_{p-1}` with token `x_p` at position p
  (`common/speculative.cpp:common_speculative_impl_draft_mtp::process`, "shift the tgt embeddings to the right by
  one position").
- Both pair the same (residual, token) values, and the rope positions differ by a constant 1.
  **INFERENCE:** harmless, because RoPE only sees relative offsets inside the layer's own KV and every cell is
  shifted alike.

## 3. Target-side changes (trunk graph)

- `llama_model_qwen4exp::graph::graph` sets no `res->t_h_nextn`. It must expose `res_hc` after the last layer's
  FFN combine (the input to the final `hc_head` mixer): `[n·hc, T]`, reshaped 2-D.
  `llama_context` copies `n_embd_out()` floats per row (`src/llama-context.cpp`, the "extract nextn embeddings"
  block), which is 10240 here.
- draft-mtp asks the target for **unmasked** nextn rows (`llama_set_embeddings_nextn(ctx_tgt, true, false)` in the
  draft-mtp constructor). The trunk graph currently prunes rows to `inp_out_ids` at the last layer. It must keep
  all rows when `cparams.embeddings_nextn && !cparams.embeddings_nextn_masked`, as glm5next does with its
  `mask_early` flag (`src/models/glm5next.cpp`, the graph tail).
  **INFERENCE:** pruning only after copying `t_h_nextn` keeps the decode cost unchanged when MTP is off.

## 4. Draft-side changes

- **llama-arch**: `LLM_TENSOR_NEXTN_HC_HEAD_{NORM,DOWN,UP}` ("blk.%d.nextn.hc_head_*"), registered as
  `LLM_TENSOR_LAYER_REPEATING / GGML_OP_MUL_MAT` (norm: `GGML_OP_MUL`). Add `nextn.hc_head_*` fields to
  `llama_layer_nextn` (`src/llama-model.h`).
- **`llama_model_qwen4exp::load_arch_tensors`**: currently loops `il < n_layer` only. Changes:
  - Add an MTP pass for `[n_layer, n_layer_all)`, as in `llama_model_qwen35moe::load_arch_tensors` /
    `load_block_mtp`, with:
    - `mtp_flags = ml.load_mtp ? 0 : TENSOR_SKIP`;
    - full-attention + indexer + HC + MoE tensors;
    - `nextn.eh_proj {2n, n}`, `nextn.enorm {n}`, `nextn.hnorm {hc·n}` (a flat gamma: one RMS over 10240);
    - `nextn.hc_head_norm {n, hc}` with `TENSOR_ALLOW_RESHAPE`, plus `nextn.hc_head_down/up`.
  - Detect an MTP-only file (`blk.0.hc_attn_norm` absent, as qwen35moe probes `blk.0.attn_norm`) and make every
    trunk tensor `TENSOR_NOT_REQUIRED` in that case: the layers, `output_hc_*`, and `per_layer_tok_embd`. The last
    one is created `TENSOR_READ_LAZY` with no not-required flag today.
  - `n_layer_nextn` is read generically (`src/llama-model.cpp`, `LLM_KV_NEXTN_PREDICT_LAYERS`).
    `dsv4_compress_ratios` is read over `n_layer_all`, which the new 49-entry array satisfies.
  - The recurrent-layer default formula already marks `il >= n_layer()` non-recurrent
    (`load_arch_hparams`, `is_recr_impl`).
- **Graph**: `llama_model_qwen4exp::build_arch_graph` returns a `graph_mtp` for `LLM_GRAPH_TYPE_DECODER_MTP`.
  Template: `llama_model_glm5next::graph_mtp` (the closest: hybrid memory, an indexer-carrying NextN layer, and it
  reuses the trunk's `build_layer_attn`) and `llama_model_qwen35moe::graph_mtp`.
  - Its `llm_graph_input_embd_h` must be constructed with `n_embd_out()` (10240), not `n_embd`. `set_input`
    asserts `n_embd == h->ne[0]` (`src/llama-graph.cpp:llm_graph_input_embd_h::set_input`). The MTP batch's embd
    width is already `n_embd_out` (`src/llama-batch.cpp:llama_batch_ext_select_n_embd_inp`).
  - `graph_mtp` must derive from `llama_model_qwen4exp::graph`'s builder (or share its helpers) to reach
    `build_hc_mix`, `build_hc_combine`, `build_layer_attn` and `build_layer_ffn`.
  - `res->t_h_nextn = R` (step 7, before the mixer), and `t_logits` from step 8.
- **Memory** (`src/llama-model.cpp:llama_model::create_memory`): the qwen4exp branch gives attention to
  `il < n_layer() && !is_recr(il)` and the indexer to the same set.
  - For `ctx_type == LLAMA_CONTEXT_TYPE_MTP` it must instead give attention to `il >= n_layer()` only, and
    nothing recurrent. That is exactly the glm5next `mtp_ctx` clause; extend the clause's arch test to qwen4exp.
  - Leave `filter_idx` empty there (dense MTP attention, §2.3) unless the indexer is wanted (§6, Q3).
  - The comment there says why the trunk's cache must not be used: "a rejected draft then fails seq_rm" with
    `n_rs_seq = 0`.

## 5. Plugging into draft-mtp

- With `n_layer_nextn = 1` and no shared memory, `common_speculative_impl_draft_mtp` runs in its single-head mode
  ("neither (qwen35 / qwen35moe)").
- `process()` runs a catch-up decode on `ctx_dft` for every target batch (prompt and verify window), pairing each
  token with the target's shifted `h_nextn` and carrying the last row across calls in `pending_h`.
- `draft()` then decodes one row per drafting sequence per step, feeding back
  `llama_get_embeddings_nextn_ith(ctx_dft, i_last)`, which is the MTP layer's own `t_h_nextn` (§2.7). That matches
  Strata's chain.
- The draft sampler is top-k 10, with `p_min` gating continuation.
- Auto-detection (`common_speculative_*`, the `nextn.eh_proj` probe on `blk.{block_count-1}`) finds this head
  because eh_proj is kept under its standard name.

Two packagings (Q1):

- **(a) Separate file (`-md mtp-….gguf`).** Works with this GGUF as it is. It loads a second `llama_model`
  holding duplicate copies of token_embd and output (about 1.2 GB of VRAM), unless the loader learns to borrow
  them from `ctx_other`'s model.
- **(b) Same model.** Append the blk.48 tensors to the target GGUF (`gguf-split`/merge, `block_count` 49,
  `nextn_predict_layers` 1). Then no `-md`: `common_speculative_init_result` builds the MTP context against
  `model_tgt` (the `else if (spec_mtp)` branch), and the embedding and head are shared.

**Recommendation: (b)**, given the VRAM headroom. This GGUF is its source: the blk.48 tensors and the three
metadata changes.

## 6. Multi-slot (6 server slots)

- The server builds the speculative state with `n_seq = n_parallel` (`tools/server/server-context.cpp`:
  `common_speculative_init(params_base.speculative, params_base.n_parallel)`). draft-mtp keeps
  `pending_h` / `verify_h` / sampler per sequence.
- `process()` requires one seq per token and each sequence's tokens contiguous in the batch. It asserts
  `n_seq_id == 1` and documents the contiguity assumption as a TODO.
  **INFERENCE:** the server's batch construction keeps a slot's tokens contiguous, but this is unchecked and
  needs a test with interleaved slots.
- One draft step decodes up to 6 rows (one per drafting slot) at distinct positions. Those batch sizes stay
  within `build_hc_mix`'s fused path (`nt <= 8`).
  **INFERENCE:** a 6-row MoE step touches up to 60 distinct experts out of 512, so all experts must be resident
  wherever the layer runs. Placement: card with the most headroom, or the fork's CPU-expert path
  (`LLAMA_MOE_OVERLAP`, `moe_split`), Q4.
- **MTP KV**: the draft context gets `n_ctx` = the target's (`common_speculative_init_result`, "the draft context
  holds as many tokens per sequence as the target context"), unified across slots. Per cell, K+V = 2 × 2 kv-heads ×
  256 = 1024 elements: 2 KiB at F16, about 1.06 KiB at Q8_0. **INFERENCE:** 196,608 cells (6 × 32k) costs
  384 MiB F16 / 204 MiB Q8_0. Add 256 B/cell F16 if the indexer cache is enabled.
- **Rejected drafts**: the draft region is rolled back by `llama_memory_seq_rm` on the draft memory. The server
  checkpoints and rolls back `ctx_dft` by `ctx_dft_seq_rm_type` (`server-context.cpp`, `use_ckpt_dft`). An
  attention-only MTP memory supports full seq_rm, so that path stays simple.

## 7. Open questions

1. **Packaging**: same-model (b) or separate file (a)? See §5. (b) saves about 1.2 GB of VRAM.
2. **`hnorm` scope**: Strata normalizes R_in as one 10240-wide vector (`record_forward`). The HC modules normalize
   per stream. Verify against vLLM's `qwen4_exp` MTP source before trusting Strata. If it is per-stream, `hnorm`
   loads as `{n, hc}` with `TENSOR_ALLOW_RESHAPE` instead; the stored gamma is the same.
3. **Dense or QSA for the MTP attention.** Dense (Strata) is simpler and needs no indexer cache. QSA matches the
   model past about 2k cells. **INFERENCE:** for drafting, dense over a window is the better first cut.
4. **Expert precision vs VRAM**: Q4_0 now. Q2-class (Strata) or CPU-resident experts are the fallbacks if card
   headroom shrinks.
5. **The target exposes R for all rows** (§3): measure the cost of the unmasked tail at prompt length. It is a
   10240-float copy per token.
6. **Interleaved multi-slot batches** in `process()` (§6): needs a test.
7. **Rope offset** (§2, last paragraph): confirm it is harmless by measuring acceptance with and without the
   shift, once the graph exists.
8. **The (1 + w) fold on the MTP gammas.** The converter applies the trunk's rule, and Strata does the same
   (`tools/mtp_rt.py`) while drafting at 0.89 acceptance, which is the strong evidence. The raw MTP q/k-norm
   gammas average 2.68, against about 0.44 for the trunk's blk.47 (2026-10-05, from
   `scripts/qwen4exp-mtp-gguf-check.py`'s reference print). Confirm against vLLM's MTP module if acceptance
   comes in low.
