# qwen4exp MTP block: tensor inventory

Source: `Qwen/Qwen3.8-Flash-Next` BF16 checkpoint at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
(Strata `tools/mtp_fetch.py` `PINNED_REVISION`). The block is 31 `mtp.*` tensors, 5,214,306,560 bytes, spread
over 28 of the 131 shards. All 31 were fetched by HTTP range reads into
`/data/models/qwen3.8-flash-next-mtp/tensors/`, one raw BF16 file per tensor. Each file matches the
sha256 in mtp_fetch.py's `SHA256` table (`fetch` and `verify`, both exit 0, 2026-10-05).

Hyperparameters from the same revision's `config.json`: `mtp_num_hidden_layers: 1`,
`mtp.layer_types: ["full_attention"]`, `mtp_use_dedicated_embeddings: false` (the block shares the trunk's
embedding and LM head), `mtp_use_hidden_state_from_layer: null` (it reads the trunk's final residual).

The **GGUF** column gives the name and type in `/data/models/qwen3.8-flash-next-mtp/mtp-qwen3.8-flash-next-Q8_0-expQ4_0.gguf`
(blk.48 = `n_layer`, the NextN index). The **trunk counterpart** column gives the role the tensor plays in our
qwen4exp graph (`src/models/qwen4exp.cpp`), with "none" where the trunk has no such tensor. GGUF shapes are
ggml order (ne0 first), and HF shapes are row-major.

| HF tensor | dtype | HF shape | GGUF tensor (type) | trunk counterpart |
|---|---|---|---|---|
| `mtp.fc_embedding.weight` | BF16 | 2560x2560 | `blk.48.nextn.eh_proj` cols 0..2559 (Q8_0, [5120, 2560] together) | **none**: projects the normed token embedding |
| `mtp.fc_hidden.weight` | BF16 | 2560x2560 | `blk.48.nextn.eh_proj` cols 2560..5119 | **none**: projects each of the 4 normed residual streams |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | 2560 | `blk.48.nextn.enorm` (F32, 1+w) | **none** (NextN enorm) |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | 10240 | `blk.48.nextn.hnorm` (F32, 1+w) | **none** (NextN hnorm; one gamma over all hc*n_embd) |
| `mtp.hyper_connection_mixer.hc_norm.weight` | BF16 | 10240 | `blk.48.nextn.hc_head_norm` (F32, 1+w) | `output_hc_norm` (final mixer); **separate weights** |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | BF16 | 320x10240 | `blk.48.nextn.hc_head_down` (Q8_0) | `output_hc_down` |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | BF16 | 10240x320 | `blk.48.nextn.hc_head_up` (Q8_0) | `output_hc_up` |
| `mtp.layers.0.attn_hyper_connection.hc_norm.weight` | BF16 | 10240 | `blk.48.hc_attn_norm` (F32, 1+w) | `blk.N.hc_attn_norm` |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight` | BF16 | 320x10240 | `blk.48.hc_attn_down` (Q8_0) | `blk.N.hc_attn_down` |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight` | BF16 | 10240x320 | `blk.48.hc_attn_up` (Q8_0) | `blk.N.hc_attn_up` |
| `mtp.layers.0.attn_hyper_connection.block_inject_weight.weight` | BF16 | 4x10240 | `blk.48.hc_attn_inject` (F32) | `blk.N.hc_attn_inject` |
| `mtp.layers.0.self_attn.q_proj.weight` | BF16 | 12288x2560 | `blk.48.attn_q` (Q8_0; [q\|gate] per head) | full-attention `blk.{3,7,..,47}.attn_q` |
| `mtp.layers.0.self_attn.k_proj.weight` | BF16 | 512x2560 | `blk.48.attn_k` (Q8_0) | full-attention `attn_k` |
| `mtp.layers.0.self_attn.v_proj.weight` | BF16 | 512x2560 | `blk.48.attn_v` (Q8_0) | full-attention `attn_v` |
| `mtp.layers.0.self_attn.o_proj.weight` | BF16 | 2560x6144 | `blk.48.attn_output` (Q8_0) | full-attention `attn_output` |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | 256 | `blk.48.attn_q_norm` (F32, 1+w) | `attn_q_norm` |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | 256 | `blk.48.attn_k_norm` (F32, 1+w) | `attn_k_norm` |
| `mtp.layers.0.self_attn.indexer.index_qk_proj.weight` | BF16 | 640x2560 | `blk.48.indexer.q_proj` rows 0..511 + `blk.48.indexer.k_proj` rows 512..639 (BF16) | QSA indexer `indexer.q_proj`/`k_proj`. The trunk has it too, so this is **not** new. Strata runs the MTP attention dense and never reads it |
| `mtp.layers.0.self_attn.indexer.q_layernorm.weight` | BF16 | 128 | `blk.48.indexer.q_norm` (F32, 1+w) | `indexer.q_norm` |
| `mtp.layers.0.self_attn.indexer.k_layernorm.weight` | BF16 | 128 | `blk.48.indexer.k_norm` (F32, 1+w) | `indexer.k_norm` |
| `mtp.layers.0.mlp_hyper_connection.hc_norm.weight` | BF16 | 10240 | `blk.48.hc_ffn_norm` (F32, 1+w) | `blk.N.hc_ffn_norm` |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | BF16 | 320x10240 | `blk.48.hc_ffn_down` (Q8_0) | `blk.N.hc_ffn_down` |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | BF16 | 10240x320 | `blk.48.hc_ffn_up` (Q8_0) | `blk.N.hc_ffn_up` |
| `mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight` | BF16 | 4x10240 | `blk.48.hc_ffn_inject` (F32) | `blk.N.hc_ffn_inject` |
| `mtp.layers.0.mlp.gate.weight` | BF16 | 512x2560 | `blk.48.ffn_gate_inp` (F32) | `ffn_gate_inp` (router) |
| `mtp.layers.0.mlp.experts.gate_up_proj` | BF16 | 512x1280x2560 | `blk.48.ffn_gate_exps` rows 0..639 + `blk.48.ffn_up_exps` rows 640..1279 (Q4_0, [2560, 640, 512] each) | `ffn_gate_exps` / `ffn_up_exps` (target: IQ3_S) |
| `mtp.layers.0.mlp.experts.down_proj` | BF16 | 512x2560x640 | `blk.48.ffn_down_exps` (Q4_0, [640, 2560, 512]) | `ffn_down_exps` (target: Q8_0) |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | BF16 | 640x2560 | `blk.48.ffn_gate_shexp` (Q8_0) | `ffn_gate_shexp` |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | BF16 | 640x2560 | `blk.48.ffn_up_shexp` (Q8_0) | `ffn_up_shexp` |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | BF16 | 2560x640 | `blk.48.ffn_down_shexp` (Q8_0) | `ffn_down_shexp` |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | BF16 | 1x2560 | `blk.48.ffn_gate_inp_shexp` (F32, [2560]) | `ffn_gate_inp_shexp` |

Also in the MTP GGUF, copied byte for byte from the target UD-IQ4_XS: `token_embd.weight` (Q8_0) and
`output.weight` (Q6_K). They are the shared embedding and LM head, as the `--mtp` convention keeps them.

## What has no counterpart in the trunk graph

- **`fc_embedding` / `fc_hidden`** (stored as `nextn.eh_proj`) and **`pre_fc_norm_embedding` / `pre_fc_norm_hidden`**
  (`nextn.enorm` / `nextn.hnorm`) are the NextN input fusion. The qwen35/glm5next NextN blocks have them too, but
  in a different form. There, `eh_proj` maps concat(e, h) of width 2*n_embd to n_embd once per token. Here it runs
  once per residual stream: e is shared and h_s is one stream's slice of the 10240-wide residual.
  `hnorm` is a single 10240-wide gamma, not the grouped [n_embd, hc] norm the HC modules use.
- **`hyper_connection_mixer`** (stored as `nextn.hc_head_*`) has the same role as the trunk's `output_hc_*` final
  mixer, but its weights are separate. No existing GGUF tensor name fitted, so three new ones were added
  (`gguf-py` `MODEL_TENSOR.NEXTN_HC_HEAD_{NORM,DOWN,UP}`). The C++ side does not have them yet.
- The **indexer** does have a trunk counterpart (every full-attention layer's QSA indexer). The MTP indexer has
  no consumer if the MTP attention runs dense, as Strata's does (`include/strata/core/mtp.hpp`, header comment).
- The block has **no `shared_head.norm` / `shared_head.head` / `embed_tokens`** of its own. Its final norm is the
  mixer, and it reuses the trunk's head and embedding (`mtp_use_dedicated_embeddings: false`).

## Conventions carried by the conversion

- Every 1-D gamma is stored zero-centred in the checkpoint and folded to (1 + w) in the GGUF. This is
  `Qwen3NextModel.modify_tensors` for `*norm.weight` and `Qwen4ExpTextModel.modify_tensors` for the indexer norms,
  the same rules the trunk was converted with. Strata applies the same fold (`tools/mtp_rt.py` docstring:
  "the RMSNorm weights as F32 with the Gemma "+1" applied"). `scripts/qwen4exp-mtp-gguf-check.py` cross-checks
  it: the MTP layer's and blk.47's hc/q/k norm means sit on the same side.
- Per-role tensor types copy the target's types: dense Q8_0, router/inject/1-D F32, indexer BF16. The fused
  `ggml_hc_mix` path in `build_hc_mix` needs Q8_0 down/up and an F32 inject.
