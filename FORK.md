# llama.cpp fork — hybrid MoE streaming (GPU + RAM + NVMe tail)

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The default branch
`fork-merged-master` merges upstream up to `4b1a27fa0` (b11191, 2026-09-25); the original
branch `fork/phase-a` sits on `cd0fa60` (b10286). Targets the *hybrid* MoE regime: the model
nearly fits across GPU + system RAM with a small NVMe tail — between the all-VRAM and all-disk
cases. Developed on DeepSeek-V4-Flash UD-IQ4_XS on a single RTX 5090 (32 GB) + 128 GB RAM, and
since September 2026 on Qwen3.8-Flash-Next (`qwen4exp`) UD-IQ4_XS across two RTX 5090s. Four
upstream commits were cherry-picked when the fork was based on b9994; the b10286 merge brings
upstream's own copies of the same changes, so those four are now redundant duplicates kept for
history. Original authorship is preserved on both copies.

Several mechanisms since October 2026 are ports of ideas from
[Strata](https://github.com/Niko1221/Strata) (Niko1221), a separate inference engine that
serves Qwen3.8-Flash-Next with a GPU expert cache; each is credited where it is described below,
and collected under [Credits](#credits). One dial comes from
[Avalon](https://github.com/freriadoc/Avalon), itself carrying a Strata mechanism.

## What it adds

- **Heat-driven expert split** — pack the globally hottest experts into VRAM from a
  profiled per-layer heat map; a sentinel id encoding splits `mul_mat_id` so resident
  experts compute on GPU, the rest on CPU. Placement is invisible to the math.
- **Online heat (self-tuning)** — routing counters piggyback on the CPU id sweep at zero
  cost; when the GPU hit-rate decays (workload/register drift), the runtime re-ranks and
  repins experts mid-run. Starting on a deliberately wrong map recovers ~86 % of the
  matched-map ceiling autonomously.
- **Adaptive VRAM tier (Strata's rule)** — `LLAMA_MOE_ADAPT=1` replaces the window repin with
  the adaptive tier of Strata's `generate.cpp` (`adapt`, at Strata `36fa455`): every 4 decode
  calls, per layer, non-resident experts whose decayed usage clears a floor displace resident
  ones by a margin, best gains first, up to 96 swaps a round. Each GPU-resident expert now has its
  own sentinel id, so the CPU sweep counts resident hits per expert (the tier's victim choice).
  `LLAMA_MOE_ADAPT_ASYNC=1` moves the swaps onto a copy stream: a victim is evicted at once (the
  CPU serves it) and the incoming expert is admitted one round later, with no host drain.
- **Router locality bias** — add a small ε to GPU-resident experts' selection scores
  pre-top-k (mixture weights stay unbiased). A speed/quality dial.
- **Preventive quality gate (default on)** — per token per layer, in-graph: the biased
  selection is kept only if it retains ≥ (1−δ) of the unbiased top-k's probability mass;
  otherwise that token falls back to unbiased routing *before* any expert runs.
- **Bias-aware repin threshold** — the routing counters see post-bias selections, so the
  repin trigger compensates for the bias-inflated hit-rate (measured freeze mode otherwise).
- **Layer-major prompt processing** (`qwen4exp`) — `LLAMA_PREFILL_CHUNK=C` runs a prompt batch
  of C micro-batches one layer at a time across all C, so each host layer's non-resident experts
  cross PCIe once per chunk instead of once per micro-batch. With `LLAMA_MOE_PREFETCH_PP=R` those
  uploads run on a copy stream into an R-slot ring ahead of the matmul that reads them, and
  `LLAMA_PREFILL_CHUNK_GPU_MASK=1` builds the KQ mask and the sparse-attention block bias on the
  GPU from 16 bytes of metadata per cell instead of filling and copying them on the host. Output
  is bit-identical to the micro-batch path. Modelled on Strata's chunked prompt path (Strata
  runs prompts in 8192-token chunks).
- **Qwen3.8-Flash-Next MTP draft head** — the model's multi-token-prediction block loads as
  NextN layer `blk.{n_layer}` of the target and drafts for `--spec-type draft-mtp`, sharing the
  target's embedding and head. The draft graph is transcribed from Strata's `mtp.cpp`
  (`MtpDrafter::record_forward`). `LLAMA_SPEC_MAX_GEN_SLOTS` stops new drafts when more slots
  are generating than drafting pays for.
- **Decode-path kernels** — one `q8_1` quantization of an activation shared by every
  `mul_mat_vec_q` that reads it; `GGML_OP_HC_MIX`, the `qwen4exp` hyper-connection read as one op
  (three CUDA kernels in place of twelve); and `LLAMA_MOE_OVERLAP=1`, which runs a split layer's
  GPU experts and shared expert while its CPU experts compute.
- **Host-memory residency** — `LLAMA_MLOCK_TENSORS` locks named host tensors (Strata's
  `--ple-io ram` mode, applied to Flash-Next's 26.8 GiB per-layer token-embedding table), and
  `GGML_CUDA_REGISTER_HOST` pins the host copies of the experts so prompt-time uploads are DMA
  from page-locked memory.

## Environment variables

Placement and online heat:

| var | default | meaning |
|---|---|---|
| `LLAMA_MOE_HEAT_FILE` | unset (off) | per-layer×expert f32 heat map; enables the split. The file must be exactly the model's full shape or the persisted short form (NextN rows excluded); any other length is refused with a warning and the split runs on an all-zero map |
| `LLAMA_MOE_HEAT_VRAM_MB` | 20000 | VRAM budget (MiB) for packed experts. **One value**: one pack on the first GPU, every host layer eligible (single-device behaviour, unchanged). **Comma list** (`15000,9000`): budget per GPU device in `CUDA_VISIBLE_DEVICES` order; each host layer's hot experts are packed only on the device that owns the layer under `-sm layer`, missing trailing values are 0, layers owned by the CPU are not packed. One `moe-heat-split: device i ...` line per device at load. An allocation failure on any device disables the split on all of them (WARN; all experts stay on the host) |
| `LLAMA_MOE_HEAT_SPREAD` / `LLAMA_MOE_HEAT_PER_LAYER` | 0 / unset | per-layer cap on packed experts (spread the budget over every host layer / explicit cap). With a budget list the spread cap is computed per device from that device's budget and layers |
| `LLAMA_MOE_HEAT_FILL` | 0 | `1`: after every expert with heat > 0, pack zero-heat experts too until each layer reaches its cap. Without it a map that names fewer experts than the cap leaves slots, and budget, unspent (measured: a 15,144 MiB budget packed 14,672 MiB, and no budget above ~16,000 MiB could be spent). Use with `LLAMA_MOE_ADAPT`, which replaces the filler as traffic arrives. The cap line at WARN carries the packed count and MiB |
| `LLAMA_MOE_HEAT_ANY_BACKEND` | 0 | the split now refuses a pack device whose backend is not CUDA, ROCm or MUSA (only those implement the sentinel id contract); `1` forces it |
| `LLAMA_MOE_HEAT_ONLINE` | 0 | enable online counting + auto-repin (also required by `LLAMA_MOE_ADAPT`) |
| `LLAMA_MOE_COUNT_DECAY` | 0.5 | EMA decay of the online counters per healthy window (cap 0.95); see [Profiling](#profiling-a-heat-map-that-is-worth-serving-2026-10-01) |
| `LLAMA_MOE_REPIN_THRESH` | 0.45 | hit-rate floor that triggers a repin |
| `LLAMA_MOE_REPIN_MIN_HITS` | 20000 | min window sample before judging |
| `LLAMA_MOE_REPIN_BIAS_K` | 0.4 | threshold compensation per unit of router bias |
| `LLAMA_MOE_REPIN_GAIN` | unset (off) | **WIP, unmeasured.** A fraction `f > 0`: repin when re-ranking from the window's counters would raise the hit-rate by at least `f`, instead of the fixed floor. The trace and repin lines print the projected gain. Unset keeps the threshold rule unchanged |
| `LLAMA_MOE_REPIN_MAX_SWAPS` | 0 (uncapped) | cap on expert swaps per repin pass. **Comma list**: one cap per pack device (same indexing as the budget list); 0 = uncapped, **negative = never swap on that device** (use for a slow-link card) |
| `LLAMA_MOE_ADAPT` | 0 | `1`: Strata's adaptive tier in place of the window repin (needs `LLAMA_MOE_HEAT_ONLINE=1`). Logs `moe-heat-online: hit-rate X% (window N ids) -> adaptive tier, S swaps` every 64 decode calls. Refused for `openai-moe` with expert biases |
| `LLAMA_MOE_ADAPT_EVERY` / `_MIN` / `_MARGIN` / `_DECAY` / `_SWAPS` | 4 / 2.0 / 1.5 / 0.7 / 96 | the tier's round cadence (decode calls), usage floor for a candidate, margin a candidate must beat its victim by, usage decay per round, and swaps per round. Defaults are Strata's |
| `LLAMA_MOE_ADAPT_ASYNC` | 0 | `1`: swaps run on a second stream of the device; a victim leaves the maps at once and its replacement is admitted the next round. Only the main (trunk) context runs rounds; an MTP drafter's decodes still advance the cadence |
| `LLAMA_MOE_ROUTER_BIAS` | 0 | ε added to resident experts' selection scores |
| `LLAMA_MOE_BIAS_GATE` | 0.05 | max relative routed-mass displacement per token; `0` = ungated (logs a warning — measured cost at ε=0.5: +2.6 % PPL on a narrow register but **+22 % on a wide one, +29 % on a mismatched map**; the gate holds all of these at noise level) |
| `LLAMA_MOE_TAIL_SKIP` | 0 (off) | `R > 0`: in a decode-shaped pass (< 32 tokens) of a split layer, do not compute a non-resident expert that every token of the pass routes at rank ≥ R (ranked by router weight). Port of Avalon `510abcb1a`, which carries Strata's `STRATA_ROUTE_TAIL_SKIP`. **Output-changing and not recommended**: on Flash-Next at R=7 it fails a KL ≤ 0.005 / top-1 ≥ 99 % gate by ~10× (see below). Unset, the graph is built exactly as before |
| `LLAMA_MOE_HEAT_TRACE` | 0 | per-window hit-rate trace (raise verbosity: see below). When split layers span two or more devices a `per-device hit-rate` line follows each aggregate line (also on every repin). Each hit-rate line now ends with `[all N MoE layers X%: a split, b whole-GPU, c whole-CPU]`, a hit-rate over every MoE layer that is comparable with Strata's "decode expert cache hit rate" |

Prompt processing:

| var | default | meaning |
|---|---|---|
| `LLAMA_MOE_SPLIT_PP` | unset (off) | engage the heat-split on prompt-processing batches too (default: split runs for tg-shaped batches only, pp uses the stock chain). **Opt-in, and the dominant pp lever: ~2.6–2.8× prompt throughput** — measured on V4-Flash at `-ub 2048`, ~200 t/s with it on vs ~75 off (2026-07-21). Numerically parity-safe (~3–4 dp; the earlier m2 CUDA illegal-access was fixed in `48cea759b`). Set `LLAMA_MOE_SPLIT_PP=1` for prompt-heavy workloads |
| `LLAMA_PREFILL_CHUNK` | unset (off) | `C ≥ 2`: a prompt-only batch of more than one micro-batch runs in chunks of C micro-batches, layer-major. `qwen4exp` only; other models, pooling/embeddings, per-layer input embeddings and pipeline parallelism keep the micro-batch path (one WARN each). Raise `-b` to `C × -ub` for it to engage |
| `LLAMA_PREFILL_CHUNK_GROUP` | 4 | layers per pass (1 or a multiple of the layer-type period). `1` is the setting that lets the prefetch ring keep a layer's experts for the whole chunk |
| `LLAMA_PREFILL_CHUNK_GPU_MASK` | 0 | `1`: chunk passes build the KQ mask and the `qwen4exp` sparse-attention block bias on the device (`GGML_OP_MASK_FROM_META`). SWA, ALiBi, more than 32 sequences in a stream and graphs outside the chunk loop keep the host fill. One WARN line per process says which path ran |
| `LLAMA_PREFILL_CHUNK_MIXED_BATCH` | 2048 | with `LLAMA_PREFILL_CHUNK` set, `llama-server` stops filling a batch at this many tokens once it carries a generating slot's token, and caps the slots after one that is reaching the end of its prompt; prompt-only batches still fill to `-b`. `0` = off |
| `LLAMA_MOE_PREFETCH_PP` | unset (off) | `R`: under `LLAMA_MOE_SPLIT_PP`, upload the next host layers' non-resident experts on a copy stream into an R-slot ring, ahead of the matmul that reads them, with no id read-back or device drain. Bit-identical to the stock partial upload. On its own it does not pay on Flash-Next (the ring's VRAM comes out of the expert pack); with the chunk loop and `GROUP=1` each layer is uploaded once per chunk |
| `LLAMA_MOE_PREFETCH_PP_MIN` | 1024 | smallest micro-batch the ring serves; rounded down to a multiple of the micro-batch's sequence count, so 3+ concurrent prompts still use it |
| `LLAMA_MOE_PREFETCH_CHUNK_ONLY` | 1 under `LLAMA_PREFILL_CHUNK` | the ring serves chunk passes only; other wide graphs take the stock routed upload. `0` restores the ring on every wide graph; `1` applies the rule without the chunk loop |
| `LLAMA_PIPELINE_WITH_OVERRIDES` | 0 | **Experimental, unmeasured.** Allow pipeline parallelism across cards when tensor overrides are present (`--n-cpu-moe` is one, so hybrid runs never pipeline otherwise); the reserve's no-pipeline fallback covers an OOM |
| `LLAMA_MOE_PREFETCH` | — | router-lookahead async prefetch (phase A). Measured no-op when weights are fully RAM/page-cache resident (Qwen3-235B warm-cache test); intended for genuinely NVMe-tail-streaming configs |

Decode path, speculative decoding, memory:

| var | default | meaning |
|---|---|---|
| `LLAMA_MOE_OVERLAP` | 0 | `1`: on a split layer, the GPU experts and (`qwen4exp`) the shared expert run while the CPU experts compute — graph order plus a scheduler cut and eager device→host copies. Moves work in time only; temperature-0 text identical |
| `GGML_CUDA_Q8_1_CACHE` | on | quantize an activation to `q8_1` once for every `mul_mat_vec_q` that reads it (stream 0, invalidated on any write); `0` turns it off |
| `LLAMA_FUSED_HC_MIX` | on | `qwen4exp` uses `GGML_OP_HC_MIX` for batches of ≤ 8 tokens; `0` keeps the unfused graph |
| `GGML_CUDA_TOPK_CUB` | 0 | CUDA top-k now takes one batched launch for all rows (bitonic for ≤ 1024 columns, radix select above) instead of a CUB call per row; a tie at the cut takes the lowest columns, so the selected set is deterministic. `1` restores the per-row CUB path |
| `LLAMA_SPEC_MAX_GEN_SLOTS` | unset (every step may draft) | `llama-server`: start no new speculative draft in a step where more than K slots are generating; a draft in flight is still verified |
| `LLAMA_SPEC_DRAFT_UBATCH` | inherits `-ub` | cap the draft context's micro-batch (smaller values only). The `qwen4exp` NextN block at `-ub 2048` reserves a 1,959 MiB compute buffer |
| `LLAMA_GRAPH_KEEP_PLANS` | on | keep one scheduler plan per output arena, so an MTP drafter alternating catch-up and draft batches no longer rebuilds its graph and plan twice a step; `0` turns it off |
| `LLAMA_SAMPLER_RESERVE_ALWAYS` | 0 | a backend-sampler change now re-reserves only when the reserved shapes do not cover it, and a reserve keeps the prefetch ring; `1` restores upstream's reserve on every change |
| `LLAMA_MLOCK_TENSORS` | unset | `<substr>[,...]`: after load, `mlock` the host ranges of tensors whose names contain a substring (parallel chunks, all-or-nothing per range; failures logged with errno). Needs `RLIMIT_MEMLOCK` |
| `GGML_CUDA_REGISTER_HOST` | unset | the CUDA backend's own switch; the fork now uses it to register every host-resident `*_exps` tensor (page-aligned, coalesced) with CUDA, so prompt-time uploads are DMA from pinned memory |
| `GGML_CUDA_REGISTER_HOST_EXTRA` | unset | `<substr>[,...]`: with `GGML_CUDA_REGISTER_HOST`, pin those host tensors too (a stand-in for `LLAMA_MLOCK_TENSORS` where `RLIMIT_MEMLOCK` is unavailable; measurement knob) |
| `LLAMA_MMAP_NO_PREFETCH` | 0 | skip whole-file MADV_WILLNEED at load |

Diagnostics (off by default; no effect on output):

| var | meaning |
|---|---|
| `LLAMA_MOE_COUNT_ALL` | the online counters also read the stock graph's top-k ids, so a run with every expert on the host counts every routing exactly (used by `llama-moe-route-trace`) |
| `LLAMA_MOE_TRACE_IDS` | record each layer's expert ids in token order (`llama-moe-route-trace --ids-out`) |
| `LLAMA_MOE_PF_TRACE` / `_N` | one WARN line per planned prefetch graph and chunk switch, up to N lines (400) |
| `LLAMA_PREFILL_CHUNK_TRACE` | one WARN line per `llama_decode`: path taken and why, chunk passes, reserve time, graph-build and `set_inputs` time per input class, prefetch uploads/reads |
| `LLAMA_SERVER_BATCH_TRACE` | one WARN line per server decode: per-slot prompt/generated tokens and outputs, decode and host time |
| `LLAMA_GRAPH_STATS=S` | every S seconds: graph and scheduler-plan rebuilds per context, their cause, shapes and host time |
| `GGML_CUDA_GRAPH_STATS=S` | every S seconds: CUDA graph captures, launches and the first changed node of each property change |
| `GGML_CUDA_GRAPH_SHAPE_KEY` | `1`: key the CUDA graph cache by split shape. **Measured no change** on Flash-Next (93–98 % of split computes run eagerly either way); kept for the stats it enables |
| `LLAMA_MOE_PF_GROW_FAIL_AFTER=N` | test only: fail every prefetch slot growth after the first N, to exercise the fallback |

**Two or more GPUs** (`-sm layer -ts a,b`): what the budget list changes is only where each host layer's packed experts live — with its own layer's device, so a token still crosses devices once per forward pass. Unchanged: cold-expert prompt uploads (op offload) still go to backend 0, so the widest-link card must be device 0 (a WARN fires at load when it is not, by link width); KV follows its layer; the router bias and bias gate are per-layer and follow the pack.

Repin/trace events log at WARN; `llama-cli` defaults to error-only, so pass `-lv 2` there.
On `llama-server` that is not enough: `-lv 2` suppresses library INFO, the
class carrying the `moe-heat-split:` placement line, while server INFO still prints. Use `-lv 4`
(or `-v`). `ENABLED` reports the split was built, not used: only a non-zero online hit rate shows
it is consulted.

## Measured (fi-prose register, 5090 + DDR5, tg tokens/s)

| config | tg | PPL delta |
|---|---|---|
| dense offload baseline (`-ncmoe`) | ~11 | — |
| heat split, matched map | ~22 | 0 (placement alone can't change PPL) |
| + online, starting from a *wrong* map | 19–25 avg | 0 after repin |
| + bias ε=0.5, gate δ=0.05 | **30** | +0.2 % (noise) |
| + bias ε=0.5, ungated | 40 | +2.6 % |

## Prompt processing (V4-Flash IQ4_XS, 5090 + DDR5, pp tokens/s)

Two levers, both on the flagship. `-ub`/`-b` are stock flags; `LLAMA_MOE_SPLIT_PP` is the
fork's, and it does the heavy lifting.

| config | pp (t/s, warm) | measured |
|---|---|---|
| `-ub 512`, split-pp on | ~64 | 2026-07 |
| `-ub 2048 -b 2048`, split-pp on | **~200** (209–214) | 2026-07 |
| `-ub 2048 -b 2048`, **split-pp off** | ~75 | 2026-07-21 |

So raising the micro-batch is a stock 3.3× (64 → ~200), but the isolation A/B shows that gain
is the fork's: with `LLAMA_MOE_SPLIT_PP` off, `-ub 2048` reaches only ~75 t/s — **split-pp is a
~2.6–2.8× multiplier and the dominant pp contributor.** For prompt-heavy work, run `-ub 2048 -b
2048 LLAMA_MOE_SPLIT_PP=1`. tg is unaffected by split-pp (it engages only for pp-shaped batches).

## Qwen3.8-Flash-Next (`qwen4exp`) on two RTX 5090s

UD-IQ4_XS (~94 GB in three shards), 2× RTX 5090 32 GB (the second card on a PCIe Gen4 x4 link),
16-core DDR5 host with 128 GB RAM, served by `llama-server` with up to six slots and a unified
262,144-cell KV cache. Each row is its own interleaved A/B (arms alternated A B B A …) against
the build before it, under a constant CPU load on the cores the server does not use. Decode
figures are aggregate t/s; N is the number of concurrently generating slots.

| change | commits | measured | before → after |
|---|---|---|---|
| adaptive VRAM tier (Strata) | `727aa4166` | single-stream decode at 4K / 32K context; split-layer expert hit-rate | 46.2 → 63.7 / 38.2 → 54.3 t/s; 62.1 % → 87.4 % |
| quantize-once + `HC_MIX` + overlap | `4d7211d6b` `ca8debcd2` `3932eebad` `dca6b25be` | tg128, two runs per arm | 72.7 / 71.9 → 79.3 / 82.3 t/s (~+12 %) |
| of which `LLAMA_MOE_OVERLAP` alone | `dca6b25be` | tg128, same binary toggled | 78.1 / 77.3 → 79.4 / 81.7 t/s (~+3.6 %) |
| MTP draft head (draft length 2), gated at 1 generating slot | `655476e08` `a4dab742d` | decode N=1; N=2–6 | 61.6 → 86.3 t/s (+40 %); level (−1 % to +1 %) |
| `LLAMA_MOE_HEAT_FILL` at +1,000 MiB budget | `d42f9dac5` | mean decode N=1–6; split-layer hit-rate | 103.8 → 111.1 t/s (+7 %); 83.8 % → 86.6 % |
| `LLAMA_MLOCK_TENSORS=per_layer_token_embd` | `2182e2e43` `2ac78d114` | lone long prompts | a slow mode at 776–900 t/s (the table faulted back from NVMe: 559,405 major faults in two prompts) → 8 of 8 prompts at 1,364–1,405 t/s, 0 faults; 27,466 MiB locked in 3.2 s |
| `GGML_CUDA_REGISTER_HOST` (split-pp off) | `13ec3b18e` | lone 35K–72K-token prompts, mean | 604 → 766 t/s (+27 %) |
| `LLAMA_MOE_ADAPT_ASYNC` | `e983d9710` `503ae9c26` | decode N=2 / N=3 / N=1 settled | +14.7 % / +6.9 % / +16 % (the first pass after load is 5 % slower while the tier settles) |
| AVX-512 `iq3_s` dot + KV `seq_rm` by position index | `21eb85758` `c1086eb30` | decode summed over slot phases | +1.4 % (dot product alone, one thread: +12 % hot, +6 % cold cache; `seq_rm` 106.5 → 0.60 µs) |
| chunk loop + prefetch ring + device masks | `a29370360` … `b87cdc6ba` | lone 36K/47K-token prompts, median; short prompt's time to first token in a concurrent trio; decode | 1,044 → 1,923 t/s (+84 %); 28.2 → 19.3 s; +1.38 % ± 1.91 % (neutral) |
| one scheduler plan per output arena | `890b00f88` | drafter graph/plan rebuilds per step; their host time; decode | 2.00 → 0.026; 0.362 → 0.042 ms/step; +1.57 % ± 0.83 % (5 pairs, not significant) |
| `LLAMA_MOE_TAIL_SKIP=7` (default off) | `69c912551` | decode; quality vs an A/A floor of KL 0.000 | +17.3 % ± 2.5 %; KL 0.013–0.104, top-1 agreement 85–97 % — **rejected** |

The chunk-loop row is `LLAMA_PREFILL_CHUNK=8`, `LLAMA_PREFILL_CHUNK_GROUP=1`,
`LLAMA_MOE_PREFETCH_PP=3`, `LLAMA_PREFILL_CHUNK_GPU_MASK=1`, `-b 8192`. The prefetch ring without
the chunk loop gave no prompt gain on the same base (median 668 → 652 t/s on the longest prompt),
and an earlier measurement on a smaller pack gave +12 % prompt for −8 % decode, because its slots
come out of the expert pack. `HC_MIX` changes output only as a summation-order change does: KLD
0.031–0.032 against the unfused graph, equal to `GGML_CUDA_DISABLE_FUSION=1`'s 0.031. MTP without
the gate (draft length 3) is +44 % at one generating slot, +2 % at two, and −6 / −9 / −13 % at three, four and six:
once a step is wide, each slot's verify batch costs more than its accepted drafts save.

Against Strata, on the same two cards: on 2 October, both serving the same UD-Q4_K_XL file,
Strata decoded 125–146 t/s single-stream against the fork's 20–34, prefilled ~2,000 t/s at ~40K
tokens against ~300, and held a 95–98 % decode expert-cache hit-rate against 61–82 %. By late
October the fork, on UD-IQ4_XS (so the comparison is directional), decoded 142 t/s single-stream
(MTP on), prefilled lone long prompts at 1,923 t/s, and held an 85–89 % split-layer hit-rate.

Reference recipe (the configuration behind the late-October figures):
`--n-cpu-moe 31 -sm layer -ts 31,19 -ub 1024 -b 8192 -t 6 -tb 8 --backend-sampling`, q5_1 KV,
`--spec-type draft-mtp` on the four-shard set with draft length 2,
`LLAMA_SPEC_MAX_GEN_SLOTS=2 LLAMA_SPEC_DRAFT_UBATCH=256`,
`LLAMA_MOE_HEAT_FILE=<profiled map> LLAMA_MOE_HEAT_SPREAD=1 LLAMA_MOE_HEAT_VRAM_MB=20314
LLAMA_MOE_HEAT_FILL=1 LLAMA_MOE_HEAT_ONLINE=1 LLAMA_MOE_ADAPT=1 LLAMA_MOE_ADAPT_ASYNC=1
LLAMA_MOE_OVERLAP=1 LLAMA_MOE_SPLIT_PP=1 GGML_CUDA_REGISTER_HOST=1
LLAMA_MLOCK_TENSORS=per_layer_token_embd LLAMA_PREFILL_CHUNK=8 LLAMA_PREFILL_CHUNK_GROUP=1
LLAMA_PREFILL_CHUNK_GPU_MASK=1 LLAMA_MOE_PREFETCH_PP=3`, no router bias. Two of its choices are
layout rather than code, each measured against the layout before it: `-ub 1024` with the pack
raised by the freed compute buffer gave +13 % decode at N=1 and +16 % at N=2 for 22 % less lone
prefill; `-ts 31,19` with the pack at 20,314 MiB, against `-ts 33,17`, gave +11.9 % decode at N=1
(142 vs 127 t/s) and +10.9 % prefill on the longest prompt.

**Serving the MTP head.** The published checkpoint's `mtp.*` tensors are converted by
`scripts/qwen4exp-mtp-gguf.py` (it runs the converter's own `Qwen4ExpTextModel` over the 31
tensors Strata's `mtp_fetch.py` range-reads from the BF16 checkpoint) and checked by
`scripts/qwen4exp-mtp-gguf-check.py`; `convert_hf_to_gguf.py --mtp` exports the same block from a
full checkpoint. `scripts/qwen4exp-mtp-shards.py` then builds a four-shard set whose shards 2–3
are symlinks to the target's, so the weights exist once on disk and in the page cache. A context
that does not ask for the drafter skips the block, but `-sm layer` counts one more slot (50
instead of 49), so restate `-ts` for it (the reference recipe's `31,19` is over 50 slots). Design notes: `docs/qwen4exp-mtp.md`, `docs/qwen4exp-mtp-inventory.md`.

## Architecture support

The machinery is generic — it operates on the standard `ffn_*_exps` expert layout and the
shared `build_moe_ffn` graph path. Each architecture is enabled by a whitelist entry plus a
two-line builder change, and ships only after validation on real hardware. An architecture
adopted from upstream needs the builder change too: without the split passed into
`build_moe_ffn` the loader packs experts that the graph never reads (`fca6dbedf`). Layers whose
shape the split chain cannot carry (per-expert scale tensors, expert biases on only some
projections, biases with merged `gate_up`, biases left on a device while the experts are on the
host) are now skipped at load with a counted warning instead of aborting at graph build.

| arch | status |
|---|---|
| `deepseek4` | validated (DeepSeek-V4-Flash — all numbers above) |
| `qwen3moe` | validated (Qwen3-30B-A3B Q6, flat-map bootstrap: +32 % tg over `-ncmoe` baseline, 84 % hit-rate with no profiled map) |
| `minimax-m2` | validated (MiniMax-M2.7 IQ3, 75 G true hybrid regime: a 6 GB expert budget captures 76 % of routing on 256-expert layers, +9 % tg at that deliberately small budget) |
| `qwen35moe` | validated (Qwen3.6-35B-A3B Q5, merged gate_up: +21 % tg, flat-map bootstrap) |
| `deepseek2` | validated (GLM-4.7-Flash Q4 — converts of GLM-4.7 declare this arch — MLA + merged gate_up: **+39 % tg**, flat-map bootstrap) |
| `qwen4exp` | validated (Qwen3.8-Flash-Next UD-IQ4_XS, 512 experts × 48 layers; model support from upstream PR #27742). Single RTX 5090, profiled map, 15,000 MiB budget, 118K-token depth: **22.0 t/s tg vs 19.5 at the stock `--n-cpu-moe 40` (+13 %)** and 245 vs 217 t/s pp (+7 % tg over the best stock setting, `--n-cpu-moe 36`, for 556 MiB more VRAM); two cards: see [its section](#qwen38-flash-next-qwen4exp-on-two-rtx-5090s). Fork additions: MTP draft head, layer-major prompt loop, `HC_MIX`, whole-block sparse-attention selection |
| `hy-v3` | validated (Tencent Hy3 295B IQ1_M, 89 G, hybrid regime: 8.4 → 12.7 t/s (+51 %) at a 14 GB expert budget, 85.6 % hit-rate. Budget lesson: leave the GPU headroom for compute buffers — 22 GB starved context allocation) — **not re-verified since the b10286 merge**: no Tencent Hy3 295B weights on the development machine |
| `kimi-linear` | validated (Kimi-Linear-48B-A3B, self-quantized Q4_K_M — no community GGUFs exist: +20 % tg, flat-map bootstrap) — **not re-verified since the b10286 merge**: no Kimi-Linear-48B-A3B weights on the development machine |
| `glm4-moe` | validated (GLM-4.5-Air Q4 64 G, genuine glm4moe GGUF, hybrid regime: +14 % tg at default budget) — **not re-verified since the b10286 merge**: no GLM-4.5-Air weights on the development machine |
| `openai-moe` | ported (gpt-oss-120b MXFP4; whitelist + expert-bias split support). Runs with the split engaged and generates cleanly, but on a 16-token smoke run only — **throughput never measured, so not validated**. Its expert biases keep the shared resident sentinel, so `LLAMA_MOE_ADAPT` is refused on it |
| `glm5next` | validated on two cards (GLM-5.3-Flash UD-IQ3_XXS 120 G, upstream PR #27754 merged into the fork; needs `-fa off` and `NVIDIA_TF32_OVERRIDE=0`). Converged map, per-device packs 20000,20000 MiB: **13.0 t/s tg vs 10.4 with whole layers on the GPUs (+25 %)**, 7.7–8.0 all-host; frozen map validated at a 58–59 % served hit-rate on a fresh corpus slice |
| `mimo2` | validated on two cards (MiMo-V2.6-Flash-MOPD MXFP4 167 G — the experts are natively MXFP4, so this is full quality). Converged map (profiled with per-window re-ranking), packs 22000,22000 MiB: **12.6 t/s tg vs 10.1 whole-layer (+25 %)**, 90.5 vs 76.6 pp2048. Routing is content-dependent: a static map transfers at ~40 % hit-rate; serve it frozen — continuous repinning reached 63–70 % hit but halved decode (swap traffic over PCIe) |
| `bailingmoe2` | plumbed; NOTE: Ling-2.6-flash itself is BailingMoeV2_5 (MLA + linear-attn hybrid) — needs new upstream converter+runtime support, not just this fork's plumbing |
| `qwen3next`, `step35` | next — two-line recipe; merged gate_up supported |

Models without a profiled heat map bootstrap from a **flat map**: online repin measures the
real per-expert heat during the first hundreds of tokens and repacks VRAM by itself —
self-profiling, no logging pipeline needed.

## Profiling a heat map that is worth serving (2026-10-01)

Three properties of the online counters decide whether `--moe-heat-label` persists a useful map:

- **Packed experts are invisible to the CPU op.** Their routings reach `mul_mat_id` as sentinels,
  so the raw counters hold the COLD set only. Until 6f945a95a the persisted map was those raw
  counts, the inverse of the placement it came from; it now persists the repin's own estimate
  (exact counts for host experts, sentinel mass shared by current heat for packed ones). Since
  `ae3233624` each packed expert has its own sentinel, so its hits are counted exactly too.
- **The counters are an EMA.** `LLAMA_MOE_COUNT_DECAY` (default 0.5) keeps half per healthy
  window, so a default-decay profile is ~2 windows of routing however long it runs. Profile at
  0.95 (the cap).
- **A healthy window never repins** (`LLAMA_MOE_REPIN_THRESH`, default 0.45), so the placement —
  and with it the packed experts' estimated heat — freezes at window 1. Profile with the
  threshold at 0.99 (re-rank every window), then **serve frozen** at the default. On
  Flash-Next the default rule had effectively frozen the map in service (3 repins in 388K decode
  tokens of replayed traffic); `LLAMA_MOE_ADAPT` sidesteps the threshold altogether.

Validate a map by serving it frozen (`LLAMA_MOE_REPIN_THRESH=0`, no label, so nothing is
overwritten) on a corpus slice it was not counted on, and reading the trace's served hit-rate. A
map's capture scored on its own counts overstates it. Persisted maps exclude NextN layers
(`hparams.n_layer()`); the loader zero-pads them since f3901a0ec, and since `a20b379a7` it refuses a
file of any length other than the full or the short form (a map of another model is no longer
silently reinterpreted).

For exact routing data, `llama-moe-route-trace` (below) records per-conversation, per-layer
expert counts with every expert on the host, and with `--ids-out` the routing in token order —
enough to replay a cache policy offline. The adaptive tier was chosen that way: on a recorded
Flash-Next snapshot, Strata's rule held 89.9 % split-layer hit-rate at 3.53 swaps per token against
the window repin's 84.1 % at 11.95.

## Tools and tests

- `llama-moe-route-trace` — evaluates segment files (one continuing KV sequence per
  conversation) and writes each segment's per-layer expert counts; refuses to run without
  `LLAMA_MOE_COUNT_ALL=1`. `--ids-out` writes `[token][layer][n_used]` `uint16` ids in token
  order (needs `LLAMA_MOE_TRACE_IDS=1` and `-ub == -b`).
- `llama-prefill-dump` — dumps `h_nextn`, logits and sequence state after one prompt, for
  bit-comparing the chunk loop against the micro-batch path; `--sample N` keeps the backend
  sampler chain and dumps N sampled tokens; `--warm N` decodes N tokens first (adaptive-tier
  rounds before the prompt).
- `scripts/qwen4exp-mtp-gguf.py`, `qwen4exp-mtp-gguf-check.py`, `qwen4exp-mtp-shards.py` — the
  MTP GGUF and the four-shard serving set (above).
- New CPU unit tests: `test-moe-online-hits`, `test-moe-adapt-select`, `test-moe-adapt-async`,
  `test-moe-heat-select`, `test-moe-heat-map`, `test-moe-prefetch`, `test-moe-prefetch-policy`,
  `test-moe-pf-ring`, `test-moe-tail-skip`, `test-prefill-chunk`, `test-mask-from-meta`,
  `test-server-batch-policy`, `test-qsa-select`, `test-hc-mix-composition`, `test-iq3s-dot`,
  `test-kv-seqrm`, `test-mlock-tensors`, `test-sched-plan`, `test-plan-restore`,
  `test-layer-walk`; CUDA: `test-cuda-q8-1-cache`, `test-cuda-graph-key`. `test-backend-ops`
  gains the heat split's sentinel/duplicate-id `MUL_MAT_ID` cases over F16/BF16/F32/Q4_K/Q8_0/
  IQ4_XS/Q4_0, `HC_MIX`, and router- and indexer-shaped `TOP_K` cases.
- `tests/INVARIANTS.md` states each mechanism's invariant and the test that pins it, with the
  mutations each test was shown to catch. `tests/prefill-chunk-mutate.py`,
  `prefill-chunk-kmutate.py` and `prefill-chunk-pfmutate.py` re-run those mutations;
  `tests/prefill-chunk-gate-*.sh` are the GPU gates used for the chunk loop.

## Fixes

- **CUDA MMQ over-read** (`f24965e3c`, `fbd98f26e`): the last column tile reads up to J−1 blocks
  past `src1` and `ids_dst`, and upstream's padding (#24127) sized it for far smaller J. When the
  buffer ended near a VMM mapping boundary this was an Xid 31 MMU fault, reported as "an illegal
  memory access" in whichever op ran next. Both buffers are now padded for the largest tile; the
  `ids_dst` half matches upstream's later fix (#29953).
- **Heat split, bounds and lifetime** (`5e212f2b6`): models with more than 256 experts no longer
  write past the prefetch bitmap; counters beyond 128 layers × 1024 experts run uncounted instead
  of reading past their table; a second model in the process no longer shares the first's
  counters; the packed buffers are freed with the model (a full pack leaked per reload).
- **Asynchronous host→device inputs** (`a630cb437`): a later host write could land on a pinned
  buffer a copy was still reading (silent wrong activations, a second window on two GPUs); every
  host write now waits for the pending copies.
- **Sentinel ids on every expert kernel** (`2ee562dc1`, `2ede83198`, `d290b6c31`): `mmf`, `mmvf`,
  the cuBLAS fallback and single-token `mmvq` gave NaN or out-of-range reads on a sentinel; F16/BF16/F32
  experts never see the duplicate-id encoding (their kernels keep one slot per expert); the
  `mm_ids_helper` store is sized for every entry it can emit.
- **Clamped SwiGLU** (`eb5a64b32`): the split chain now picks the fused or unfused clamp form by
  the same predicate as the stock chain, so a newly whitelisted clamped arch cannot change numerics.
- **EAGLE-3.1** (`fb2d322e6`): `fc_norm` weights are created at the target-slice width, and a
  partial set is refused instead of dereferencing null.
- **Portability** (`03547bc55`): `--moe-heat-label` and the heat-map sidecar build on MSVC;
  libllama reaches the CPU backend's counters through its registry, so `GGML_BACKEND_DL` builds link.
- **`qwen4exp` sparse attention** (`806a42c45`, `095d2aac4`): selection now takes
  whole 4-cell blocks plus the query's tail, as upstream's later port does, instead of a cell-level
  top-k that cut a block and broke ties in atomic order, so the attended set and the prompt output
  no longer vary run to run past the selection budget; the radix top-k breaks a tie at the cut by
  lowest column.
- **Model save / training init** (`ce14b34b3`): `llama_model_save_to_file` (and with it
  `test-generate-models` and its dependents) segfaulted, because two walks read `llama_layer`
  as an array of tensor pointers; both now enumerate the model's tensor registry.

## Measured models (single 5090 + 128 GB DDR5; flat-map bootstrap, ε=0.5 gated, tg tokens/s)

| model | quant / size | baseline → fork | note |
|---|---|---|---|
| DeepSeek-V4-Flash | IQ4_XS 129 G | ~11 → 30 | flagship config; profiled maps + full dial (see tables above) |
| Tencent Hy3 295B | IQ1_M 89 G | 8.4 → 12.7 (+51 %) | 14 GB expert budget, 85.6 % hit |
| GLM-4.7-Flash | Q4_K_XL 17.5 G | 41.3 → 57.6 (+39 %) | ships as `deepseek2` arch |
| Qwen3-30B-A3B | Q6_K_XL 26 G | 34.5 → 45.5 (+32 %) | |
| Qwen3.6-35B-A3B | Q5_K_XL 25 G | 51.5 → 62.1 (+21 %) | merged gate_up |
| Kimi-Linear-48B | Q4_K_M 29.7 G (self-quant) | 47.2 → 56.6 (+20 %) | no community GGUFs exist |
| Qwen3-235B-A22B | Q3_K_XL 97 G | 7.7 → 8.8 (+14 %) | prefetch arm no-op (warm cache) |
| GLM-4.5-Air | Q4_K_XL 64 G | 11.5 → 13.1 (+14 %) | genuine `glm4moe` |
| MiniMax-M2.7 | IQ3_XXS 75 G | 16.5 → 18.0 (+9 %) | 6 GB budget catches 76 % of 256-expert routing |
| Qwen3.5-122B-A10B | Q4_K_XL 72 G | 19.1 → 20.6 (+8 %) | 6 GB budget |

All non-flagship rows are flat-map bootstrap at mostly-default budgets — the *floor* of what
tuned maps and budgets give, not the ceiling. Qwen3.8-Flash-Next is measured under a different
setup (profiled map, no router bias) and has [its own section](#qwen38-flash-next-qwen4exp-on-two-rtx-5090s).

**Each row is a single measurement, not a repeated one.** Run-to-run spread on the development
machine, measured over 20 reps at fixed seed and prompt (2026-08-10): 0.9 % sd for Qwen3-30B-A3B,
6.0 % sd for MiniMax-M2.7 — so a single-vs-single comparison carries roughly 1–8 % uncertainty
depending on the model. The large deltas are safe; the **+8 % and +9 % rows sit within about one
standard deviation of their own measurement** and should be read as "no regression, probably a
small gain" rather than as established figures. On the two-card Flash-Next server the A/B harness
resolves about 2 % between runs, so the Flash-Next rows of 1–2 % above are directional.

## What to run on what hardware

Guidance, not law — tg depends heavily on RAM bandwidth. "Hybrid regime" means the model
spans GPU + RAM with at most a modest NVMe tail; that's where this fork pays.

| hardware class | suggestion |
|---|---|
| 10–12 GB VRAM + 64 GB RAM (e.g. 3080 desktop) | **GLM-4.7-Flash Q4_K_XL (~18 GB, validated +39 %)** or Kimi-Linear-48B Q4 (~30 GB, validated +20 %) — honest 4-bit quality, fits RAM+VRAM easily. Adventurous today: DeepSeek-V4-Flash at a ~2-bit dynamic quant (~65–70 GB, NVMe tail, unmeasured — expect single-digit tg, rough 2-bit quality) |
| 24–32 GB VRAM + 96–128 GB RAM (e.g. 4090/5090 workstation) | DeepSeek-V4-Flash UD-IQ4_XS (129 GB) — the measured configuration above |
| 2× 32 GB VRAM + 128 GB RAM | Qwen3.8-Flash-Next UD-IQ4_XS (~94 GB) with the reference recipe above: adaptive tier, MTP, layer-major prompt loop |
| CPU-heavy boxes, 192 GB+ RAM, small GPU | Large MoEs at 4-bit (V4-Flash-class and up); the heat split keeps the small card saturated with the hot experts |

Quant-size rule of thumb for the hybrid budget: fit ≈ VRAM + RAM − (OS + KV cache + ~10 %
headroom); anything past that streams from NVMe, which the placement tolerates but tg pays for.

## Credits

- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221), read at `36fa455`. The fork's
  adaptive VRAM tier is Strata's `adapt` rule with Strata's defaults; filling the tier at load
  follows Strata's tier, which starts full; `LLAMA_MLOCK_TENSORS` on the per-layer token
  embeddings is Strata's `--ple-io ram` mode; the `qwen4exp` MTP draft graph is transcribed from
  Strata's `mtp.cpp`, and the MTP GGUF builder reads tensors fetched by Strata's `mtp_fetch.py`;
  the copy-stream prefetch ring and the layer-major chunk loop follow Strata's streaming ring and
  chunked prompt path; the all-layer hit-rate is printed to be comparable with Strata's. Strata
  also set the targets the Flash-Next work measured itself against.
- **[Avalon](https://github.com/freriadoc/Avalon)** (freriadoc): `LLAMA_MOE_TAIL_SKIP` ports its
  `GGML_CUDA_MOE_TAIL_SKIP` (`510abcb1a`), which carries Strata's `STRATA_ROUTE_TAIL_SKIP`.
- Upstream llama.cpp contributors for `qwen4exp` model support (#27742) and the `draft-mtp`
  speculative mode the MTP head plugs into.
