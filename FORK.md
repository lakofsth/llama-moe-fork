# llama.cpp fork — hybrid MoE streaming (GPU + RAM + NVMe tail)

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at `14d3ba4` (b9994),
branch `fork/phase-a`. Targets the *hybrid* MoE regime: the model nearly fits across
GPU + system RAM with a small NVMe tail — between the all-VRAM and all-disk cases.
Developed and measured on DeepSeek-V4-Flash UD-IQ4_XS on a single RTX 5090 (32 GB) +
128 GB RAM. Four upstream commits are cherry-picked with original authorship preserved.

## What it adds

- **Heat-driven expert split** — pack the globally hottest experts into VRAM from a
  profiled per-layer heat map; a sentinel id encoding splits `mul_mat_id` so resident
  experts compute on GPU, the rest on CPU. Placement is invisible to the math.
- **Online heat (self-tuning)** — routing counters piggyback on the CPU id sweep at zero
  cost; when the GPU hit-rate decays (workload/register drift), the runtime re-ranks and
  repins experts mid-run. Starting on a deliberately wrong map recovers ~86 % of the
  matched-map ceiling autonomously.
- **Router locality bias** — add a small ε to GPU-resident experts' selection scores
  pre-top-k (mixture weights stay unbiased). A speed/quality dial.
- **Preventive quality gate (default on)** — per token per layer, in-graph: the biased
  selection is kept only if it retains ≥ (1−δ) of the unbiased top-k's probability mass;
  otherwise that token falls back to unbiased routing *before* any expert runs.
- **Bias-aware repin threshold** — the routing counters see post-bias selections, so the
  repin trigger compensates for the bias-inflated hit-rate (measured freeze mode otherwise).

## Environment variables

| var | default | meaning |
|---|---|---|
| `LLAMA_MOE_HEAT_FILE` | unset (off) | per-layer×expert f32 heat map; enables the split |
| `LLAMA_MOE_HEAT_VRAM_MB` | 20000 | VRAM budget for packed experts |
| `LLAMA_MOE_HEAT_ONLINE` | 0 | enable online counting + auto-repin |
| `LLAMA_MOE_REPIN_THRESH` | 0.45 | hit-rate floor that triggers a repin |
| `LLAMA_MOE_REPIN_MIN_HITS` | 20000 | min window sample before judging |
| `LLAMA_MOE_REPIN_BIAS_K` | 0.4 | threshold compensation per unit of router bias |
| `LLAMA_MOE_ROUTER_BIAS` | 0 | ε added to resident experts' selection scores |
| `LLAMA_MOE_BIAS_GATE` | 0.05 | max relative routed-mass displacement per token; `0` = ungated (logs a warning: measured +2.6–5.2 % PPL) |
| `LLAMA_MOE_HEAT_TRACE` | 0 | per-window hit-rate trace (use with `-lv 2`) |
| `LLAMA_MOE_SPLIT_PP` | — | split path for prompt-processing batches |
| `LLAMA_MOE_PREFETCH` | — | router-lookahead async prefetch (phase A) |
| `LLAMA_MMAP_NO_PREFETCH` | 0 | skip whole-file MADV_WILLNEED at load |

Repin/trace events log at WARN; `llama-cli` defaults to error-only, so pass `-lv 2`.

## Measured (fi-prose register, 5090 + DDR5, tg tokens/s)

| config | tg | PPL delta |
|---|---|---|
| dense offload baseline (`-ncmoe`) | ~11 | — |
| heat split, matched map | ~22 | 0 (placement alone can't change PPL) |
| + online, starting from a *wrong* map | 19–25 avg | 0 after repin |
| + bias ε=0.5, gate δ=0.05 | **30** | +0.2 % (noise) |
| + bias ε=0.5, ungated | 40 | +2.6 % |

Only two deepseek4-specific points exist (the init gate in `llama-model.cpp` and the
`moe_heat_repin` override); the rest rides the shared MoE graph path — porting to other
MoE architectures is planned.
