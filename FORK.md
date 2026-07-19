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
| `LLAMA_MOE_BIAS_GATE` | 0.05 | max relative routed-mass displacement per token; `0` = ungated (logs a warning — measured cost at ε=0.5: +2.6 % PPL on a narrow register but **+22 % on a wide one, +29 % on a mismatched map**; the gate holds all of these at noise level) |
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

## Architecture support

The machinery is generic — it operates on the standard `ffn_*_exps` expert layout and the
shared `build_moe_ffn` graph path. Each architecture is enabled by a whitelist entry plus a
two-line builder change, and ships only after validation on real hardware:

| arch | status |
|---|---|
| `deepseek4` | validated (DeepSeek-V4-Flash — all numbers above) |
| `qwen3moe` | validated (Qwen3-30B-A3B Q6, flat-map bootstrap: +32 % tg over `-ncmoe` baseline, 84 % hit-rate with no profiled map) |
| `minimax-m2` | validated (MiniMax-M2.7 IQ3, 75 G true hybrid regime: a 6 GB expert budget captures 76 % of routing on 256-expert layers, +9 % tg at that deliberately small budget) |
| `qwen35moe` | validated (Qwen3.6-35B-A3B Q5, merged gate_up: +21 % tg, flat-map bootstrap) |
| `deepseek2` | validated (GLM-4.7-Flash Q4 — converts of GLM-4.7 declare this arch — MLA + merged gate_up: **+39 % tg**, flat-map bootstrap) |
| `hy-v3` | validated (Tencent Hy3 295B IQ1_M, 89 G, hybrid regime: 8.4 → 12.7 t/s (+51 %) at a 14 GB expert budget, 85.6 % hit-rate. Budget lesson: leave the GPU headroom for compute buffers — 22 GB starved context allocation) |
| `kimi-linear` | validated (Kimi-Linear-48B-A3B, self-quantized Q4_K_M — no community GGUFs exist: +20 % tg, flat-map bootstrap) |
| `glm4-moe` | plumbing in place; awaiting a genuine glm4moe GGUF for validation |
| `bailingmoe2` (Ling-2.6-flash), `qwen3next`, `step35` | next — two-line recipe; merged gate_up supported |

Models without a profiled heat map bootstrap from a **flat map**: online repin measures the
real per-expert heat during the first hundreds of tokens and repacks VRAM by itself —
self-profiling, no logging pipeline needed.

## What to run on what hardware

Guidance, not law — tg depends heavily on RAM bandwidth. "Hybrid regime" means the model
spans GPU + RAM with at most a modest NVMe tail; that's where this fork pays.

| hardware class | suggestion |
|---|---|
| 10–12 GB VRAM + 64 GB RAM (e.g. 3080 desktop) | Once `glm4moe` validates: **GLM-4.7-Flash Q4_K_XL (~18 GB)** — honest 4-bit quality, fits RAM+VRAM easily. Adventurous today: DeepSeek-V4-Flash at a ~2-bit dynamic quant (~65–70 GB, NVMe tail, unmeasured — expect single-digit tg, rough 2-bit quality) |
| 24–32 GB VRAM + 96–128 GB RAM (e.g. 4090/5090 workstation) | DeepSeek-V4-Flash UD-IQ4_XS (129 GB) — the measured configuration above |
| CPU-heavy boxes, 192 GB+ RAM, small GPU | Large MoEs at 4-bit (V4-Flash-class and up); the heat split keeps the small card saturated with the hot experts |

Quant-size rule of thumb for the hybrid budget: fit ≈ VRAM + RAM − (OS + KV cache + ~10 %
headroom); anything past that streams from NVMe, which the placement tolerates but tg pays for.
