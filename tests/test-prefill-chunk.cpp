// PREFILL-CHUNK (tests/INVARIANTS.md): with LLAMA_PREFILL_CHUNK=C a prompt batch of several ubatches runs layer-major
// over chunks of C ubatches (LLAMA_PREFILL_CHUNK_GROUP=G layers per pass) and gives, bit for bit, the logits, the
// h_nextn rows and the memory state (KV cells and data, recurrent cells and states) of the micro-batch-major path,
// applying each ubatch to the memory exactly once. A tiny random qwen4exp (12 layers, attention every 4th, PLE on
// layer 1, MoE, QSA) is built in memory as test-llama-archs builds it, and decoded on the CPU by both paths.
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

static std::set<std::string> g_gpu_mask_notes; // LLAMA_PREFILL_CHUNK_GPU_MASK: lines the library logged

static const uint32_t N_VOCAB = 128;
static const uint32_t N_EMBD  = 256;
static const uint32_t N_HEAD  = 2;
static const uint32_t N_FF    = 384;
static const uint32_t N_LAYER = 12;
static const uint32_t N_CTX   = 512;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const size_t seed = *(const size_t *) userdata ^ std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 0.1f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & v : tmp) v = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & v : tmp) v = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the keys test-llama-archs (get_gguf_ctx) gives a qwen4exp MoE fixture, with more layers
static gguf_context_ptr get_gguf_ctx() {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    const uint32_t n_embd_head = N_EMBD / N_HEAD;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            N_CTX);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          N_EMBD);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               N_LAYER);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       N_FF);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    // the served model: attention on layers 3, 7, 11
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         N_CTX/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));

    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(N_LAYER, 4));
    {
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        const uint32_t ple_head_dim        = N_EMBD/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, N_VOCAB);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*N_VOCAB;
        }

        // as served: PLE on layer 1, which is not the first layer of a group of 1
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 1 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   n_embd_head);
    // a realistic top-k, so the QSA selection (and its block tables) varies with n_kv
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(16));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    {
        std::vector<std::string> tokenizer_list(N_VOCAB);
        std::vector<float>       tokenizer_scores(N_VOCAB, 0.0f);
        ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
        for (uint32_t i = 0; i < N_VOCAB; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        N_FF);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, N_FF / 2);
    ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,              N_FF);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,                      uint32_t(4));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                 uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,               uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,                1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,                 uint32_t(1));

    ms.add_kv(LLM_KV_SSM_INNER_SIZE,     uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,    uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,     uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, N_HEAD);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,    uint32_t(2));

    return ret;
}

struct run_cfg {
    uint32_t c;        // 0: LLAMA_PREFILL_CHUNK unset
    uint32_t g;
    bool     trace = false; // LLAMA_PREFILL_CHUNK_TRACE=1 for this run
    bool     gpu_mask = false; // LLAMA_PREFILL_CHUNK_GPU_MASK=1: the mask and QSA bias built by GGML_OP_MASK_FROM_META
};

struct scenario {
    const char * name;
    uint32_t n_seq;      // sequences in each batch
    uint32_t n_rs_seq;   // recurrent rollback snapshots
    bool     nextn_masked;
    std::vector<uint32_t> steps; // tokens per sequence in each decode call
    int      rollback;   // after step 0, remove this many tokens from the end of every sequence (MTP rejection); -1: all of them
    bool     first_out;  // the first token of each sequence requests output: a generating sequence, so no chunk path
    bool     unified = false; // one KV stream for all sequences (the served -kvu)
    bool     sampling = false; // backend samplers (the served --backend-sampling): output on each sequence's last token only
};

struct run_out {
    std::vector<std::vector<float>>   logits;  // per decode call, the output rows
    std::vector<std::vector<float>>   nextn;   // per decode call, the h_nextn rows
    std::vector<std::vector<uint8_t>> state;   // per decode call, llama_state_get_data
    std::vector<llama_context::input_record> inputs;
    std::vector<llama_token> sampled; // backend-sampled tokens, per decode call and output row
    int64_t n_apply  = 0;
    int64_t n_passes = 0;
    int64_t n_chunk_ub = 0; // ubatches the chunk path ran
    int64_t n_fill = 0, n_reuse = 0; // PREFILL-CHUNK-INPUTS
    // LLAMA_PREFILL_CHUNK_TRACE (the last decode's): input classes timed, input bytes, tensor_set calls and bytes
    size_t  tr_classes = 0; bool tr_mem_hybrid = false;
    int64_t tr_input_bytes = 0, tr_ts_calls = 0, tr_ts_bytes = 0;
    bool    ok = true;
};

static run_out run(llama_model * model, const scenario & sc, const run_cfg & cfg, const std::vector<llama_token> & toks) {
    // the trace (read per decode) is on for one configuration only: its runs must match the untraced baseline bit for bit
    if (cfg.trace) setenv("LLAMA_PREFILL_CHUNK_TRACE", "1", 1); else unsetenv("LLAMA_PREFILL_CHUNK_TRACE");
    if (cfg.gpu_mask) setenv("LLAMA_PREFILL_CHUNK_GPU_MASK", "1", 1); else unsetenv("LLAMA_PREFILL_CHUNK_GPU_MASK");
    if (cfg.c == 0) {
        unsetenv("LLAMA_PREFILL_CHUNK");
        unsetenv("LLAMA_PREFILL_CHUNK_GROUP");
    } else {
        setenv("LLAMA_PREFILL_CHUNK",       std::to_string(cfg.c).c_str(), 1);
        setenv("LLAMA_PREFILL_CHUNK_GROUP", std::to_string(cfg.g).c_str(), 1);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = N_CTX;
    cp.n_batch         = N_CTX;
    cp.n_ubatch        = 16;
    cp.n_seq_max       = sc.n_seq;
    cp.kv_unified      = sc.unified;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.n_rs_seq        = sc.n_rs_seq;
    cp.no_perf         = true;

    // the served sampler chain, one per sequence, each with its own seed
    std::vector<llama_sampler *> chains;
    std::vector<llama_sampler_seq_config> sconf;
    if (sc.sampling) {
        for (uint32_t s = 0; s < sc.n_seq; ++s) {
            llama_sampler * ch = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(ch, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(ch, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(ch, llama_sampler_init_temp(0.6f));
            llama_sampler_chain_add(ch, llama_sampler_init_dist(1234 + s));
            chains.push_back(ch);
            sconf.push_back({ (llama_seq_id) s, ch });
        }
        cp.samplers   = sconf.data();
        cp.n_samplers = sconf.size();
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    GGML_ASSERT(ctx);
    llama_set_embeddings_nextn(ctx, true, sc.nextn_masked);
    ctx->prefill_chunk_trace = true;

    run_out out;
    const int n_vocab = (int) N_VOCAB;
    const int n_embd_out = llama_model_n_embd_out(model);

    std::vector<llama_pos> pos(sc.n_seq, 0);
    size_t it = 0;

    for (size_t step = 0; step < sc.steps.size(); ++step) {
        const uint32_t n = sc.steps[step];
        llama_batch batch = llama_batch_init(n*sc.n_seq, 0, 1);
        // a prompt slice: no output on a sequence's first token in the batch, so the batch is prompt-only;
        // every other row requests output, so every logits row is compared
        for (uint32_t s = 0; s < sc.n_seq; ++s) {
            for (uint32_t i = 0; i < n; ++i) {
                const bool out = sc.sampling ? i + 1 == n : (i > 0 || sc.first_out);
                common_batch_add(batch, toks[it++ % toks.size()], pos[s]++, { (llama_seq_id) s }, out);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed (%s, step %zu)\n", sc.name, step);
            out.ok = false;
            llama_batch_free(batch);
            break;
        }

        std::vector<float> lg, nx;
        int n_out = 0;
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i]) continue;
            if (sc.sampling) {
                out.sampled.push_back(llama_get_sampled_token_ith(ctx, i));
                n_out++;
                continue;
            }
            const float * l = llama_get_logits_ith(ctx, i);
            lg.insert(lg.end(), l, l + n_vocab);
            n_out++;
        }
        const float * h = llama_get_embeddings_nextn(ctx);
        GGML_ASSERT(h);
        const int n_rows = sc.nextn_masked ? n_out : batch.n_tokens;
        nx.assign(h, h + (size_t) n_rows*n_embd_out);
        out.logits.push_back(std::move(lg));
        out.nextn.push_back(std::move(nx));

        std::vector<uint8_t> st(llama_state_get_size(ctx));
        st.resize(llama_state_get_data(ctx, st.data(), st.size()));
        out.state.push_back(std::move(st));

        llama_batch_free(batch);

        if (step == 0 && sc.rollback != 0) {
            for (uint32_t s = 0; s < sc.n_seq; ++s) {
                pos[s] = sc.rollback < 0 ? 0 : pos[s] - sc.rollback;
                if (!llama_memory_seq_rm(llama_get_memory(ctx), s, sc.rollback < 0 ? -1 : pos[s], -1)) {
                    fprintf(stderr, "rollback refused (%s)\n", sc.name);
                    out.ok = false;
                }
            }
        }
    }

    out.inputs   = ctx->prefill_chunk_inputs;
    out.n_apply  = ctx->prefill_chunk_n_apply;
    out.n_passes = ctx->prefill_chunk_n_passes;
    out.n_chunk_ub = ctx->prefill_chunk_n_chunk_ubatch;
    out.n_fill   = ctx->prefill_chunk_input_cache.n_fill;
    out.tr_classes     = ctx->prefill_trace_class_us.size();
    for (const auto & kv : ctx->prefill_trace_class_us) out.tr_mem_hybrid |= kv.first.find("llm_graph_input_mem_hybrid") != std::string::npos;
    out.tr_input_bytes = ctx->prefill_trace_input_bytes;
    out.tr_ts_calls    = ctx->prefill_trace_ts_calls;
    out.tr_ts_bytes    = ctx->prefill_trace_ts_bytes;
    out.n_reuse  = ctx->prefill_chunk_input_cache.n_reuse;

    llama_free(ctx);
    for (auto * ch : chains) llama_sampler_free(ch);
    return out;
}

static bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0);
}

int main(int argc, char ** argv) {
    size_t seed = 42;
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (strstr(text, "LLAMA_PREFILL_CHUNK_GPU_MASK:")) {
            g_gpu_mask_notes.insert(text);
        }
        if (level >= GGML_LOG_LEVEL_WARN && strstr(text, "LLAMA_PREFILL_CHUNK")) {
            fputs(text, stderr);
        }
    }, nullptr);

    // CPU only: the result must not depend on a device
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu);
    std::vector<ggml_backend_dev_t> devs = { cpu, nullptr };

    llama_model_params mp = llama_model_default_params();
    mp.devices = devs.data();

    gguf_context_ptr gguf_ctx = get_gguf_ctx();
    llama_model * model = llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, mp);
    GGML_ASSERT(model);

    std::vector<llama_token> toks(4096);
    {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(1, N_VOCAB - 1);
        for (auto & t : toks) t = dis(gen);
    }

    const std::vector<scenario> scenarios = {
        // 200 tokens: 13 ubatches of 16, the last one short; then a warm continuation of 90
        { "one-seq",          1, 0, false, { 200, 90 }, 0, false },
        { "one-seq-masked",   1, 0, true,  { 200, 90 }, 0, false },
        // rs_idx != 0 on entry: two tokens rolled back after the first batch, as an MTP rejection does
        { "rollback",         1, 2, false, { 120, 90 }, 2, false },
        // three sequences per batch, split_equal ubatches holding all three
        { "three-seq",        3, 0, false, { 70, 40 },  0, false },
        { "three-seq-rollbk", 3, 2, false, { 70, 40 },  1, false },
        // as served: one KV stream (-kvu), so split_equal fills ubatches across sequences
        { "three-seq-kvu",    3, 2, false, { 70, 40 },  1, false, true },
        // the sequence is cleared and its recurrent cell taken again by a new prompt: stale state in the cell
        { "reuse",            1, 0, false, { 100, 150 }, -1, false },
        // every token requests output, as a generating sequence's do: the micro-batch path, and the same result
        { "generating",       2, 0, false, { 60, 40 },  0, true },
        // backend sampling, as served: the tokens drawn must be the same (the dist sampler draws once per ubatch graph)
        { "sampling",         3, 2, false, { 70, 50, 40 }, 0, false, true, true },
    };

    const std::vector<run_cfg> cfgs = {
        { 2, 4 }, { 3, 4 }, { 5, 4 }, { 2, 1, true }, { 3, 1 }, { 4, 8 }, { 3, 12 },
        // the masks built from metadata by GGML_OP_MASK_FROM_META (here on the CPU backend)
        { 2, 1, false, true }, { 3, 4, false, true }, { 5, 12, false, true },
    };

    int n_fail = 0, n_check = 0;

    for (const auto & sc : scenarios) {
        const run_out base = run(model, sc, { 0, 0 }, toks);
        if (!base.ok || base.n_passes != 0) {
            fprintf(stderr, "FAIL %s: baseline run (ok %d, passes %" PRId64 ")\n", sc.name, base.ok, base.n_passes);
            n_fail++;
            continue;
        }

        for (const auto & cfg : cfgs) {
            const run_out cur = run(model, sc, cfg, toks);
            n_check++;

            bool ok = cur.ok;
            std::string why;

            // the chunk path must have run, or the comparison below proves nothing; and must not for a generating batch
            if (!sc.first_out && cur.n_passes == 0) { ok = false; why += " chunk-path-not-taken"; }
            if ( sc.first_out && cur.n_passes != 0) { ok = false; why += " chunk-path-taken-by-a-generating-batch"; }

            // every input a pass saw is the one the ubatch's single graph saw on the micro-batch path:
            // the memory metadata seen for (layer, ubatch) is the metadata apply() left for that ubatch
            {
                // ggml names an unnamed leaf by its place in the graph (leaf_<n>), which differs between graphs.
                // A backend sampler's uniform_<n> is written only for a sequence with an output in the ubatch (an
                // inactive sampler runs on a dummy row and its result is dropped): on either path the others hold
                // whatever the compute buffer held, so they are not compared; the sampled tokens are.
                // the GPU-mask runs' metadata inputs have no counterpart in the baseline (it sends the masks)
                auto skip  = [](const std::string & n) { return n.rfind("uniform_", 0) == 0 || n.find("_meta") != std::string::npos; };
                auto named = [](const std::string & n) { return !n.empty() && n.rfind("leaf_", 0) != 0; };
                std::map<std::pair<int64_t, std::string>, const std::vector<uint8_t> *> by_name;
                std::multiset<std::pair<int64_t, std::vector<uint8_t>>> unnamed;
                for (const auto & r : base.inputs) {
                    if (skip(r.name)) continue;
                    if (!named(r.name)) unnamed.insert({ r.ubatch, r.data });
                    else by_name[{ r.ubatch, r.name }] = &r.data;
                }
                int n_named = 0, n_bad = 0;
                std::string first_bad;
                for (const auto & r : cur.inputs) {
                    if (skip(r.name)) continue;
                    bool match;
                    if (!named(r.name)) {
                        match = unnamed.count({ r.ubatch, r.data }) > 0;
                    } else {
                        n_named++;
                        auto f = by_name.find({ r.ubatch, r.name });
                        match = f != by_name.end() && *f->second == r.data;
                    }
                    if (!match) {
                        if (n_bad++ == 0) first_bad = (named(r.name) ? r.name : std::string("(unnamed)")) + "@ub" + std::to_string(r.ubatch);
                    }
                }
                if (n_bad) { ok = false; why += " inputs " + std::to_string(n_bad) + " differ, first " + first_bad; }
                if (n_named == 0 && !sc.first_out) { ok = false; why += " no-inputs-traced"; }
                // the GPU-mask runs must have used the op: their passes send the mask's and the QSA bias's metadata
                if (cfg.gpu_mask && cur.n_passes > 0) {
                    bool kq = false, qsa = false;
                    for (const auto & r : cur.inputs) { kq |= r.name == "attn_inp_kq_meta_cells"; qsa |= r.name.rfind("qsa_blk_meta", 0) == 0; }
                    if (!kq)  { ok = false; why += " no-kq-meta"; }
                    if (!qsa) { ok = false; why += " no-qsa-meta"; }
                }
                if (!cfg.gpu_mask) {
                    for (const auto & r : cur.inputs) {
                        if (r.name.find("_meta") != std::string::npos) { ok = false; why += " meta-without-gpu-mask"; break; }
                    }
                }
            }
            // PREFILL-CHUNK-INPUTS: each chunk ubatch computes its KQ mask and its QSA set (one compress ratio in this
            // fixture) once; a group size that leaves several attention passes per ubatch reuses them
            if (!sc.first_out) {
                if (cur.n_fill != 2*cur.n_chunk_ub) { ok = false; why += " input-fills " + std::to_string(cur.n_fill) + "!=2x" + std::to_string(cur.n_chunk_ub); }
                if (cfg.g < N_LAYER && cur.n_reuse == 0) { ok = false; why += " no-input-reuse"; }
                if (cfg.g >= N_LAYER && cur.n_reuse != 0) { ok = false; why += " input-reuse-with-one-pass"; }
            }
            if (base.n_fill != 0 || base.n_reuse != 0) { ok = false; why += " baseline-used-the-cache"; }

            // the trace's split of set_inputs: filled when on (it times the mem_hybrid input, counts the input bytes and
            // the tensor_set calls of the token and position inputs), empty when off
            if (cfg.trace && cur.n_passes > 0) {
                if (cur.tr_classes == 0 || !cur.tr_mem_hybrid) { ok = false; why += " trace-no-input-classes"; }
                if (cur.tr_input_bytes <= 0)                     { ok = false; why += " trace-no-input-bytes"; }
                if (cur.tr_ts_calls <= 0 || cur.tr_ts_bytes <= 0) { ok = false; why += " trace-no-tensor_set"; }
                if (cur.tr_ts_bytes > cur.tr_input_bytes)        { ok = false; why += " trace-tensor_set-bytes>input-bytes"; }
            }
            if (!cfg.trace && (cur.tr_classes || cur.tr_input_bytes || cur.tr_ts_calls)) { ok = false; why += " trace-fields-set-while-off"; }

            // each ubatch applied once, as on the micro-batch path
            if (cur.n_apply != base.n_apply) { ok = false; why += " apply-count " + std::to_string(cur.n_apply) + "!=" + std::to_string(base.n_apply); }

            for (size_t s = 0; ok && s < base.logits.size(); ++s) {
                if (s >= cur.logits.size()) { ok = false; why += " missing-step"; break; }
                if (!same_bytes(base.logits[s], cur.logits[s])) { ok = false; why += " logits@" + std::to_string(s); }
                if (!same_bytes(base.nextn[s],  cur.nextn[s]))  { ok = false; why += " nextn@"  + std::to_string(s); }
                if (base.state[s] != cur.state[s])              { ok = false; why += " state@"  + std::to_string(s); }
            }
            if (base.sampled != cur.sampled) { ok = false; why += " sampled-tokens"; }
            if (sc.sampling && base.sampled.size() != 3*sc.steps.size()) { ok = false; why += " sampled-count " + std::to_string(base.sampled.size()); }

            printf("%-4s %-17s C=%u G=%-2u%s passes=%-4" PRId64 " applies=%" PRId64 " input fills=%" PRId64 " reuses=%" PRId64 "%s\n",
                    ok ? "ok" : "FAIL", sc.name, cfg.c, cfg.g, cfg.gpu_mask ? " gm" : "   ", cur.n_passes, cur.n_apply, cur.n_fill, cur.n_reuse, why.c_str());
            if (!ok) n_fail++;
        }
    }

    llama_model_free(model);
    llama_backend_free();

    // the device path announces itself once per process: both builds on the device, never a host fallback
    {
        int dev = 0, host = 0;
        for (const auto & t : g_gpu_mask_notes) {
            dev  += t.find("on the device") != std::string::npos;
            host += t.find("on the host")   != std::string::npos;
        }
        n_check++;
        const bool ok = dev == 2 && host == 0;
        printf("%s gpu-mask notes: %d on the device, %d on the host\n", ok ? "ok  " : "FAIL", dev, host);
        if (!ok) n_fail++;
    }
    printf("%d of %d checks failed\n", n_fail, n_check);
    return n_fail == 0 ? 0 : 1;
}
