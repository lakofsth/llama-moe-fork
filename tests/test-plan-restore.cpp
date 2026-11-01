// PLAN-PER-ARENA, llama level (tests/INVARIANTS.md PLAN-PER-ARENA): the conditions under which
// llama_context::plan_restore makes an output arena's graph current again from its kept scheduler plan, and the sites
// that drop kept plans. tests/test-sched-plan.cpp pins the scheduler mechanism (ggml_backend_sched_plan_*); this test
// pins what llama_context decides with it:
//   (1) a graph that reads the MoE placement (llm_graph_result::reads_moe_placement) is NOT restored once the placement
//       epoch has moved since its plan was kept (a repin or an adaptive-tier swap), and IS restored when it has not;
//   (2) a graph that does not read the placement (the MTP drafter's split-less NextN graph) IS restored across an
//       epoch move;
//   (3) every site in src/llama-context.cpp that drops kept plans actually drops them. The sites are derived from the
//       source at run time (argv[1]), and a site this test does not exercise fails it.
//
// A tiny random llama model is built in memory and decoded on the CPU, alternating a 3-token batch with no output
// (arena 0, the drafter's catch-up) and a 1-token batch with one output (arena 1, the drafter's draft step), so each
// arena keeps a plan while the other runs. On the CPU no heat split is built, so no graph reads the placement and
// nothing moves the placement epoch (both need a GPU): the test sets llm_graph_result::reads_moe_placement on the
// arena's graph after it is built, and moves the epoch the kept plan was made at (llama_context::gf_plan_epoch) away
// from the process-wide one, which is the comparison plan_restore makes. The bumps themselves (the call sites of
// moe_heat_repin and moe_heat_adapt in decode) are not reached here.
//
// Two sites are shadowed and no case here can tell their removal: sched_reserve's own `gf_plan_key = {0, 0}` and
// memory_update's graph_plans_clear are each followed, in the same call, by graph_reserve, which clears again. Their
// cases assert the state after the call; only the derived site set notices one of them going.
//
// The context's private state is read and written through `#define private public` around llama-context.h: the
// decision is private and has no API.

#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-graph.h"
#include "../src/llama-memory.h"
#include "../src/llama-model.h"

#define private public
#define protected public
#include "../src/llama-context.h"
#undef protected
#undef private

static const uint32_t N_VOCAB = 128;
static const uint32_t N_EMBD  = 128;
static const uint32_t N_HEAD  = 2;
static const uint32_t N_FF    = 192;
static const uint32_t N_LAYER = 2;
static const uint32_t N_CTX   = 256;

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); g_fail++; } } while (0)

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

static gguf_context_ptr get_gguf_ctx() {
    const llm_arch arch = LLM_ARCH_LLAMA;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,     llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,               N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,           N_CTX);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,         N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,              N_LAYER);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,      N_FF);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,     N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,  N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
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
    return ret;
}

// what one llama_decode did with the graph of the arena it ran in
enum class step_kind { REUSE, RESTORE, REBUILD, FAILED };

static const char * kind_name(step_kind k) {
    switch (k) {
        case step_kind::REUSE:   return "reuse";
        case step_kind::RESTORE: return "restore";
        case step_kind::REBUILD: return "rebuild";
        case step_kind::FAILED:  return "failed";
    }
    return "?";
}

struct driver {
    llama_model   * model;
    llama_context * ctx = nullptr;
    llama_pos       pos = 0;
    int             tok = 1;

    explicit driver(llama_model * m) : model(m) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = N_CTX;
        cp.n_batch         = N_CTX;
        cp.n_ubatch        = N_CTX;
        cp.n_seq_max       = 1;
        cp.n_threads       = 2;
        cp.n_threads_batch = 2;
        cp.no_perf         = true;
        ctx = llama_init_from_model(model, cp);
        GGML_ASSERT(ctx);
    }
    ~driver() { llama_free(ctx); }

    // one decode of n tokens, the last one requesting output when out is set; classified from the context's own state
    // before and after: the arena that ran, whether its graph was the active one before, and whether a graph was built
    step_kind step(int n, bool out) {
        const int arena = out ? 1 : 0;
        llm_graph_result * before_active = ctx->gf_res_prev_active;
        llm_graph_result * res_arena     = ctx->gf_res_prev[arena].get();
        const int32_t reused0 = ctx->n_reused;

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, (llama_token) (tok++ % (N_VOCAB - 1) + 1), pos++, { 0 }, out && i + 1 == n);
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return step_kind::FAILED;
        }
        GGML_ASSERT(ctx->gf_res_prev_active == ctx->gf_res_prev[arena].get()); // the arena we meant ran
        if (ctx->n_reused == reused0) {
            return step_kind::REBUILD;
        }
        GGML_ASSERT(ctx->n_reused == reused0 + 1);
        return before_active == res_arena ? step_kind::REUSE : step_kind::RESTORE;
    }

    step_kind catch_up() { return step(3, false); } // arena 0
    step_kind draft()    { return step(1, true);  } // arena 1

    // alternate until both arenas hold a kept plan and the last switch was a restore (positive control: the
    // mechanism this test refuses is live in this configuration)
    bool warm() {
        step_kind k = step_kind::FAILED;
        for (int i = 0; i < 3; ++i) {
            catch_up();
            k = draft();
        }
        return k == step_kind::RESTORE && ctx->gf_plan_key[0] != 0 && ctx->gf_plan_key[1] != 0;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// (1) and (2): the placement epoch

// graph reads the placement, epoch moved since its plan was kept -> rebuilt, not restored
static void test_reads_placement_epoch_moved(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(1) warm-up: alternating arenas never restored a kept plan");
    // the plan both arenas hold was made at the process-wide epoch; on the CPU nothing moves it
    CHECK(d.ctx->gf_plan_epoch[0] == d.ctx->gf_plan_epoch[1], "(1) the two arenas' plans were kept at different epochs");

    // arena 0's graph reads the placement, and a repin moved the epoch after its plan was kept
    d.ctx->gf_res_prev[0]->reads_moe_placement = true;
    d.ctx->gf_plan_epoch[0] ^= 1;
    const uint64_t key0 = d.ctx->gf_plan_key[0];

    const step_kind k = d.catch_up();
    CHECK(k == step_kind::REBUILD, "(1) a graph that reads the MoE placement was %s after the placement epoch moved "
          "(expected rebuild: its kept plan names the old placement)", kind_name(k));
    CHECK(d.ctx->gf_plan_key[0] != key0, "(1) arena 0 still holds the plan kept before the epoch moved");
    // the rebuilt graph is kept at the current epoch again, and restores from there
    CHECK(d.ctx->gf_plan_epoch[0] == d.ctx->gf_plan_epoch[1], "(1) the rebuilt plan was not kept at the current epoch");
    d.draft();
    const step_kind k2 = d.catch_up();
    CHECK(k2 == step_kind::RESTORE, "(1) after the rebuild, the next switch was %s (expected restore)", kind_name(k2));
}

// graph reads the placement, epoch NOT moved -> restored (the flag alone refuses nothing)
static void test_reads_placement_epoch_same(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(1b) warm-up: alternating arenas never restored a kept plan");
    d.ctx->gf_res_prev[0]->reads_moe_placement = true;
    const step_kind k = d.catch_up();
    CHECK(k == step_kind::RESTORE, "(1b) a graph that reads the MoE placement was %s with the epoch unmoved "
          "(expected restore)", kind_name(k));
}

// graph does not read the placement (the NextN drafter graph), epoch moved -> restored
static void test_no_placement_epoch_moved(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(2) warm-up: alternating arenas never restored a kept plan");
    d.catch_up(); // arena 0 current, arena 1's plan aside
    CHECK(!d.ctx->gf_res_prev[1]->reads_moe_placement, "(2) on the CPU a graph read the MoE placement");
    d.ctx->gf_plan_epoch[1] ^= 1;
    const uint64_t key1 = d.ctx->gf_plan_key[1];

    const step_kind k = d.draft();
    CHECK(k == step_kind::RESTORE, "(2) a graph that does not read the MoE placement was %s after the placement epoch "
          "moved (expected restore: nothing it reads changed)", kind_name(k));
    CHECK(d.ctx->gf_plan_key[1] == key1, "(2) arena 1's kept plan was replaced");
}

// ---------------------------------------------------------------------------------------------------------------------
// (3) the sites that drop kept plans

// the sites in src/llama-context.cpp, derived from the source: every call of graph_plans_clear() or
// ggml_backend_sched_plan_forget/_clear, and every assignment `gf_plan_key = {0, 0}` or `gf_plan_key[...] = 0`,
// named by the llama_context member function it is in
static std::multiset<std::string> derive_sites(const char * path) {
    std::multiset<std::string> out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        return out;
    }
    std::string line, fn;
    while (std::getline(f, line)) {
        // a member function definition starts at column 0: "<type> llama_context::<name>("
        if (!line.empty() && line[0] != ' ' && line[0] != '/' && line[0] != '#') {
            const size_t p = line.find("llama_context::");
            if (p != std::string::npos) {
                const size_t b = p + strlen("llama_context::");
                const size_t e = line.find('(', b);
                if (e != std::string::npos) {
                    fn = line.substr(b, e - b);
                }
            }
        }
        const size_t c = line.find("//");
        const std::string code = c == std::string::npos ? line : line.substr(0, c);
        const bool site = code.find("graph_plans_clear();") != std::string::npos ||
                          code.find("ggml_backend_sched_plan_forget(") != std::string::npos ||
                          code.find("ggml_backend_sched_plan_clear(") != std::string::npos ||
                          code.find("gf_plan_key = {0, 0}") != std::string::npos ||
                          (code.find("gf_plan_key[") != std::string::npos && code.find("] = 0;") != std::string::npos);
        if (site) {
            out.insert(fn);
        }
    }
    return out;
}

// graph_plans_clear itself: the keys and the scheduler's kept plans
static void test_clear_body(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(3) warm-up: alternating arenas never restored a kept plan");
    const uint64_t k0 = d.ctx->gf_plan_key[0];
    // positive control: arena 0's plan is aside and usable (this makes it current; arena 1's goes aside)
    CHECK(ggml_backend_sched_plan_use(d.ctx->sched.get(), k0), "(3) control: arena 0's kept plan was not usable");
    const uint64_t k1 = d.ctx->gf_plan_key[1];
    d.ctx->graph_plans_clear();
    d.ctx->gf_res_prev_active = nullptr; // as every caller does: the scheduler no longer holds what the graph assumed
    CHECK(d.ctx->gf_plan_key[0] == 0 && d.ctx->gf_plan_key[1] == 0, "(3) graph_plans_clear left a plan key");
    CHECK(!ggml_backend_sched_plan_use(d.ctx->sched.get(), k1), "(3) graph_plans_clear left the aside plan usable");
    CHECK(!ggml_backend_sched_plan_use(d.ctx->sched.get(), k0), "(3) graph_plans_clear left the current plan usable");
    // the context still decodes after it
    CHECK(d.catch_up() == step_kind::REBUILD, "(3) after graph_plans_clear the next decode did not rebuild");
}

static bool keys_clear(const llama_context * ctx) { return ctx->gf_plan_key[0] == 0 && ctx->gf_plan_key[1] == 0; }

static void test_site_sched_reserve(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(3 sched_reserve) warm-up failed");
    const int64_t n0 = d.ctx->n_sched_reserve;
    d.ctx->sched_need_reserve = true; // as every setter that changes the graph's shape does (n_threads, adapters, ...)
    d.ctx->sched_reserve();
    CHECK(d.ctx->n_sched_reserve == n0 + 1, "(3 sched_reserve) control: sched_reserve did not reserve");
    CHECK(keys_clear(d.ctx), "(3 sched_reserve) a plan key survived sched_reserve");
    CHECK(d.catch_up() != step_kind::FAILED, "(3 sched_reserve) decode failed after the reserve");
}

static void test_site_graph_reserve(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(3 graph_reserve) warm-up failed");
    CHECK(llama_graph_reserve(d.ctx, 1, 1, 1) != nullptr, "(3 graph_reserve) llama_graph_reserve failed");
    CHECK(keys_clear(d.ctx), "(3 graph_reserve) a plan key survived graph_reserve");
}

static void test_site_memory_update(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(3 memory_update) warm-up failed");
    // a context shift: the KV cache has a pending shift, which memory_update applies (a graph on the same scheduler)
    llama_memory_seq_add(llama_get_memory(d.ctx), 0, 0, -1, 1);
    const bool updated = d.ctx->memory_update(false);
    CHECK(updated, "(3 memory_update) control: the shift did not make memory_update run an update");
    CHECK(keys_clear(d.ctx), "(3 memory_update) a plan key survived the memory update");
    d.pos++;
    CHECK(d.catch_up() == step_kind::REBUILD, "(3 memory_update) the first decode after the update did not rebuild");
}

// process_ubatch: a rebuild of the arena that ran last drops that arena's plan, and only that plan; the other arena's
// plan, aside, survives it and is restored at the next switch
static void test_site_process_ubatch(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(3 process_ubatch) warm-up failed");
    CHECK(d.catch_up() == step_kind::RESTORE, "(3 process_ubatch) control: the switch to arena 0 did not restore");
    const uint64_t k0 = d.ctx->gf_plan_key[0];
    const uint64_t k1 = d.ctx->gf_plan_key[1];
    // arena 0 ran last; a different shape in arena 0 cannot reuse its graph -> rebuild in place
    CHECK(d.step(2, false) == step_kind::REBUILD, "(3 process_ubatch) a new shape in the same arena did not rebuild");
    CHECK(d.ctx->gf_plan_key[0] != k0 && d.ctx->gf_plan_key[0] != 0, "(3 process_ubatch) the rebuilt arena kept no new plan");
    CHECK(d.ctx->gf_plan_key[1] == k1, "(3 process_ubatch) the other arena's plan key changed");
    const step_kind k = d.draft();
    CHECK(k == step_kind::RESTORE, "(3 process_ubatch) after arena 0 rebuilt, arena 1's kept plan was %s "
          "(expected restore: the rebuild must drop only its own arena's plan)", kind_name(k));
    CHECK(!ggml_backend_sched_plan_use(d.ctx->sched.get(), k0), "(3 process_ubatch) the rebuilt arena's old plan is still usable");

    // a pass larger than a decode (> 32 tokens) keeps no plan: arena 0's rebuild drops its plan and leaves no key
    // naming it (the key is not overwritten by a new plan here, so this is the one place its reset is observable)
    CHECK(d.step(40, false) == step_kind::REBUILD, "(3 process_ubatch) a 40-token pass did not rebuild");
    CHECK(d.ctx->gf_plan_key[0] == 0, "(3 process_ubatch) after a rebuild that keeps no plan, arena 0 still names one");
}

static void test_site_opt_epoch_iter(llama_model * model) {
    driver d(model);
    CHECK(d.warm(), "(3 opt_epoch_iter) warm-up failed");

    // one datapoint of a whole context (common_opt_dataset_init: (n_tokens - n_ctx - 1)/stride datapoints)
    std::vector<llama_token> toks(2*N_CTX + 1);
    for (size_t i = 0; i < toks.size(); ++i) toks[i] = (llama_token) (i % (N_VOCAB - 1) + 1);
    ggml_opt_dataset_t dataset = common_opt_dataset_init(d.ctx, toks, N_CTX);
    GGML_ASSERT(ggml_opt_dataset_ndata(dataset) == 1);

    // llama_opt_init cannot be called on this fork: it walks llama_layer as an array of tensor pointers, and the fork's
    // llama_layer holds non-pointer members (llama_moe_split's vectors, dev_idx, online_counted), so it dereferences
    // garbage. What it sets up is made here instead, with one trainable tensor (the output norm, F32), which is all
    // llama_opt_epoch -> opt_epoch_iter needs to run its loop
    lr_opt lr;
    lr.lr0 = 1e-6f; lr.epochs = 1; lr.epoch = 0;
    lr.init();
    model->hparams.n_ctx_train = d.ctx->n_ctx();
    ggml_opt_params op = ggml_opt_default_params(d.ctx->sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    op.opt_period      = 1;
    op.get_opt_pars    = common_opt_lr_pars;
    op.get_opt_pars_ud = &lr;
    op.optimizer       = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    d.ctx->opt_ctx = ggml_opt_init(op);
    GGML_ASSERT(model->output_norm && model->output_norm->type == GGML_TYPE_F32);
    ggml_set_param(model->output_norm);

    ggml_opt_result_t rt = ggml_opt_result_init();
    ggml_opt_result_t re = ggml_opt_result_init();
    llama_opt_epoch(d.ctx, dataset, rt, re, ggml_opt_dataset_ndata(dataset), nullptr, nullptr);
    int64_t ndata = 0;
    ggml_opt_result_ndata(rt, &ndata);
    CHECK(ndata > 0, "(3 opt_epoch_iter) control: the epoch trained on nothing");
    CHECK(keys_clear(d.ctx), "(3 opt_epoch_iter) a plan key survived an optimizer epoch");
    ggml_opt_result_free(rt);
    ggml_opt_result_free(re);
    ggml_opt_dataset_free(dataset);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path of src/llama-context.cpp>\n", argv[0]);
        return 2;
    }
    // the mechanism under test: kept plans on (the default), graph reuse on, no GRAPH-STATS
    unsetenv("LLAMA_GRAPH_KEEP_PLANS");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu);
    std::vector<ggml_backend_dev_t> devs = { cpu, nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.devices = devs.data();

    size_t seed = 42;
    gguf_context_ptr gguf_ctx = get_gguf_ctx();
    llama_model * model = llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, mp);
    GGML_ASSERT(model);

    // (3) the derived site set must be the set exercised below: a new site, or one moved to another function, fails
    // here until it has a case
    {
        const std::multiset<std::string> want = {
            "graph_plans_clear",  // the body: ggml_backend_sched_plan_clear and the keys (test_clear_body)
            "graph_plans_clear",
            "sched_reserve",      // test_site_sched_reserve
            "memory_update",      // test_site_memory_update
            "process_ubatch",     // test_site_process_ubatch (plan_forget and the key)
            "process_ubatch",
            "graph_reserve",      // test_site_graph_reserve
            "opt_epoch_iter",     // test_site_opt_epoch_iter
        };
        const std::multiset<std::string> got = derive_sites(argv[1]);
        if (got != want) {
            std::string g;
            for (const auto & s : got) g += " " + s;
            CHECK(false, "(3) the plan-dropping sites in %s are{%s }, not the set this test exercises", argv[1], g.c_str());
        }
    }

    test_reads_placement_epoch_moved(model);
    test_reads_placement_epoch_same(model);
    test_no_placement_epoch_moved(model);
    test_clear_body(model);
    test_site_sched_reserve(model);
    test_site_graph_reserve(model);
    test_site_memory_update(model);
    test_site_process_ubatch(model);
    test_site_opt_epoch_iter(model);

    llama_model_free(model);
    llama_backend_free();

    if (g_fail) {
        fprintf(stderr, "test-plan-restore: %d check(s) failed\n", g_fail);
        return 1;
    }
    printf("test-plan-restore: OK\n");
    return 0;
}
