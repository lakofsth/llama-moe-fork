// PREFILL-CHUNK-PF-ONLY (tests/INVARIANTS.md): under LLAMA_PREFILL_CHUNK with LLAMA_MOE_PREFETCH_PP, only a chunk
// loop pass reads its host experts through the prefetch ring (unless LLAMA_MOE_PREFETCH_CHUNK_ONLY=0); without the
// chunk loop the prefetch applies to every wide graph as before. src/llama-moe-prefetch-policy.h, as build_moe_ffn uses it.
#include "../src/llama-moe-prefetch-policy.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int n_fail = 0, n_check = 0;
static void check(bool ok, const std::string & what) { n_check++; if (!ok) { n_fail++; printf("FAIL %s\n", what.c_str()); } }

static void env(const char * k, const char * v) { if (v) setenv(k, v, 1); else unsetenv(k); }

static llama_moe_prefetch_policy with(const char * pf, const char * chunk, const char * only) {
    env("LLAMA_MOE_PREFETCH_PP", pf);
    env("LLAMA_PREFILL_CHUNK", chunk);
    env("LLAMA_MOE_PREFETCH_CHUNK_ONLY", only);
    env("LLAMA_MOE_PREFETCH_PP_MIN", nullptr);
    return llama_moe_prefetch_policy::from_env();
}

int main() {
    {   // the stock prefetch (no chunk loop): every wide graph, chunk pass or not
        const auto p = with("2", nullptr, nullptr);
        check(p.use(true, true, 1024, false), "no chunk: a 1024-token graph uses the ring");
        check(!p.use(true, true, 1023, false), "no chunk: below LLAMA_MOE_PREFETCH_PP_MIN it does not");
        check(!p.use(false, true, 4096, false) && !p.use(true, false, 4096, false), "no pf map / no op_offload: no ring");
    }
    {   // the recipe: chunk loop on, the micro-batch path keeps the stock routed upload
        const auto p = with("3", "8", nullptr);
        check(p.use(true, true, 1024, true), "chunk: a chunk pass uses the ring");
        check(!p.use(true, true, 2048, false), "chunk: a micro-batch graph (a generating batch) does not");
    }
    {
        const auto p = with("3", "8", "0");
        check(p.use(true, true, 2048, false), "LLAMA_MOE_PREFETCH_CHUNK_ONLY=0: every wide graph, as before");
    }
    {
        const auto p = with("2", "1", nullptr);
        check(p.use(true, true, 2048, false), "LLAMA_PREFILL_CHUNK=1 is off: every wide graph");
        const auto q = with("2", nullptr, "1");
        check(!q.use(true, true, 2048, false) && q.use(true, true, 2048, true), "LLAMA_MOE_PREFETCH_CHUNK_ONLY=1 without the chunk loop");
        const auto r = with(nullptr, "8", nullptr);
        check(!r.use(true, true, 2048, true), "prefetch unset: never");
    }
    printf("%d of %d checks failed\n", n_fail, n_check);
    return n_fail == 0 ? 0 : 1;
}
