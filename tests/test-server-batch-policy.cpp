// PREFILL-CHUNK-MIXED (tests/INVARIANTS.md): under LLAMA_PREFILL_CHUNK a server batch that carries a generating token
// holds at most LLAMA_PREFILL_CHUNK_MIXED_BATCH tokens (default 2048, the served -b); one without fills to n_batch;
// without LLAMA_PREFILL_CHUNK (or C < 2) nothing changes. The cap is tools/server/server-batch-policy.h, applied by
// update_slots after the generating slots' tokens are in the batch and before any prompt token is added.
#include "../tools/server/server-batch-policy.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int n_fail = 0, n_check = 0;

static void check(bool ok, const std::string & what) {
    n_check++;
    if (!ok) { n_fail++; printf("FAIL %s\n", what.c_str()); }
}

static server_batch_policy with(const char * chunk, const char * mixed) {
    if (chunk) setenv("LLAMA_PREFILL_CHUNK", chunk, 1); else unsetenv("LLAMA_PREFILL_CHUNK");
    if (mixed) setenv("LLAMA_PREFILL_CHUNK_MIXED_BATCH", mixed, 1); else unsetenv("LLAMA_PREFILL_CHUNK_MIXED_BATCH");
    return server_batch_policy::from_env();
}

int main() {
    // today: no chunk, no cap, whatever is generating
    for (const char * c : { (const char *) nullptr, "0", "1" }) {
        const auto p = with(c, "512");
        for (int g : { 0, 1, 3, 9 }) {
            check(p.cap(8192, g) == 8192, std::string("chunk=") + (c ? c : "unset") + " gen=" + std::to_string(g) + ": n_batch changed");
        }
    }
    // the recipe: chunk on, served -b 8192
    {
        const auto p = with("8", nullptr);
        check(p.cap(8192, 0) == 8192, "prompt-only batch: must fill to n_batch");
        check(p.cap(8192, 1) == 2048, "one generating token: the served 2048");
        check(p.cap(8192, 3) == 2048, "a sampled token and two drafts: the served 2048");
        check(p.cap(1024, 1) == 1024, "n_batch below the cap: n_batch");
    }
    {
        const auto p = with("4", "4096");
        check(p.cap(8192, 2) == 4096 && p.cap(8192, 0) == 8192, "LLAMA_PREFILL_CHUNK_MIXED_BATCH=4096");
    }
    {
        const auto p = with("8", "0");
        check(p.cap(8192, 5) == 8192, "LLAMA_PREFILL_CHUNK_MIXED_BATCH=0: no cap");
    }
    printf("%d of %d checks failed\n", n_fail, n_check);
    return n_fail == 0 ? 0 : 1;
}
