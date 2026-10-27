// llama-prefill-dump (fork-chunk-loop): prefill one prompt as llama-server does for one slot (batches of n_batch,
// output on the last prompt token only, h_nextn on every row as the draft-mtp drafter asks), then write the bytes a
// bit-identity gate compares: every h_nextn row and every output logits row in token order (PREFIX.bin), the slot's
// memory state (PREFIX.state), and a summary with a digest per decode and the LLAMA_PREFILL_CHUNK counters (PREFIX.txt).
// Backend samplers are turned off: LLAMA_PREFILL_CHUNK refuses a context that has them.
//
// usage: llama-prefill-dump --prompt-file F --out PREFIX [--n-tokens N] [--all-logits] <llama-server args>
//   --all-logits: every row but each batch's first requests output (more rows compared; not the server's pattern)
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "../../src/llama-context.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static uint64_t fnv1a(const void * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    std::string prompt_path, out_prefix;
    int  n_tokens_max = 20000;
    bool all_logits   = false;

    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) { prompt_path  = argv[++i]; continue; }
        if (!strcmp(argv[i], "--out")         && i + 1 < argc) { out_prefix   = argv[++i]; continue; }
        if (!strcmp(argv[i], "--n-tokens")    && i + 1 < argc) { n_tokens_max = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--all-logits"))                  { all_logits   = true; continue; }
        rest.push_back(argv[i]);
    }
    if (prompt_path.empty() || out_prefix.empty()) {
        fprintf(stderr, "usage: %s --prompt-file F --out PREFIX [--n-tokens N] [--all-logits] <llama-server args>\n", argv[0]);
        return 1;
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    params.sampling.backend_sampling = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    std::string text;
    {
        std::ifstream f(prompt_path, std::ios::binary);
        if (!f) { LOG_ERR("cannot read %s\n", prompt_path.c_str()); return 1; }
        std::stringstream ss; ss << f.rdbuf(); text = ss.str();
    }

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("failed to load the model\n"); return 1; }

    // as common/speculative.cpp does on the target context for draft-mtp
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);

    std::vector<llama_token> toks = common_tokenize(ctx, text, true, false);
    if ((int) toks.size() > n_tokens_max) {
        toks.resize(n_tokens_max);
    }
    const int n_tok   = (int) toks.size();
    const int n_batch = (int) llama_n_batch(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd  = llama_model_n_embd_out(model);

    llama_memory_clear(llama_get_memory(ctx), true);

    FILE * fbin = fopen((out_prefix + ".bin").c_str(), "wb");
    FILE * ftxt = fopen((out_prefix + ".txt").c_str(), "w");
    if (!fbin || !ftxt) { LOG_ERR("cannot write %s.*\n", out_prefix.c_str()); return 1; }

    const char * env_c = getenv("LLAMA_PREFILL_CHUNK");
    const char * env_g = getenv("LLAMA_PREFILL_CHUNK_GROUP");
    fprintf(ftxt, "LLAMA_PREFILL_CHUNK=%s LLAMA_PREFILL_CHUNK_GROUP=%s n_tokens=%d n_batch=%d n_ubatch=%u n_vocab=%d n_embd_out=%d all_logits=%d\n",
            env_c ? env_c : "(unset)", env_g ? env_g : "(unset)", n_tok, n_batch, llama_n_ubatch(ctx), n_vocab, n_embd, all_logits);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    uint64_t h_all = 0xcbf29ce484222325ull;
    double t_all = 0.0;

    for (int i0 = 0; i0 < n_tok; i0 += n_batch) {
        const int n = std::min(n_batch, n_tok - i0);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            const bool last = i0 + i == n_tok - 1;
            common_batch_add(batch, toks[i0 + i], i0 + i, { 0 }, last || (all_logits && i > 0));
        }

        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("decode failed at token %d\n", i0);
            return 1;
        }
        llama_synchronize(ctx);
        const double dt = (ggml_time_us() - t0) / 1e6;
        t_all += dt;

        const float * h = llama_get_embeddings_nextn(ctx);
        if (!h) { LOG_ERR("no h_nextn rows\n"); return 1; }
        const size_t n_h = (size_t) n*n_embd;
        fwrite(h, sizeof(float), n_h, fbin);
        uint64_t hd = fnv1a(h, n_h*sizeof(float));

        int n_out = 0;
        for (int i = 0; i < n; ++i) {
            if (!batch.logits[i]) continue;
            const float * l = llama_get_logits_ith(ctx, i);
            fwrite(l, sizeof(float), n_vocab, fbin);
            hd = fnv1a(l, (size_t) n_vocab*sizeof(float), hd);
            n_out++;
        }
        h_all = fnv1a(&hd, sizeof(hd), h_all);

        fprintf(ftxt, "decode %d..%d outputs %d digest %016" PRIx64 " %.3f s %.0f t/s\n", i0, i0 + n, n_out, hd, dt, n/dt);
        fflush(ftxt);
        LOG_INF("decode %d..%d: %.3f s, %.0f t/s\n", i0, i0 + n, dt, n/dt);
    }

    {
        std::vector<uint8_t> st(llama_state_seq_get_size(ctx, 0));
        st.resize(llama_state_seq_get_data(ctx, st.data(), st.size(), 0));
        FILE * fst = fopen((out_prefix + ".state").c_str(), "wb");
        if (!fst) { LOG_ERR("cannot write %s.state\n", out_prefix.c_str()); return 1; }
        fwrite(st.data(), 1, st.size(), fst);
        fclose(fst);
        fprintf(ftxt, "state %zu bytes digest %016" PRIx64 "\n", st.size(), fnv1a(st.data(), st.size()));
    }

    fprintf(ftxt, "all digest %016" PRIx64 " prefill %.3f s %.0f t/s chunk_passes %" PRId64 " applies %" PRId64 "\n",
            h_all, t_all, n_tok/t_all, ctx->prefill_chunk_n_passes, ctx->prefill_chunk_n_apply);

    fclose(fbin);
    fclose(ftxt);
    llama_batch_free(batch);
    return 0;
}
