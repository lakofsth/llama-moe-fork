// llama-moe-route-trace — per-segment expert routing counts, for asking whether a turn's PROMPT
// routing predicts the routing of the tokens decoded after it (fork/repin-trigger, 2026-10-02).
//
// Input (--segments FILE): a sequence of segments, each a header line then exactly <nbytes> of text
// and a newline:
//     @@SEG <conversation-id> <kind> <nbytes>
// Segments of one conversation are evaluated in order on ONE continuing KV sequence, as a server
// with a prompt cache sees them: each segment is prefilled on top of everything before it. A new
// conversation id clears the KV. <kind> is a free label (prompt / completion) copied to the output.
// Feeding a completion as a batch gives each token the routing it had when it was decoded: the
// model is causal, so the hidden state at a position depends only on the tokens before it.
//
// Output (--out FILE): one line per segment:
//     <conversation-id> <kind> <seg-index> <n_tokens> <n_layer> <n_expert> then, per layer,
//     <total> followed by <n_expert> counts
// Counts come from the CPU mul_mat_id ids sweep (ggml-cpu.c), so run with every expert host-side
// and nothing offloaded, which makes them exact:
//     LLAMA_MOE_HEAT_ONLINE=1 LLAMA_MOE_COUNT_ALL=1 LLAMA_MOE_REPIN_MIN_HITS=9000000000000000000
//     GGML_OP_OFFLOAD_MIN_BATCH=1000000  ... -ngl 99 --n-cpu-moe <n_layer>
// (MIN_HITS that high keeps the online repin block from decaying the counters mid-segment.)
// Every token is decoded as an output: the graph computes the last layer for output rows only,
// so anything less leaves the last layer's routing unmeasured.
// A segment that would overflow the context ends that conversation: the rest of it is skipped and
// logged, never truncated silently.
//
// --ids-out FILE (with LLAMA_MOE_TRACE_IDS=1) also writes the routing in TOKEN ORDER, for replaying
// a cache policy step by step: a 16-byte header "MOEIDS1\0", int32 n_layer, int32 n_used, then for
// every segment written to --out, in the same order, n_tokens x n_layer x n_used uint16 expert ids
// (0xFFFF for a layer with no routed experts). File size = 16 + 2*n_layer*n_used*sum(n_tokens).
// Each layer's ids are swept once per expert matmul, so the tap holds 2-3 copies per decode; they
// must agree exactly or the run stops. Needs -ub equal to -b (one graph per decode).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct segment { std::string conv, kind, text; };

static bool read_segments(const std::string & path, std::vector<segment> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { LOG_ERR("cannot open %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream h(line);
        std::string tag; segment s; long long n = -1;
        if (!(h >> tag >> s.conv >> s.kind >> n) || tag != "@@SEG" || n < 0) {
            LOG_ERR("bad segment header: %.80s\n", line.c_str()); return false;
        }
        s.text.resize((size_t) n);
        if (n > 0 && !f.read(&s.text[0], n)) { LOG_ERR("short segment body in %s\n", s.conv.c_str()); return false; }
        f.get(); // the newline after the body
        out.push_back(std::move(s));
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string seg_path, out_path, ids_path;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--segments") && i + 1 < argc) { seg_path = argv[++i]; continue; }
        if (!strcmp(argv[i], "--out")      && i + 1 < argc) { out_path = argv[++i]; continue; }
        if (!strcmp(argv[i], "--ids-out")  && i + 1 < argc) { ids_path = argv[++i]; continue; }
        rest.push_back(argv[i]);
    }
    if (seg_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s --segments FILE --out FILE [--ids-out FILE] <llama args: -m MODEL -c N -b N -ub N ...>\n", argv[0]);
        return 1;
    }
    const char * cnt_all = getenv("LLAMA_MOE_COUNT_ALL");
    const char * online  = getenv("LLAMA_MOE_HEAT_ONLINE");
    if (!cnt_all || !atoi(cnt_all) || !online || !atoi(online)) {
        fprintf(stderr, "set LLAMA_MOE_HEAT_ONLINE=1 and LLAMA_MOE_COUNT_ALL=1 (see the header) — without them every count is 0\n");
        return 1;
    }
    if (!ids_path.empty()) {
        const char * tr = getenv("LLAMA_MOE_TRACE_IDS");
        if (!tr || !atoi(tr)) { fprintf(stderr, "--ids-out needs LLAMA_MOE_TRACE_IDS=1\n"); return 1; }
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    std::vector<segment> segs;
    if (!read_segments(seg_path, segs)) return 1;

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("failed to load the model\n"); return 1; }

    char buf[256];
    std::string arch = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0 ? buf : "";
    const int n_expert = llama_model_meta_val_str(model, (arch + ".expert_count").c_str(), buf, sizeof(buf)) > 0 ? atoi(buf) : 0;
    const int n_layer  = llama_model_n_layer(model);
    const int n_ctx    = (int) llama_n_ctx(ctx);
    const int n_batch  = (int) llama_n_batch(ctx);
    if (n_expert <= 0) { LOG_ERR("no %s.expert_count: not an MoE model\n", arch.c_str()); return 1; }

    FILE * out = fopen(out_path.c_str(), "w");
    if (!out) { LOG_ERR("cannot write %s\n", out_path.c_str()); return 1; }

    FILE * ids_out = NULL;
    int n_used = 0, copies = 0;            // learned from the first decode, then held fixed
    std::vector<int> moe_layer(n_layer, -1); // -1 unknown, 0 dense, 1 routed
    std::vector<uint16_t> seg_ids;        // the current segment's [token][layer][n_used]
    if (!ids_path.empty()) {
        if ((int) llama_n_ubatch(ctx) != n_batch) {
            LOG_ERR("--ids-out needs -ub equal to -b (got %d vs %d): split ubatches interleave the tap's copies\n",
                    (int) llama_n_ubatch(ctx), n_batch);
            return 1;
        }
        ids_out = fopen(ids_path.c_str(), "wb");
        if (!ids_out) { LOG_ERR("cannot write %s\n", ids_path.c_str()); return 1; }
    }
    // drain the tap after one decode of n tokens into seg_ids; false = the tap is inconsistent
    auto drain = [&](int n) -> bool {
        const size_t base = seg_ids.size();
        if (n_used > 0) seg_ids.resize(base + (size_t) n*n_layer*n_used, 0xFFFF);
        for (int il = 0; il < n_layer; il++) {
            int64_t len = 0; int32_t k = 0;
            const int32_t * v = ggml_cpu_moe_trace_ids(il, &len, &k);
            const int routed = len > 0 ? 1 : 0;
            if (moe_layer[il] < 0) moe_layer[il] = routed;
            if (moe_layer[il] != routed) { LOG_ERR("layer %d: routed in one decode, not in another\n", il); return false; }
            if (!routed) continue;
            if (n_used == 0) {
                n_used = k;
                seg_ids.resize(base + (size_t) n*n_layer*n_used, 0xFFFF);
            }
            if (k != n_used) { LOG_ERR("layer %d: %d ids per token, layer 0's width is %d\n", il, k, n_used); return false; }
            const int64_t one = (int64_t) n*k;
            if (len % one != 0) { LOG_ERR("layer %d: %lld ids is not a whole number of %d-token copies\n", il, (long long) len, n); return false; }
            const int c = (int) (len/one);
            if (copies == 0) copies = c;
            if (c != copies) { LOG_ERR("layer %d: %d copies, expected %d\n", il, c, copies); return false; }
            for (int r = 1; r < c; r++) {
                if (memcmp(v, v + r*one, (size_t) one*sizeof(int32_t)) != 0) {
                    LOG_ERR("layer %d: sweep copy %d disagrees with copy 0\n", il, r); return false;
                }
            }
            for (int t = 0; t < n; t++) for (int j = 0; j < k; j++) {
                const int32_t e = v[(int64_t) t*k + j];
                if (e < 0 || e >= n_expert) { LOG_ERR("layer %d: id %d outside [0,%d) — sentinel ids mean a split graph, not a trace run\n", il, e, n_expert); return false; }
                seg_ids[base + ((size_t) t*n_layer + il)*n_used + j] = (uint16_t) e;
            }
        }
        if (n_used == 0) { LOG_ERR("no layer routed any token — is LLAMA_MOE_COUNT_ALL set?\n"); return false; }
        return true;
    };

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::string conv; std::string skipping;
    int pos = 0, seg_idx = 0, n_written = 0, n_skipped = 0;
    for (const auto & s : segs) {
        if (s.conv != conv) {
            conv = s.conv; pos = 0; seg_idx = 0; skipping.clear();
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        if (skipping == conv) { n_skipped++; continue; }
        const auto toks = common_tokenize(ctx, s.text, /*add_special*/ pos == 0, /*parse_special*/ true);
        if (pos + (int) toks.size() > n_ctx) {
            LOG_WRN("%s: segment %d (%zu tokens) would pass n_ctx %d at pos %d — skipping the rest of this conversation\n",
                    conv.c_str(), seg_idx, toks.size(), n_ctx, pos);
            skipping = conv; n_skipped++; continue;
        }
        ggml_cpu_moe_online_reset();
        seg_ids.clear();
        for (size_t i = 0; i < toks.size(); i += n_batch) {
            common_batch_clear(batch);
            const size_t n = std::min(toks.size() - i, (size_t) n_batch);
            for (size_t j = 0; j < n; j++) {
                // every token an output: the graph prunes the LAST layer to output rows
                // (inp_out_ids), so with logits on the final token only, the last layer's
                // routing would cover one token per batch. Costs n_batch x n_vocab floats.
                common_batch_add(batch, toks[i + j], pos++, { 0 }, true);
            }
            ggml_cpu_moe_trace_reset();
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed at pos %d\n", conv.c_str(), pos);
                fclose(out); llama_batch_free(batch); return 1;
            }
            if (ids_out) {
                llama_synchronize(ctx);
                if (!drain((int) n)) {
                    LOG_ERR("%s seg %d: routing tap inconsistent; stopping\n", conv.c_str(), seg_idx);
                    fclose(out); fclose(ids_out); llama_batch_free(batch); return 1;
                }
            }
        }
        llama_synchronize(ctx);
        if (ids_out) {
            if (ftell(ids_out) == 0 && n_used > 0) {
                const int32_t hdr[2] = { n_layer, n_used };
                fwrite("MOEIDS1", 1, 8, ids_out); fwrite(hdr, sizeof(int32_t), 2, ids_out);
            }
            if (!seg_ids.empty()) fwrite(seg_ids.data(), sizeof(uint16_t), seg_ids.size(), ids_out);
            fflush(ids_out);
        }
        fprintf(out, "%s %s %d %zu %d %d", conv.c_str(), s.kind.c_str(), seg_idx, toks.size(), n_layer, n_expert);
        int64_t grand = 0;
        for (int il = 0; il < n_layer; il++) {
            int64_t sent = 0, tot = 0;
            const int64_t * c = ggml_cpu_moe_online_counts(il, &sent, &tot);
            fprintf(out, " %lld", (long long) tot);
            for (int e = 0; e < n_expert; e++) fprintf(out, " %lld", c ? (long long) c[e] : 0LL);
            grand += tot;
        }
        fprintf(out, "\n");
        fflush(out);
        if (!toks.empty() && grand == 0) {
            LOG_ERR("%s seg %d: %zu tokens and zero routings counted — the expert matmuls are not running on the CPU "
                    "(need --n-cpu-moe all layers and GGML_OP_OFFLOAD_MIN_BATCH above the batch); stopping\n",
                    conv.c_str(), seg_idx, toks.size());
            fclose(out); llama_batch_free(batch); return 1;
        }
        seg_idx++; n_written++;
    }
    fclose(out);
    if (ids_out) fclose(ids_out);
    llama_batch_free(batch);
    LOG_INF("wrote %d segments, skipped %d\n", n_written, n_skipped);
    return 0;
}
