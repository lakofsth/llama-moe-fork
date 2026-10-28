// PREFILL-CHUNK-GPU-MASK (tests/INVARIANTS.md): GGML_OP_MASK_FROM_META, on the CPU and (when present) on CUDA, gives
// byte for byte the masks the host writes: the attention KQ mask of llama_kv_cells_fill_kq_mask (what
// llama_kv_cache::set_input_kq_mask runs) from llama_kv_cells_fill_kq_meta's metadata, and the QSA per-block bias of
// llama_qsa_fill_blk_bias from llama_qsa_fill_blk_meta's. Random caches and ubatches: n_kv not a multiple of 256, one
// KV stream holding several sequences (unified KV) and several streams, cells of sequences the ubatch does not hold
// (other slots), empty cells, cells shared by two sequences, 2-D (M-RoPE) positions with ties, causal and not, F16 and
// F32. A CUDA device is used only when one is visible.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0, n_check = 0;
static void check(bool ok, const std::string & what) { n_check++; if (!ok) { n_fail++; printf("FAIL %s\n", what.c_str()); } }

struct backend { std::string name; ggml_backend_t be; };

// out = op(cells, toks) on a backend; returns the result's bytes
static std::vector<uint8_t> run_op(ggml_backend_t be, const std::vector<int32_t> & cells, int64_t n_cells,
        const std::vector<int32_t> & toks, int64_t n_tps, int64_t n_s, ggml_type type, ggml_mask_meta_mode mode) {
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * c = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_cells, n_s);
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_tps, n_s);
    ggml_tensor * o = ggml_mask_from_meta(ctx, c, t, type, mode);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(c, cells.data(), 0, ggml_nbytes(c));
    ggml_backend_tensor_set(t, toks.data(),  0, ggml_nbytes(t));
    ggml_backend_graph_compute(be, gf);
    std::vector<uint8_t> out(ggml_nbytes(o));
    ggml_backend_tensor_get(o, out.data(), 0, out.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// a host tensor of the given shape (for the host fills, which write ->data)
struct host_tensor {
    ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; ggml_tensor * t = nullptr;
    host_tensor(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    }
    ~host_tensor() { ggml_backend_buffer_free(buf); ggml_free(ctx); }
    std::vector<uint8_t> bytes() const { return std::vector<uint8_t>((uint8_t *) t->data, (uint8_t *) t->data + ggml_nbytes(t)); }
};

static void kq_case(std::mt19937 & rng, const std::vector<backend> & bes, int it, int64_t n_kv, int n_stream, int n_seq_cache,
        int n_seq_ub, int64_t n_tps, bool is_2d, bool causal, ggml_type type) {
    std::uniform_int_distribution<int> u(0, 1 << 20);
    // one cells array per KV stream; with one stream, n_seq_cache sequences share it (unified); sequence s lives in
    // stream s % n_stream, and the ubatch's stream k holds sequences k, k + n_stream, ... below n_seq_ub
    std::vector<llama_kv_cells> v_cells(n_stream);
    std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ, 0);
    for (int s = 0; s < n_seq_cache; ++s) seq_to_stream[s] = s % n_stream;
    for (int st = 0; st < n_stream; ++st) {
        auto & cells = v_cells[st];
        cells.resize((uint32_t) n_kv);
        for (int64_t j = 0; j < n_kv; ++j) {
            if (u(rng) % 7 == 0) continue; // empty
            const int s = (int) (u(rng) % n_seq_cache);
            if ((int) seq_to_stream[s] != st) continue;
            cells.pos_set((uint32_t) j, (llama_pos) (u(rng) % (n_tps + 300)));
            cells.seq_add((uint32_t) j, s);
            if (u(rng) % 9 == 0) { // shared by a second sequence of the same stream
                const int s2 = (s + n_stream) % n_seq_cache;
                if ((int) seq_to_stream[s2] == st && s2 != s) cells.seq_add((uint32_t) j, s2);
            }
            if (is_2d) {
                llama_kv_cell_ext e; e.x = u(rng) % 4; e.y = u(rng) % 4;
                cells.ext_set((uint32_t) j, e);
            }
        }
    }
    // the ubatch: n_stream streams of n_tps tokens; stream k's tokens belong to sequences mapping to KV stream k
    const int64_t n_tokens = n_tps*n_stream;
    const int n_pos = is_2d ? 4 : 1;
    std::vector<llama_pos> pos(n_tokens*n_pos);
    std::vector<int32_t> n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id> seq_data(n_tokens);
    std::vector<llama_seq_id *> seq_ptr(n_tokens);
    for (int k = 0; k < n_stream; ++k) {
        std::vector<int> mine;
        for (int s = 0; s < n_seq_ub; ++s) if ((int) seq_to_stream[s] == k) mine.push_back(s);
        if (mine.empty()) mine.push_back(k);
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = k*n_tps + ii;
            seq_data[i] = mine[u(rng) % mine.size()];
            pos[i] = (llama_pos) (u(rng) % (n_tps + 300));
            if (is_2d) { pos[i + n_tokens] = u(rng) % 4; pos[i + n_tokens*2] = u(rng) % 4; pos[i + n_tokens*3] = pos[i]; }
        }
    }
    for (int64_t i = 0; i < n_tokens; ++i) seq_ptr[i] = &seq_data[i];
    llama_ubatch ub = {};
    ub.n_tokens = (uint32_t) n_tokens; ub.n_pos = (uint32_t) n_pos; ub.pos = pos.data();
    ub.n_seq_id = n_seq_id.data(); ub.seq_id = seq_ptr.data();

    llama_hparams hp = {};
    hp.use_alibi = false;
    hp.non_causal_type = LLAMA_NON_CAUSAL_TYPE_ALL;

    host_tensor mask(type, n_kv, n_tps, 1, n_stream);
    llama_kv_cells_fill_kq_mask(hp, &ub, v_cells, seq_to_stream, 0, LLAMA_SWA_TYPE_NONE, mask.t, causal);
    const std::vector<uint8_t> ref = mask.bytes();

    host_tensor cm(GGML_TYPE_I32, 4, n_kv, n_stream, 1), tm(GGML_TYPE_I32, 4, n_tps, n_stream, 1);
    const bool ok = llama_kv_cells_fill_kq_meta(&ub, v_cells, seq_to_stream, cm.t, tm.t);
    char tag[200];
    snprintf(tag, sizeof(tag), "kq #%d n_kv=%lld streams=%d seqs=%d/%d n_tps=%lld 2d=%d causal=%d %s", it, (long long) n_kv,
             n_stream, n_seq_ub, n_seq_cache, (long long) n_tps, is_2d, causal, ggml_type_name(type));
    check(ok, std::string(tag) + ": metadata refused");
    if (!ok) return;
    const int mode = llama_kv_cells_kq_meta_mode(hp, LLAMA_SWA_TYPE_NONE, causal, is_2d);
    check(mode >= 0, std::string(tag) + ": no mode");
    const std::vector<int32_t> cv((int32_t *) cm.t->data, (int32_t *) cm.t->data + 4*n_kv*n_stream);
    const std::vector<int32_t> tv((int32_t *) tm.t->data, (int32_t *) tm.t->data + 4*n_tps*n_stream);
    for (const auto & b : bes) {
        const auto got = run_op(b.be, cv, n_kv, tv, n_tps, n_stream, type, (ggml_mask_meta_mode) mode);
        size_t first = 0;
        while (first < got.size() && first < ref.size() && got[first] == ref[first]) first++;
        check(got == ref, std::string(tag) + " on " + b.name + ": differs from the host fill at byte " + std::to_string(first));
    }
}

static void qsa_case(std::mt19937 & rng, const std::vector<backend> & bes, int it, int64_t n_blocks, int64_t n_tps, int n_seq) {
    std::uniform_int_distribution<int> u(0, 1 << 20);
    const int32_t n_bid = (int32_t) (u(rng) % (n_blocks + 1));
    std::vector<int32_t> bid_idx(n_blocks, 0);
    for (int32_t b = 0; b < n_bid; ++b) bid_idx[b] = 4*b + (b ? 0 : 0); // ascending block starts, as set_input_qsa has them
    std::vector<uint32_t> blk_set(n_blocks); // which of n_seq sequences each block holds
    for (auto & x : blk_set) x = (uint32_t) u(rng) & ((1u << n_seq) - 1);
    std::vector<llama_seq_id> tok_seq(n_tps);
    std::vector<int64_t> tok_tail(n_tps);
    for (int64_t ii = 0; ii < n_tps; ++ii) { tok_seq[ii] = u(rng) % n_seq; tok_tail[ii] = (u(rng) % (4*n_blocks + 8)) / 4 * 4; }
    const auto has = [&](int32_t b, llama_seq_id s) { return ((blk_set[b] >> s) & 1u) != 0; };

    std::vector<float> ref(n_blocks*n_tps);
    llama_qsa_fill_blk_bias(ref.data(), n_blocks, n_bid, bid_idx.data(), n_tps, tok_seq.data(), tok_tail.data(), has);
    std::vector<int32_t> bm(4*n_blocks), tmv(4*n_tps);
    const bool ok = llama_qsa_fill_blk_meta(bm.data(), tmv.data(), n_blocks, n_bid, bid_idx.data(), n_tps, tok_seq.data(), tok_tail.data(), has);
    char tag[160];
    snprintf(tag, sizeof(tag), "qsa #%d n_blocks=%lld n_bid=%d n_tps=%lld seqs=%d", it, (long long) n_blocks, n_bid, (long long) n_tps, n_seq);
    check(ok, std::string(tag) + ": metadata refused");
    const std::vector<uint8_t> refb((uint8_t *) ref.data(), (uint8_t *) ref.data() + ref.size()*sizeof(float));
    for (const auto & b : bes) {
        const auto got = run_op(b.be, bm, n_blocks, tmv, n_tps, 1, GGML_TYPE_F32, GGML_MASK_META_STRICT);
        check(got == refb, std::string(tag) + " on " + b.name + ": differs from the host bias");
    }
}

int main() {
    std::vector<backend> bes;
    bes.push_back({ "CPU", ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) });
    if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        bes.push_back({ ggml_backend_dev_name(gpu), ggml_backend_dev_init(gpu, nullptr) });
    } else {
        printf("no GPU device: CPU only\n");
    }

    std::mt19937 rng(1234);
    int it = 0;
    for (int64_t n_kv : { 256, 777, 1000, 4103 }) {
        for (int st : { 1, 2 }) {
            for (bool is_2d : { false, true }) {
                for (bool causal : { true, false }) {
                    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
                        // unified: 6 sequences in the cache, 3 in the ubatch; split: 2 streams, a sequence each
                        kq_case(rng, bes, it++, n_kv, st, st == 1 ? 6 : 2, st == 1 ? 3 : 2, st == 1 ? 64 : 48, is_2d, causal, type);
                    }
                }
            }
        }
    }
    for (int64_t n_blocks : { 1, 63, 257, 1000 }) {
        for (int n_seq : { 1, 3, 7 }) {
            qsa_case(rng, bes, it++, n_blocks, 64, n_seq);
        }
    }

    for (auto & b : bes) ggml_backend_free(b.be);
    printf("%d of %d checks failed (%zu backends)\n", n_fail, n_check, bes.size());
    return n_fail == 0 ? 0 : 1;
}
