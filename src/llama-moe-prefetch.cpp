#include "llama-moe-prefetch.h"

#include "llama-model.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

// dequantize a small router-side tensor (F32/F16/BF16) to f32
static bool tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t || !t->buffer || !t->data) return false; // not materialized (e.g. fit-params probe context)
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    switch (t->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), out.data(), 0, n*sizeof(float));
            return true;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), tmp.data(), 0, n*sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(tmp[i]);
            return true;
        }
        case GGML_TYPE_BF16: {
            std::vector<uint16_t> tmp(n);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), tmp.data(), 0, n*sizeof(uint16_t));
            for (int64_t i = 0; i < n; i++) {
                uint32_t u = ((uint32_t) tmp[i]) << 16;
                memcpy(&out[i], &u, 4);
            }
            return true;
        }
        default:
            return false;
    }
}

struct expert_slice { char * addr; size_t len; };

struct layer_info {
    // router snapshot (empty for hash layers)
    std::vector<float> w_gate; // [n_expert][n_embd] row-major per expert
    std::vector<float> w_norm; // [n_embd]
    std::vector<float> bias;   // [n_expert]
    // per-expert weight slices, 3 per expert (gate/up/down); only for host-resident layers
    std::vector<expert_slice> slices;
    bool host = false;
    std::vector<uint64_t> advised;      // dedup bitmap, (n_expert+63)/64 words — WORKER THREAD ONLY
};

} // namespace

// All prediction/madvise work runs on a dedicated worker thread; the eval callback and the
// decode-path token hook only post into the mailbox below (the inline v1 cost ~0.4ms/layer of
// hot-path matvec — measured -24% tg on Q8 — so nothing heavier than a memcpy belongs there).
struct llama_moe_prefetch {
    int   n_layer = 0;
    int   n_embd = 0;
    int   n_expert = 0;
    int   n_used = 0;
    int   n_hash = 0;
    float rms_eps = 1e-6f;
    int   k = 16;
    int   depth = 1;
    long  page = 4096;

    std::vector<layer_info> layers;
    std::vector<int32_t> tid2eid; // [n_hash][n_vocab][n_used] flattened
    int64_t n_vocab = 0;

    // mailbox (mutex-guarded): latest hidden state wins; token ids accumulate
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    have_state = false;
    int                     state_layer = -1;
    std::vector<float>      state_x;
    std::vector<int32_t>    pending_tokens;
    std::atomic<bool>       stop_flag{false};
    std::thread             worker;

    // worker-thread-only scratch + stats
    std::vector<float> scr_xn, scr_score;
    std::vector<int>   scr_idx;
    uint64_t n_tok = 0, n_pred = 0, n_advise = 0, n_dropped = 0;

    ~llama_moe_prefetch() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop_flag = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
        LLAMA_LOG_INFO("moe-prefetch: %llu tokens seen, %llu predictions (%llu stale-dropped), %llu madvise calls\n",
                (unsigned long long) n_tok, (unsigned long long) n_pred,
                (unsigned long long) n_dropped, (unsigned long long) n_advise);
    }

    // ---- worker-thread side ----

    void advise(layer_info & li, int e) {
        // the bitmap is sized from n_expert at create; an id outside it would write past it
        if (e < 0 || e >= n_expert || (size_t) (e >> 6) >= li.advised.size()) return;
        if (li.advised[e >> 6] & (1ull << (e & 63))) return;
        li.advised[e >> 6] |= 1ull << (e & 63);
#ifdef __linux__
        for (int j = 0; j < 3; j++) {
            const expert_slice & s = li.slices[3*(size_t)e + j];
            uintptr_t a = (uintptr_t) s.addr;
            uintptr_t al = a & ~((uintptr_t) page - 1);
            posix_madvise((void *) al, s.len + (a - al), POSIX_MADV_WILLNEED);
            n_advise++;
        }
#else
        (void) li; (void) e;
#endif
    }

    void clear_advised() {
        for (auto & li : layers) std::fill(li.advised.begin(), li.advised.end(), 0);
    }

    // predict layer T's top-k experts from state x (pre-norm, [n_embd]) and advise
    void predict_and_advise(int T, const float * x) {
        layer_info & li = layers[T];
        if (!li.host || li.w_gate.empty()) return;
        double ss = 0.0;
        for (int d = 0; d < n_embd; d++) ss += (double) x[d] * x[d];
        const float inv = 1.0f / sqrtf((float) (ss / n_embd) + rms_eps);

        scr_xn.resize(n_embd);
        for (int d = 0; d < n_embd; d++) scr_xn[d] = x[d] * inv * li.w_norm[d];

        scr_score.resize(n_expert);
        for (int e = 0; e < n_expert; e++) {
            const float * w = li.w_gate.data() + (size_t) e * n_embd;
            float acc = 0.0f;
            for (int d = 0; d < n_embd; d++) acc += w[d] * scr_xn[d];
            const float sp = acc > 20.0f ? acc : log1pf(expf(acc));
            scr_score[e] = sqrtf(sp) + li.bias[e];
        }
        scr_idx.resize(n_expert);
        std::iota(scr_idx.begin(), scr_idx.end(), 0);
        std::nth_element(scr_idx.begin(), scr_idx.begin() + k, scr_idx.end(),
                [&](int a, int b) { return scr_score[a] > scr_score[b]; });
        for (int i = 0; i < k; i++) advise(li, scr_idx[i]);
        n_pred++;
    }

    void process_tokens(const std::vector<int32_t> & toks) {
        const uint64_t before = n_tok;
        n_tok += toks.size();
        if ((n_tok >> 6) != (before >> 6)) clear_advised(); // dedup window: re-advise as heat drifts
        if (tid2eid.empty()) return;
        for (int il = 0; il < n_hash; il++) {
            layer_info & li = layers[il];
            if (!li.host) continue;
            const int32_t * tab = tid2eid.data() + (size_t) il * n_vocab * n_used;
            for (int32_t id : toks) {
                if (id < 0 || id >= n_vocab) continue;
                const int32_t * ex = tab + (size_t) id * n_used;
                for (int j = 0; j < n_used; j++) {
                    if (ex[j] >= 0 && ex[j] < n_expert) advise(li, ex[j]);
                }
            }
        }
    }

    void worker_loop() {
        std::vector<float>   x;
        std::vector<int32_t> toks;
        for (;;) {
            int il = -1;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return stop_flag.load() || have_state || !pending_tokens.empty(); });
                if (stop_flag) return;
                if (have_state) {
                    il = state_layer;
                    x.swap(state_x);
                    have_state = false;
                }
                toks.clear();
                toks.swap(pending_tokens);
            }
            if (!toks.empty()) process_tokens(toks);
            if (il >= 0) {
                for (int d = 1; d <= depth; d++) {
                    const int T = il + d;
                    if (T >= n_hash && T < n_layer) predict_and_advise(T, x.data());
                }
            }
        }
    }

    // ---- hot-path side (decode thread / sched callback) ----

    void post_state(int il, const float * x) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (have_state) n_dropped++; // stale prediction superseded — advisory, fine
            state_x.assign(x, x + n_embd);
            state_layer = il;
            have_state = true;
        }
        cv.notify_one();
    }

    void post_tokens(const int32_t * tokens, int32_t n) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            pending_tokens.insert(pending_tokens.end(), tokens, tokens + n);
            if (pending_tokens.size() > 8192) pending_tokens.clear(); // runaway guard
        }
        cv.notify_one();
    }
};

void llama_moe_prefetch_deleter::operator()(llama_moe_prefetch * p) const { delete p; }

llama_moe_prefetch_ptr llama_moe_prefetch_create(const llama_model & model) {
    const char * env = getenv("LLAMA_MOE_PREFETCH");
    if (!env || !*env) return nullptr;

    if (model.arch != LLM_ARCH_DEEPSEEK4) {
        LLAMA_LOG_WARN("moe-prefetch: arch not supported (deepseek4 only for now) — disabled\n");
        return nullptr;
    }

    auto p = llama_moe_prefetch_ptr(new llama_moe_prefetch());
    {
        int k = 16, depth = 1;
        if (sscanf(env, "%d,%d", &k, &depth) < 1) k = 16;
        p->k     = std::max(0, std::min(k, 128)); // 0 = observe-only diagnostic
        p->depth = std::max(1, std::min(depth, 2));
    }

    const auto & hp = model.hparams;
    // probe/vocab-only contexts (common_fit_params) have hparams but no materialized layers
    if (model.layers.size() < hp.n_layer()) {
        return nullptr;
    }
    p->n_layer  = (int) hp.n_layer();
    p->n_embd   = (int) hp.n_embd;
    p->n_expert = (int) hp.n_expert;
    p->n_used   = (int) hp.n_expert_used(); // upstream #25444: per-layer accessor; deepseek4 graph uses layer 0 too
    p->n_hash   = (int) hp.dsv4_hash_layer_count;
    // nth_element with k past the end is undefined; a tiny expert bank caps the lookahead
    p->k        = std::min(p->k, p->n_expert);
    p->rms_eps  = hp.f_norm_rms_eps;
#ifdef __linux__
    p->page = sysconf(_SC_PAGESIZE);
#endif

    p->layers.resize(p->n_layer);
    int n_host = 0, n_router = 0;

    for (int il = 0; il < p->n_layer; il++) {
        const auto & L  = model.layers[il];
        layer_info & li = p->layers[il];

        // expert slice map — host-resident layers only (GPU layers need no NVMe prefetch)
        const ggml_tensor * exps[3] = { L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps };
        bool host = true;
        for (auto * t : exps) {
            if (!t || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !t->data) { host = false; break; }
        }
        li.advised.assign(((size_t) p->n_expert + 63) / 64, 0);
        if (host) {
            li.slices.resize((size_t) 3 * p->n_expert);
            for (int j = 0; j < 3; j++) {
                for (int e = 0; e < p->n_expert; e++) {
                    li.slices[3*(size_t)e + j] = { (char *) exps[j]->data + (size_t) e * exps[j]->nb[2],
                                                   (size_t) exps[j]->nb[2] };
                }
            }
            li.host = true;
            n_host++;
        }

        if (il >= p->n_hash) {
            if (!tensor_to_f32(L.ffn_gate_inp, li.w_gate) ||
                !tensor_to_f32(L.ffn_norm,     li.w_norm) ||
                !tensor_to_f32(L.ffn_exp_probs_b, li.bias)) {
                LLAMA_LOG_WARN("moe-prefetch: router snapshot failed at layer %d — disabled\n", il);
                return nullptr;
            }
            n_router++;
        }
    }

    // hash-layer tables (I32 [n_used, n_vocab] each)
    if (p->n_hash > 0) {
        const ggml_tensor * t0 = model.layers[0].ffn_gate_tid2eid;
        if (t0) {
            p->n_vocab = t0->ne[1];
            p->tid2eid.resize((size_t) p->n_hash * p->n_vocab * p->n_used);
            for (int il = 0; il < p->n_hash; il++) {
                const ggml_tensor * t = model.layers[il].ffn_gate_tid2eid;
                if (!t || !t->buffer || !t->data || t->type != GGML_TYPE_I32 || t->ne[1] != p->n_vocab) { p->tid2eid.clear(); break; }
                ggml_backend_tensor_get(const_cast<ggml_tensor *>(t),
                        p->tid2eid.data() + (size_t) il * p->n_vocab * p->n_used,
                        0, (size_t) p->n_vocab * p->n_used * sizeof(int32_t));
            }
        }
    }

    if (n_host == 0) {
        LLAMA_LOG_INFO("moe-prefetch: no host-resident expert layers — nothing to prefetch, disabled\n");
        return nullptr;
    }

    p->worker = std::thread([raw = p.get()] { raw->worker_loop(); });

    LLAMA_LOG_INFO("moe-prefetch: ENABLED (async worker) k=%d depth=%d — %d/%d layers host-resident, %d routed, %d hash (tid2eid %s)\n",
            p->k, p->depth, n_host, p->n_layer, n_router, p->n_hash,
            p->tid2eid.empty() ? "unavailable" : "loaded");
    return p;
}

void llama_moe_prefetch_on_tokens(llama_moe_prefetch * p, const int32_t * tokens, int32_t n_tokens) {
    if (!p || !tokens || n_tokens <= 0) return;
    p->post_tokens(tokens, n_tokens);
}

bool llama_moe_prefetch_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * p = (llama_moe_prefetch *) user_data;

    // per-layer lookahead from the FFN-input state (generation-shaped batches only)
    if (strncmp(t->name, "hc_ffn_pre-", 11) == 0) {
        char * end = nullptr;
        const long il = strtol(t->name + 11, &end, 10);
        if (!end || *end != '\0' || il < 0 || il >= p->n_layer) return !ask;
        if (t->ne[1] > 4) return !ask; // pp batches: different regime, out of phase-A scope
        if (ask) return true;

        if (p->k == 0) return true; // observe-only diagnostic mode: pay the sched-split cost, do nothing
        if (!ggml_is_contiguous(t) || t->ne[0] != p->n_embd || t->type != GGML_TYPE_F32) return true;
        const int64_t nt = t->ne[1];
        float x[8192];
        if (p->n_embd > (int) (sizeof(x)/sizeof(x[0]))) return true;
        ggml_backend_tensor_get(t, x, (size_t) (nt - 1) * p->n_embd * sizeof(float),
                (size_t) p->n_embd * sizeof(float));
        p->post_state((int) il, x);
        return true;
    }

    return !ask;
}
