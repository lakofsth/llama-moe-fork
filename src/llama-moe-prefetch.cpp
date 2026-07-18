#include "llama-moe-prefetch.h"

#include "llama-model.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

// dequantize a small router-side tensor (F32/F16/BF16) to f32
static bool tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) return false;
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
    uint64_t advised[4] = {0, 0, 0, 0}; // 256-bit dedup bitmap
};

} // namespace

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

    uint64_t n_tok = 0, n_pred = 0, n_advise = 0;

    ~llama_moe_prefetch() {
        LLAMA_LOG_INFO("moe-prefetch: %llu tokens seen, %llu predictions, %llu madvise calls\n",
                (unsigned long long) n_tok, (unsigned long long) n_pred, (unsigned long long) n_advise);
    }

    void advise(layer_info & li, int e) {
        if (li.advised[e >> 6] & (1ull << (e & 63))) return;
        li.advised[e >> 6] |= 1ull << (e & 63);
#ifdef __linux__
        for (int j = 0; j < 3; j++) {
            const expert_slice & s = li.slices[3*e + j];
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
        for (auto & li : layers) memset(li.advised, 0, sizeof(li.advised));
    }

    // predict layer T's top-k experts from state x (pre-norm, [n_embd]) and advise
    void predict_and_advise(int T, const float * x) {
        layer_info & li = layers[T];
        if (!li.host || li.w_gate.empty()) return;
        // rms-norm x through T's ffn_norm
        double ss = 0.0;
        for (int d = 0; d < n_embd; d++) ss += (double) x[d] * x[d];
        const float inv = 1.0f / sqrtf((float) (ss / n_embd) + rms_eps);

        std::vector<float> xn(n_embd);
        for (int d = 0; d < n_embd; d++) xn[d] = x[d] * inv * li.w_norm[d];

        std::vector<float> score(n_expert);
        for (int e = 0; e < n_expert; e++) {
            const float * w = li.w_gate.data() + (size_t) e * n_embd;
            float acc = 0.0f;
            for (int d = 0; d < n_embd; d++) acc += w[d] * xn[d];
            const float sp = acc > 20.0f ? acc : log1pf(expf(acc));
            score[e] = sqrtf(sp) + li.bias[e];
        }
        std::vector<int> idx(n_expert);
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                [&](int a, int b) { return score[a] > score[b]; });
        for (int i = 0; i < k; i++) advise(li, idx[i]);
        n_pred++;
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
        p->k     = std::max(1, std::min(k, 128));
        p->depth = std::max(1, std::min(depth, 2));
    }

    const auto & hp = model.hparams;
    p->n_layer  = (int) hp.n_layer();
    p->n_embd   = (int) hp.n_embd;
    p->n_expert = (int) hp.n_expert;
    p->n_used   = (int) hp.n_expert_used;
    p->n_hash   = (int) hp.dsv4_hash_layer_count;
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
                if (!t || t->type != GGML_TYPE_I32 || t->ne[1] != p->n_vocab) { p->tid2eid.clear(); break; }
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

    LLAMA_LOG_INFO("moe-prefetch: ENABLED k=%d depth=%d — %d/%d layers host-resident, %d routed, %d hash (tid2eid %s)\n",
            p->k, p->depth, n_host, p->n_layer, n_router, p->n_hash,
            p->tid2eid.empty() ? "unavailable" : "loaded");
    return p;
}

bool llama_moe_prefetch_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * p = (llama_moe_prefetch *) user_data;

    // token ids at batch entry -> hash-layer prefetch (any batch size)
    if (strcmp(t->name, "inp_tokens") == 0) {
        if (ask) return true;
        const int64_t nt = ggml_nelements(t);
        p->n_tok += nt;
        // periodic dedup reset so heat drift re-advises
        if ((p->n_tok >> 6) != ((p->n_tok - nt) >> 6)) p->clear_advised();
        if (!p->tid2eid.empty() && nt > 0 && nt <= 4096) {
            std::vector<int32_t> ids(nt);
            ggml_backend_tensor_get(t, ids.data(), 0, nt*sizeof(int32_t));
            for (int il = 0; il < p->n_hash; il++) {
                layer_info & li = p->layers[il];
                if (!li.host) continue;
                const int32_t * tab = p->tid2eid.data() + (size_t) il * p->n_vocab * p->n_used;
                for (int64_t i = 0; i < nt; i++) {
                    if (ids[i] < 0 || ids[i] >= p->n_vocab) continue;
                    const int32_t * ex = tab + (size_t) ids[i] * p->n_used;
                    for (int j = 0; j < p->n_used; j++) {
                        if (ex[j] >= 0 && ex[j] < p->n_expert) p->advise(li, ex[j]);
                    }
                }
            }
        }
        return true;
    }

    // per-layer lookahead from the FFN-input state (generation-shaped batches only)
    if (strncmp(t->name, "hc_ffn_pre-", 11) == 0) {
        char * end = nullptr;
        const long il = strtol(t->name + 11, &end, 10);
        if (!end || *end != '\0' || il < 0 || il >= p->n_layer) return !ask;
        if (t->ne[1] > 4) return !ask; // pp batches: different regime, out of phase-A scope
        if (ask) return true;

        if (!ggml_is_contiguous(t) || t->ne[0] != p->n_embd || t->type != GGML_TYPE_F32) return true;
        const int64_t nt = t->ne[1];
        std::vector<float> x(p->n_embd);
        ggml_backend_tensor_get(t, x.data(), (size_t) (nt - 1) * p->n_embd * sizeof(float),
                (size_t) p->n_embd * sizeof(float));
        for (int d = 1; d <= p->depth; d++) {
            const int T = (int) il + d;
            if (T >= p->n_hash && T < p->n_layer) p->predict_and_advise(T, x.data());
        }
        return true;
    }

    return !ask;
}
