// test-qsa-select — llama_qsa_select_cells (src/llama-qsa-select.h), the cells a qwen4exp QSA query attends
// (fork-kernel-fusion, invariant S1 in tests/INVARIANTS.md).
//
// A cache of n_blocks*r cells holds positions 0..n_kv-1 in a scrambled order (cell = position*7 mod n_kv). Each query
// at position q has a tail [tail_start, q], tail_start = (q+1)/r*r; the whole blocks wholly before the tail are
// selectable (bias 0), everything else is -inf, exactly as set_input_qsa uploads it. The attended set is what the
// result unmasks after the attention mask (position <= q) and with the sentinel n_kv dropped; it must equal the top
// n_top selectable blocks whole plus the tail. Cases: distinct scores (the exact set), fewer selectable blocks than
// the budget (padding picks must not change the set), all-equal scores (whichever blocks win, they win whole), two
// streams with different tables, and randomized layouts. Runs on the CPU, and on CUDA when a GPU is visible.
#include "../src/llama-qsa-select.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

static int fails = 0;

struct scene {
    int64_t r, n_blocks, n_tps, n_stream, n_top;
    std::vector<int64_t> q;          // [n_stream*n_tps] query position
    std::vector<float>   blk_score;  // [n_stream*n_tps*n_blocks] raw block score
    int64_t mult[2] = {7, 7};        // per-stream cell = position*mult mod n_kv
};

static int64_t n_kv_of(const scene & s) { return s.n_blocks*s.r; }
static int64_t cell_of(const scene & s, int64_t st, int64_t pos) { return pos*s.mult[st] % n_kv_of(s); }

static std::vector<int32_t> run(ggml_backend_t be, const scene & s) {
    const int64_t r = s.r, nb = s.n_blocks, nt = s.n_tps, ns = s.n_stream, n_kv = n_kv_of(s);

    std::vector<float>   score(nb*nt*ns);
    std::vector<int32_t> blk_cells(r*nb*ns);
    std::vector<int32_t> tail((r - 1)*nt*ns, (int32_t) n_kv);

    for (int64_t st = 0; st < ns; ++st) {
        for (int64_t b = 0; b < nb; ++b) {
            for (int64_t m = 0; m < r; ++m) {
                blk_cells[st*r*nb + b*r + m] = (int32_t) cell_of(s, st, b*r + m);
            }
        }
        for (int64_t t = 0; t < nt; ++t) {
            const int64_t i = st*nt + t, q = s.q[i], ts = (q + 1)/r*r;
            for (int64_t b = 0; b < nb; ++b) {
                score[i*nb + b] = b*r < ts ? s.blk_score[i*nb + b] : -INFINITY;
            }
            for (int64_t p = ts, n = 0; p <= q; ++p, ++n) {
                tail[i*(r - 1) + n] = (int32_t) cell_of(s, st, p);
            }
        }
    }

    ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * t_score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nb, nt, ns);
    ggml_tensor * t_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*nb, ns);
    ggml_tensor * t_tail  = r > 1 ? ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r - 1, nt, ns) : nullptr;

    ggml_tensor * out = llama_qsa_select_cells(ctx, t_score, t_cells, t_tail, r, s.n_top);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(t_score, score.data(), 0, score.size()*sizeof(float));
    ggml_backend_tensor_set(t_cells, blk_cells.data(), 0, blk_cells.size()*sizeof(int32_t));
    if (t_tail) {
        ggml_backend_tensor_set(t_tail, tail.data(), 0, tail.size()*sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(1);
    }

    std::vector<int32_t> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(int32_t));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return res;
}

// what the query attends: unmasked (position <= q), the sentinel dropped
static std::set<int64_t> attended(const scene & s, const std::vector<int32_t> & res, int64_t st, int64_t t) {
    const int64_t r = s.r, w = r*s.n_top + r - 1, n_kv = n_kv_of(s), i = st*s.n_tps + t;
    std::vector<int64_t> pos_of(n_kv);
    for (int64_t p = 0; p < n_kv; ++p) {
        pos_of[cell_of(s, st, p)] = p;
    }
    std::set<int64_t> a;
    for (int64_t k = 0; k < w; ++k) {
        const int64_t c = res[i*w + k];
        if (c < 0 || c > n_kv) {
            printf("FAIL index %lld out of range\n", (long long) c);
            fails++;
            continue;
        }
        if (c < n_kv && pos_of[c] <= s.q[i]) {
            a.insert(pos_of[c]);
        }
    }
    return a;
}

// the positions S1 says it attends: top n_top selectable blocks by score (ties: any, checked by `whole`), plus the tail
static std::set<int64_t> expected(const scene & s, int64_t st, int64_t t) {
    const int64_t r = s.r, i = st*s.n_tps + t, q = s.q[i], ts = (q + 1)/r*r;
    std::vector<std::pair<float, int64_t>> sel;
    for (int64_t b = 0; b*r < ts; ++b) {
        sel.push_back({ -s.blk_score[i*s.n_blocks + b], b });
    }
    std::sort(sel.begin(), sel.end());
    std::set<int64_t> e;
    for (int64_t k = 0; k < std::min<int64_t>(s.n_top, sel.size()); ++k) {
        for (int64_t m = 0; m < r; ++m) {
            e.insert(sel[k].second*r + m);
        }
    }
    for (int64_t p = ts; p <= q; ++p) {
        e.insert(p);
    }
    return e;
}

// S1 with ties: every attended position outside the tail sits in a block attended whole, and the count is right
static bool whole(const scene & s, const std::set<int64_t> & a, int64_t st, int64_t t) {
    const int64_t r = s.r, i = st*s.n_tps + t, q = s.q[i], ts = (q + 1)/r*r;
    const int64_t n_sel = std::min<int64_t>(s.n_top, ts/r);
    int64_t n_blk = 0;
    for (int64_t p = ts; p <= q; ++p) {
        if (!a.count(p)) return false;
    }
    for (int64_t b = 0; b*r < ts; ++b) {
        int64_t n = 0;
        for (int64_t m = 0; m < r; ++m) n += a.count(b*r + m);
        if (n != 0 && n != r) return false;
        n_blk += n == r;
    }
    return n_blk == n_sel && (int64_t) a.size() == n_sel*r + (q - ts + 1);
}

static void check_exact(ggml_backend_t be, const char * what, const scene & s) {
    const auto res = run(be, s);
    const int before = fails;
    for (int64_t st = 0; st < s.n_stream; ++st) {
        for (int64_t t = 0; t < s.n_tps; ++t) {
            if (attended(s, res, st, t) != expected(s, st, t)) {
                printf("FAIL %s: stream %lld token %lld (q %lld)\n", what, (long long) st, (long long) t, (long long) s.q[st*s.n_tps + t]);
                fails++;
            }
        }
    }
    if (fails == before) printf("ok   %s\n", what);
}

int main() {
    ggml_backend_load_all();

    std::vector<std::pair<std::string, ggml_backend_t>> bes;
    bes.push_back({ "CPU", ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) });
    if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        bes.push_back({ ggml_backend_dev_name(gpu), ggml_backend_dev_init(gpu, nullptr) });
    }

    for (auto & [name, be] : bes) {
        printf("== backend %s\n", name.c_str());
        const int before = fails;

        // 1. the cut: tails of 1, 2, 3 and 0 cells. A cell-level top-k of width r*n_top + r - 1 fills the slots the
        //    tail leaves with part of the next block; S1 leaves them empty.
        {
            scene s { 4, 8, 4, 1, 3, { 24, 25, 26, 27 }, {} };
            for (int64_t t = 0; t < 4; ++t) {
                for (int64_t b = 0; b < 8; ++b) {
                    s.blk_score.push_back((float) ((b*5 + t*3) % 8) + 0.25f*b);
                }
            }
            check_exact(be, "cut: tails 1,2,3,0", s);
        }

        // 2. fewer selectable blocks than the budget: the padding picks are -inf blocks, which must change nothing
        {
            scene s { 4, 8, 2, 1, 4, { 9, 11 }, {} };
            for (int64_t t = 0; t < 2; ++t) {
                for (int64_t b = 0; b < 8; ++b) s.blk_score.push_back(1.0f + b);
            }
            check_exact(be, "budget above the selectable blocks", s);
        }

        // 3. all scores equal: which blocks win is the kernel's, but they win whole, 20 runs
        {
            scene s { 4, 64, 3, 1, 5, { 200, 201, 203 }, {} };
            s.blk_score.assign(64*3, 0.0f);
            bool ok = true;
            for (int rep = 0; rep < 20; ++rep) {
                const auto res = run(be, s);
                for (int64_t t = 0; t < 3; ++t) ok &= whole(s, attended(s, res, 0, t), 0, t);
            }
            printf("%s ties: whole blocks only over 20 runs\n", ok ? "ok  " : "FAIL");
            fails += !ok;
        }

        // 4. two streams, each gathering from its own block table
        {
            scene s { 4, 16, 2, 2, 3, { 40, 41, 30, 33 }, {} };
            s.mult[1] = 9;
            for (int64_t i = 0; i < 4; ++i) {
                for (int64_t b = 0; b < 16; ++b) s.blk_score.push_back((float) ((b*7 + i*5) % 16));
            }
            check_exact(be, "two streams", s);
        }

        // 5. randomized: distinct scores, budgets below and above the selectable count, ratio 2..8
        {
            std::mt19937 rng(42);
            for (int it = 0; it < 30; ++it) {
                scene s {};
                s.r        = 2 + rng() % 7;
                s.n_blocks = 8 + rng() % 120;
                s.n_tps    = 1 + rng() % 6;
                s.n_stream = 1 + rng() % 2;
                s.n_top    = 1 + rng() % s.n_blocks;
                s.mult[0]  = 1; s.mult[1] = 1;  // coprime with n_kv is not guaranteed for 7: use the identity layout
                for (int64_t i = 0; i < s.n_tps*s.n_stream; ++i) {
                    s.q.push_back(rng() % (s.n_blocks*s.r));
                    std::vector<float> v(s.n_blocks);
                    for (int64_t b = 0; b < s.n_blocks; ++b) v[b] = (float) b;
                    std::shuffle(v.begin(), v.end(), rng);
                    s.blk_score.insert(s.blk_score.end(), v.begin(), v.end());
                }
                check_exact(be, ("random " + std::to_string(it)).c_str(), s);
            }
        }

        // 6. GPU only: a tie at the cut on rows wide enough for the radix-select top-k (> 1024 columns, > 1 row).
        //    The lowest blocks among the tied must win, the same on every run (ggml-cuda top-k.cu, top_k_radix_ties);
        //    the CPU's partial_sort promises no tie order, so the CPU is not held to it.
        if (name != "CPU") {
            scene s { 4, 2048, 3, 1, 512, { 8190, 6001, 4100 }, {} };
            for (int64_t t = 0; t < 3; ++t) {
                for (int64_t b = 0; b < 2048; ++b) s.blk_score.push_back(b % 5 == 0 ? 1.0f : 0.0f);  // fewer than 512 at 1: the cut falls among the 0s
            }
            for (int rep = 0; rep < 5; ++rep) check_exact(be, ("radix tie at the cut, run " + std::to_string(rep)).c_str(), s);
        }

        printf("== backend %s: %s\n", name.c_str(), fails == before ? "PASS" : "FAIL");
    }

    for (auto & [name, be] : bes) ggml_backend_free(be);

    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
