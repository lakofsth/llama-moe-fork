#pragma once
// fork-kernel-fusion (invariant M1, tests/INVARIANTS.md): lock named host-resident tensors in RAM.
// LLAMA_MLOCK_TENSORS=<substr>[,<substr>...] names tensors by substring; their page-aligned byte ranges are mlock'ed
// after load, so an mmap'd tensor read at random on the host (per_layer_token_embd: 26.8 GiB of IQ4_NL rows, gathered
// per token by the main thread) can never be evicted from the page cache and faulted back from disk mid-prompt
// (diag fault-20261009T171017Z: 573,028 major faults, 99.99% in that tensor, the prefill slow mode).
// Pure selection (llama_mlock_select) is separate from the act (llama_mlock_ranges) so the test can check both.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

struct llama_mlock_candidate {
    const char * name;
    const void * data;
    size_t       nbytes;
    bool         host;   // the tensor's buffer is host memory
};

struct llama_mlock_result {
    std::vector<std::pair<uintptr_t, size_t>> locked;   // (start, length) of each range locked
    size_t locked_bytes = 0;
    size_t failed_bytes = 0;
    int    n_failed     = 0;
    int    first_errno  = 0;
    int    n_tensors    = 0;
};

// The comma-separated substrings of spec, empty entries dropped.
static inline std::vector<std::string> llama_mlock_patterns(const char * spec) {
    std::vector<std::string> out;
    if (spec == nullptr) {
        return out;
    }
    std::string cur;
    for (const char * p = spec; ; p++) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) {
                out.push_back(cur);
            }
            cur.clear();
            if (*p == '\0') {
                break;
            }
        } else {
            cur += *p;
        }
    }
    return out;
}

// Page-aligned, merged ranges of the host candidates whose name contains any pattern; n_tensors counts the matches.
static inline std::vector<std::pair<uintptr_t, uintptr_t>> llama_mlock_select(
        const std::vector<llama_mlock_candidate> & cands, const std::vector<std::string> & pats, size_t page,
        int * n_tensors) {
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    int n = 0;
    for (const auto & c : cands) {
        if (!c.host || c.data == nullptr || c.nbytes == 0) {
            continue;
        }
        bool hit = false;
        for (const auto & p : pats) {
            if (strstr(c.name, p.c_str()) != nullptr) {
                hit = true;
                break;
            }
        }
        if (!hit) {
            continue;
        }
        n++;
        const uintptr_t b = (uintptr_t) c.data & ~(uintptr_t) (page - 1);
        const uintptr_t e = ((uintptr_t) c.data + c.nbytes + page - 1) & ~(uintptr_t) (page - 1);
        ranges.emplace_back(b, e);
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (const auto & r : ranges) {
        if (!merged.empty() && r.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, r.second);
        } else {
            merged.push_back(r);
        }
    }
    if (n_tensors) {
        *n_tensors = n;
    }
    return merged;
}

// mlock each range; a failure is counted with its errno, never dropped silently.
// HOW IT IS POPULATED, measured (diag plelock-20261009T175816Z): a lazy tensor's range carries the loader's
// MADV_RANDOM, so mlock faulting it in from a part-cold page cache read 4 KiB at a time even after MADV_WILLNEED —
// 27,466 MiB in 101.3 s. So for the lock the range is advised SEQUENTIAL (fault-time readahead), split into chunks
// of at least 256 MiB locked by up to 16 threads (queue depth on the SSD), then advised RANDOM again, which is what
// the loader set; the locked pages no longer fault, so the advice only matters if the lock is ever released.
static inline llama_mlock_result llama_mlock_ranges(const std::vector<std::pair<uintptr_t, uintptr_t>> & ranges) {
    llama_mlock_result r;
    for (const auto & [b, e] : ranges) {
        const size_t len = e - b;
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        const size_t min_chunk = (size_t) 256 << 20;
        const size_t n_thr = std::max<size_t>(1, std::min<size_t>(16, len / min_chunk));
        const size_t chunk = ((len / n_thr) + page - 1) / page * page;
        madvise((void *) b, len, MADV_SEQUENTIAL);
        std::vector<int> err(n_thr, 0);
        std::vector<std::thread> th;
        for (size_t i = 0; i < n_thr; i++) {
            const uintptr_t cb = b + i * chunk;
            const uintptr_t ce = std::min<uintptr_t>(e, cb + chunk);
            if (cb >= ce) {
                continue;
            }
            th.emplace_back([cb, ce, i, &err] {
                if (mlock((const void *) cb, ce - cb) != 0) {
                    err[i] = errno;
                }
            });
        }
        for (auto & t : th) {
            t.join();
        }
        madvise((void *) b, len, MADV_RANDOM);
        int first = 0;
        for (int x : err) {
            if (x != 0 && first == 0) {
                first = x;
            }
        }
        if (first == 0) {
            r.locked.emplace_back(b, len);
            r.locked_bytes += len;
        } else {
            munlock((const void *) b, len);   // all or nothing per range: a half-locked range reads as neither
            if (r.n_failed == 0) {
                r.first_errno = first;
            }
            r.n_failed++;
            r.failed_bytes += len;
        }
    }
    return r;
}

static inline void llama_munlock_ranges(const std::vector<std::pair<uintptr_t, size_t>> & locked) {
    for (const auto & [b, n] : locked) {
        munlock((const void *) b, n);
    }
}
