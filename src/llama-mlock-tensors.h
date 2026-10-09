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
static inline llama_mlock_result llama_mlock_ranges(const std::vector<std::pair<uintptr_t, uintptr_t>> & ranges) {
    llama_mlock_result r;
    for (const auto & [b, e] : ranges) {
        // Read the range in with the kernel's readahead first: a lazy tensor's range is advised MADV_RANDOM by the
        // loader, so mlock populating it from a cold page cache would fault it in one 4 KiB synchronous read at a
        // time (26.8 GiB = ~7M reads). WILLNEED queues large reads; mlock then finds the pages present.
        madvise((void *) b, e - b, MADV_WILLNEED);
        if (mlock((const void *) b, e - b) == 0) {
            r.locked.emplace_back(b, e - b);
            r.locked_bytes += e - b;
        } else {
            if (r.n_failed == 0) {
                r.first_errno = errno;
            }
            r.n_failed++;
            r.failed_bytes += e - b;
        }
    }
    return r;
}

static inline void llama_munlock_ranges(const std::vector<std::pair<uintptr_t, size_t>> & locked) {
    for (const auto & [b, n] : locked) {
        munlock((const void *) b, n);
    }
}
