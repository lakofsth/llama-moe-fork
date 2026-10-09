// test-mlock-tensors — llama_mlock_select / llama_mlock_ranges (src/llama-mlock-tensors.h), the LLAMA_MLOCK_TENSORS
// lock of named host tensors (fork-kernel-fusion, invariant M1 in tests/INVARIANTS.md). Real mlock against a real file
// mapping: VmLck (/proc/self/status) must rise by exactly the matched ranges and return to its start on unlock.
#include "../src/llama-mlock-tensors.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

static int fails = 0;
static void expect(const char * what, long long got, long long want) {
    printf("%s %s: %lld (want %lld)\n", got == want ? "ok  " : "FAIL", what, got, want);
    fails += got != want;
}

static long long vmlck_kb() {
    FILE * f = fopen("/proc/self/status", "r");
    char line[256];
    long long kb = -1;
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmLck: %lld kB", &kb) == 1) {
            break;
        }
    }
    if (f) fclose(f);
    return kb;
}

int main() {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    struct rlimit rl;
    getrlimit(RLIMIT_MEMLOCK, &rl);
    if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < 4u << 20) {
        printf("SKIP: RLIMIT_MEMLOCK %llu < 4 MiB\n", (unsigned long long) rl.rlim_cur);
        return 0;
    }
    // a file-backed shared mapping, as the model loader's mmap is: 4 "tensors" of 512 KiB, unaligned starts
    char path[] = "/tmp/test-mlock-tensors-XXXXXX";
    int fd = mkstemp(path);
    const size_t T = 512 * 1024, N = 4 * T + 2 * page;
    if (fd < 0 || ftruncate(fd, (off_t) N) != 0) { printf("FAIL: temp file\n"); return 1; }
    unlink(path);
    char * base = (char *) mmap(nullptr, N, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { printf("FAIL: mmap\n"); return 1; }
    const size_t off = 100;   // tensor data need not start on a page
    std::vector<llama_mlock_candidate> c = {
        {"per_layer_token_embd.weight", base + off + 0 * T, T, true},
        {"blk.0.ffn_up_exps.weight",    base + off + 1 * T, T, true},
        {"token_embd.weight",           base + off + 2 * T, T, true},
        {"per_layer_token_embd.gpu",    base + off + 3 * T, T, false},   // matches, but not host: never locked
    };

    // 1. unset / empty spec: nothing selected
    int nt = -1;
    expect("unset: ranges", (long long) llama_mlock_select(c, llama_mlock_patterns(nullptr), page, &nt).size(), 0);
    expect("empty spec: ranges", (long long) llama_mlock_select(c, llama_mlock_patterns(",,"), page, &nt).size(), 0);

    // 2. one pattern: only the host tensor containing it, page-aligned around its bytes
    auto r1 = llama_mlock_select(c, llama_mlock_patterns("per_layer_token_embd"), page, &nt);
    expect("pattern: tensors matched", nt, 1);
    expect("pattern: ranges", (long long) r1.size(), 1);
    const uintptr_t want_b = (uintptr_t) (base + off) & ~(uintptr_t) (page - 1);
    const uintptr_t want_e = ((uintptr_t) (base + off + T) + page - 1) & ~(uintptr_t) (page - 1);
    expect("pattern: range start", (long long) (r1.empty() ? 0 : r1[0].first - want_b), 0);
    expect("pattern: range end", (long long) (r1.empty() ? 0 : r1[0].second - want_e), 0);

    // 3. two patterns, adjacent tensors: merged into one range covering both (token_embd is a substring of
    //    per_layer_token_embd too, so this also selects the first)
    auto r2 = llama_mlock_select(c, llama_mlock_patterns("token_embd,_exps"), page, &nt);
    expect("two patterns: tensors matched", nt, 3);
    expect("two patterns: merged ranges", (long long) r2.size(), 1);

    // 4. the act: VmLck rises by exactly the selected bytes, and returns on unlock
    const long long k0 = vmlck_kb();
    auto res = llama_mlock_ranges(r1);
    const long long k1 = vmlck_kb();
    expect("lock: failures", res.n_failed, 0);
    expect("lock: VmLck rise (kB)", k1 - k0, (long long) ((want_e - want_b) / 1024));
    expect("lock: locked bytes", (long long) res.locked_bytes, (long long) (want_e - want_b));
    llama_munlock_ranges(res.locked);
    expect("unlock: VmLck back", vmlck_kb() - k0, 0);

    // 5. a failure is reported with its errno: an unmapped range cannot be locked (ENOMEM)
    char * hole = (char *) mmap(nullptr, page, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    munmap(hole, page);   // now certainly unmapped
    std::vector<std::pair<uintptr_t, uintptr_t>> bad = {{(uintptr_t) hole, (uintptr_t) hole + page}};
    auto rb = llama_mlock_ranges(bad);
    expect("bad range: failures", rb.n_failed, 1);
    expect("bad range: errno is set", rb.first_errno != 0, 1);

    munmap(base, N);
    close(fd);
    printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
    return fails ? 1 : 0;
}
