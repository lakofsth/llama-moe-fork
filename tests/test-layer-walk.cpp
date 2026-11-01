// LAYER-WALK (tests/INVARIANTS.md): no code walks llama_layer (or any struct) by reinterpreting it as
// an array of ggml_tensor pointers.
//
// The fork's llama_layer embeds llama_moe_split, whose std::vectors, device pointer, int and bool are not
// tensors, so a walk that reads the struct as `sizeof(layer)/sizeof(ggml_tensor *)` pointers hands those
// bytes to code that dereferences them as tensors (llama_model_save_to_file segfaulted in
// gguf_find_tensor; llama_opt_init had the same walk).
//
// The sites are derived from the source, not listed: every .c/.cc/.cpp/.h/.hpp file under the directory
// given on the command line (src/) is read and refused if it carries either shape of such a walk:
//   (1) a cast to a pointer to ggml_tensor pointer — reinterpret_cast/static_cast<[const] [struct]
//       ggml_tensor * [const] *> or the C-style `([struct] ggml_tensor **) &x`;
//   (2) element-count arithmetic `sizeof(...) / sizeof([struct] ggml_tensor *)`.
// This is one side: a file carrying either shape is refused; a walk written in some other shape (memcpy
// into a pointer buffer, say) is not recognised. Before trusting a clean scan the test proves each pattern
// matches the shapes the two removed walks were written in, and that the scan reached both files they sat in.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const std::regex & re_cast() {
    static const std::regex re(
        R"((reinterpret_cast|static_cast)\s*<\s*(const\s+)?(struct\s+)?ggml_tensor\s*\*\s*(const\s*)?\*\s*>)"
        R"(|\(\s*(const\s+)?(struct\s+)?ggml_tensor\s*\*\s*(const\s*)?\*\s*\)\s*&)");
    return re;
}

static const std::regex & re_count() {
    static const std::regex re(R"(sizeof\s*\([^()]*\)\s*/\s*sizeof\s*\(\s*(const\s+)?(struct\s+)?ggml_tensor\s*\*\s*\))");
    return re;
}

static bool walk_shape(const std::string & line) {
    return std::regex_search(line, re_cast()) || std::regex_search(line, re_count());
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <path of src/>\n", argv[0]);
        return 2;
    }

    // the instrument sees a positive: the two walks this invariant removed, as they were written
    const char * positives[] = {
        "        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {",
        "            const struct ggml_tensor * tensor = reinterpret_cast<const struct ggml_tensor * const *>(&layer)[i];",
        "            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);",
        "    ggml_tensor ** p = (ggml_tensor **) &layer;",
    };
    for (const char * p : positives) {
        if (!walk_shape(p)) {
            fprintf(stderr, "FAIL: the scan does not recognise a known walk: %s\n", p);
            return 1;
        }
    }
    // and does not refuse an ordinary tensor-pointer array
    const char * negatives[] = {
        "    std::vector<ggml_tensor *> tensors;",
        "    struct ggml_tensor * t = ggml_get_first_tensor(ctx);",
        "    const size_t n = sizeof(buf)/sizeof(buf[0]);",
    };
    for (const char * n : negatives) {
        if (walk_shape(n)) {
            fprintf(stderr, "FAIL: the scan refuses a line that is no walk: %s\n", n);
            return 1;
        }
    }

    const fs::path root = argv[1];
    const std::set<std::string> exts = { ".c", ".cc", ".cpp", ".h", ".hpp" };
    std::set<std::string> seen;
    int n_files = 0;
    int n_bad   = 0;
    for (const auto & ent : fs::recursive_directory_iterator(root)) {
        if (!ent.is_regular_file() || !exts.count(ent.path().extension().string())) {
            continue;
        }
        n_files++;
        seen.insert(ent.path().filename().string());
        std::ifstream f(ent.path());
        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            lineno++;
            if (walk_shape(line)) {
                fprintf(stderr, "FAIL: %s:%d walks a struct as an array of ggml_tensor pointers: %s\n",
                        ent.path().string().c_str(), lineno, line.c_str());
                n_bad++;
            }
        }
    }

    for (const char * must : { "llama-model-saver.cpp", "llama-context.cpp", "llama-model.h" }) {
        if (!seen.count(must)) {
            fprintf(stderr, "FAIL: the scan did not reach %s under %s (%d files)\n", must, root.string().c_str(), n_files);
            return 1;
        }
    }
    if (n_bad) {
        fprintf(stderr, "FAIL: %d site(s) of a tensor-pointer struct walk under %s\n", n_bad, root.string().c_str());
        return 1;
    }
    printf("OK: %d files under %s, no struct walked as an array of ggml_tensor pointers\n", n_files, root.string().c_str());
    return 0;
}
