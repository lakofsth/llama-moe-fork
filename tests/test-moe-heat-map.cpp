// fork: the heat-map length contract (posture L4) — exactly the full shape, or exactly the
// persisted short form, and nothing else. Each case writes a file of a chosen byte length.

#include "../src/llama-moe-heat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <string>
#include <vector>

static std::string write_floats(const std::string & dir, const char * name, size_t n_floats, size_t extra_bytes) {
    const std::string path = dir + "/" + name;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    for (size_t i = 0; i < n_floats; i++) {
        const float v = (float) (i + 1);
        fwrite(&v, sizeof(v), 1, f);
    }
    for (size_t i = 0; i < extra_bytes; i++) {
        fputc(0x7f, f);
    }
    fclose(f);
    return path;
}

static int n_fail = 0;

static void check(const char * what, bool got, bool want) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %s, want %s\n", what, got ? "accepted" : "refused", want ? "accepted" : "refused");
        n_fail++;
    } else {
        printf("ok: %s -> %s\n", what, got ? "accepted" : "refused");
    }
}

int main() {
    const int n_layer = 4, n_expert = 8, n_short = 3; // 3 main layers + 1 NextN layer
    const size_t n_full_floats  = (size_t) n_layer * n_expert;
    const size_t n_short_floats = (size_t) n_short * n_expert;

    char tmpl[] = "/tmp/test-moe-heat-map.XXXXXX";
    const char * dir = mkdtemp(tmpl);
    if (!dir) {
        fprintf(stderr, "mkdtemp failed\n");
        return 1;
    }
    const std::string d = dir;

    std::vector<float> heat;
    std::string err;
    bool padded = false;

    // exact full form
    check("full form, exact length",
          llama_moe_heat_read_map(write_floats(d, "full", n_full_floats, 0), n_layer, n_expert, n_short, heat, err, &padded), true);
    check("full form leaves padded unset", !padded, true);
    check("full form keeps the last value", heat.size() == n_full_floats && heat[n_full_floats - 1] == (float) n_full_floats, true);

    // persisted short form (NextN rows excluded)
    check("short form, exact length",
          llama_moe_heat_read_map(write_floats(d, "short", n_short_floats, 0), n_layer, n_expert, n_short, heat, err, &padded), true);
    check("short form sets padded", padded, true);
    check("short form zero-pads the NextN row", heat.size() == n_full_floats && heat[n_full_floats - 1] == 0.0f, true);

    // the defect: a longer file (another model's map under the same label) must be refused
    check("one float too long",
          llama_moe_heat_read_map(write_floats(d, "long1", n_full_floats + 1, 0), n_layer, n_expert, n_short, heat, err, &padded), false);
    check("one trailing byte",
          llama_moe_heat_read_map(write_floats(d, "trail", n_full_floats, 1), n_layer, n_expert, n_short, heat, err, &padded), false);
    check("a whole extra layer",
          llama_moe_heat_read_map(write_floats(d, "longL", n_full_floats + n_expert, 0), n_layer, n_expert, n_short, heat, err, &padded), false);

    // too short in every way
    check("one float short of full, not the short form",
          llama_moe_heat_read_map(write_floats(d, "short1", n_full_floats - 1, 0), n_layer, n_expert, n_short, heat, err, &padded), false);
    check("short form plus one float",
          llama_moe_heat_read_map(write_floats(d, "shortp1", n_short_floats + 1, 0), n_layer, n_expert, n_short, heat, err, &padded), false);
    check("empty file",
          llama_moe_heat_read_map(write_floats(d, "empty", 0, 0), n_layer, n_expert, n_short, heat, err, &padded), false);
    check("missing file",
          llama_moe_heat_read_map(d + "/missing", n_layer, n_expert, n_short, heat, err, &padded), false);

    // no NextN layers: the short form does not exist, only the full length is accepted
    check("no short form: full length accepted",
          llama_moe_heat_read_map(write_floats(d, "nf", n_full_floats, 0), n_layer, n_expert, n_layer, heat, err, &padded), true);
    check("no short form: n_short rows refused",
          llama_moe_heat_read_map(write_floats(d, "ns", n_short_floats, 0), n_layer, n_expert, n_layer, heat, err, &padded), false);

    // refused maps come back all-zero, never half-read
    llama_moe_heat_read_map(write_floats(d, "long2", n_full_floats + 1, 0), n_layer, n_expert, n_short, heat, err, &padded);
    bool all_zero = heat.size() == n_full_floats;
    for (float v : heat) all_zero = all_zero && v == 0.0f;
    check("a refused map is returned all-zero", all_zero, true);

    // cleanup
    const char * names[] = { "full", "short", "long1", "trail", "longL", "short1", "shortp1", "empty", "nf", "ns", "long2" };
    for (const char * n : names) remove((d + "/" + n).c_str());
    rmdir(dir);

    if (n_fail) {
        fprintf(stderr, "%d failures\n", n_fail);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
