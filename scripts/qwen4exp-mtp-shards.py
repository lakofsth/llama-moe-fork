#!/usr/bin/env python3
"""qwen4exp-mtp-shards.py — serve the deployed target WITH its NextN block, without copying the target.

docs/qwen4exp-mtp.md §5 packaging (b), "same model": the draft context runs blk.{n_layer} of the target model, so
the embedding and head are shared rather than duplicated (~1.2 GB of VRAM). The deployed target is a 3-shard GGUF
whose shard 1 carries metadata only (0 tensors) and whose shards 2-3 carry every tensor. So a 4-shard set is:

  -00001-of-00004.gguf   NEW: shard 1's metadata with block_count n_layer+1, nextn_predict_layers 1, the MTP
                         GGUF's compress_ratios (one more entry), split.count 4, split.tensors.count + the block
  -00002-of-00004.gguf   SYMLINK to the target's shard 2 (llama-model-loader checks only split.no per shard)
  -00003-of-00004.gguf   SYMLINK to the target's shard 3
  -00004-of-00004.gguf   NEW: the MTP GGUF's blk.{n_layer}.* tensors, split.no 3

The symlinks keep ONE copy of the 94 GB of weights on disk and in the page cache. A context that does not ask
for the drafter (no --spec-type draft-mtp) skips every blk.{n_layer} tensor (load_arch_tensors, TENSOR_SKIP), but
its -ts split counts n_layer_all + 1 slots, so a layer split must be re-stated for the extra layer.

usage: qwen4exp-mtp-shards.py TARGET_SHARD1 MTP_GGUF OUT_DIR
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402

KV_SPLIT_NO, KV_SPLIT_COUNT, KV_SPLIT_TENSORS = "split.no", "split.count", "split.tensors.count"


def field_value(f):
    t = f.types[0]
    if t == gguf.GGUFValueType.ARRAY:
        sub = f.types[1]
        if sub == gguf.GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data], t, sub
        return [f.parts[i].tolist()[0] for i in f.data], t, sub
    if t == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8"), t, None
    return f.parts[f.data[0]].tolist()[0], t, None


def add_field(w, key, value, t, sub):
    if t == gguf.GGUFValueType.ARRAY:
        w.add_key_value(key, value, t, sub_type=sub)
    else:
        w.add_key_value(key, value, t)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    tgt1, mtp_path, out = sys.argv[1:]
    base = os.path.basename(tgt1)
    if "-00001-of-" not in base:
        sys.exit(f"{base}: not a first shard")
    stem, rest = base.split("-00001-of-")
    n_old = int(rest.split(".")[0])
    if n_old != 3:
        sys.exit(f"expected a 3-shard target, got {n_old}")

    t1 = gguf.GGUFReader(tgt1)
    mtp = gguf.GGUFReader(mtp_path)
    arch = field_value(t1.fields["general.architecture"])[0]
    if arch != "qwen4exp" or field_value(mtp.fields["general.architecture"])[0] != arch:
        sys.exit("both files must be qwen4exp")
    if len(t1.tensors) != 0:
        sys.exit(f"{base}: shard 1 carries {len(t1.tensors)} tensors; this script assumes metadata only")
    n_layer = field_value(t1.fields[f"{arch}.block_count"])[0]
    mtp_blocks = field_value(mtp.fields[f"{arch}.block_count"])[0]
    if mtp_blocks != n_layer + 1:
        sys.exit(f"MTP GGUF block_count {mtp_blocks}, target {n_layer}: expected target + 1")
    prefix = f"blk.{n_layer}."
    blk = [t for t in mtp.tensors if t.name.startswith(prefix)]
    if not blk:
        sys.exit(f"MTP GGUF has no {prefix}* tensors")
    n_tensors_old = field_value(t1.fields[KV_SPLIT_TENSORS])[0]

    os.makedirs(out, exist_ok=True)
    name = lambda i: os.path.join(out, f"{stem}-{i:05d}-of-00004.gguf")

    # shard 1: every KV of the target's shard 1, with the four overrides
    w = gguf.GGUFWriter(name(1), arch=arch, use_temp_file=False)
    over = {
        f"{arch}.block_count": n_layer + 1,
        f"{arch}.nextn_predict_layers": 1,
        KV_SPLIT_COUNT: 4,
        KV_SPLIT_TENSORS: n_tensors_old + len(blk),
    }
    cr_key = f"{arch}.attention.compress_ratios"
    for key, f in t1.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        v, t, sub = field_value(f)
        if key == cr_key:
            v, t, sub = field_value(mtp.fields[cr_key])
            if len(v) != n_layer + 1 or v[:n_layer] != field_value(f)[0]:
                sys.exit("MTP compress_ratios are not the target's plus one entry")
        if key in over:
            v = over.pop(key)
        add_field(w, key, v, t, sub)
    for key, v in over.items():   # keys the target's shard 1 did not carry
        w.add_uint32(key, v) if key != KV_SPLIT_COUNT else w.add_uint16(key, v)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    # shards 2-3: symlinks
    for i in (2, 3):
        src = os.path.join(os.path.dirname(os.path.abspath(tgt1)), f"{stem}-{i:05d}-of-00003.gguf")
        if not os.path.exists(src):
            sys.exit(f"missing {src}")
        if os.path.lexists(name(i)):
            os.remove(name(i))
        os.symlink(src, name(i))

    # shard 4: the block's tensors as stored (types and bytes unchanged)
    w = gguf.GGUFWriter(name(4), arch=arch, use_temp_file=False)
    w.add_uint16(KV_SPLIT_NO, 3)
    w.add_uint16(KV_SPLIT_COUNT, 4)
    w.add_int32(KV_SPLIT_TENSORS, n_tensors_old + len(blk))
    for t in blk:
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: shard 1 (block_count {n_layer + 1}), shards 2-3 -> target, shard 4 ({len(blk)} tensors)")


if __name__ == "__main__":
    main()
