#!/usr/bin/env python3
"""Build the qwen4exp MTP draft GGUF from the fetched mtp.* tensors and the deployed target GGUF.

The BF16 checkpoint (Qwen/Qwen3.8-Flash-Next) is ~360 GB; its MTP block is 31 tensors, 5.2 GB. Strata's
tools/mtp_fetch.py fetches only those (HTTP range reads, pinned revision, SHA256-checked). This script turns
them into the GGUF that `convert_hf_to_gguf.py --mtp` would write from the whole checkpoint, without it:

  * the tensor names and transforms come from the converter itself (conversion/qwen4exp.py,
    Qwen4ExpTextModel with mtp_only): this script feeds it the fetched tensors in place of the
    safetensors shards and writes what its filter_tensors/modify_tensors yield;
  * the metadata (hparams, tokenizer, chat template, PLE constants) is copied from the deployed target
    GGUF, so the draft's vocabulary and hparams are the target's byte for byte; only block_count,
    nextn_predict_layers and the per-layer compress ratios gain the MTP layer;
  * token_embd and output are copied from the target GGUF as they are (already quantized), as the
    converter's --mtp keeps the shared embedding and head.

Tensor types follow the target GGUF's type per role (the fused hyper-connection read, ggml_hc_mix, wants
Q8_0 down/up and F32 inject), with the routed experts at --experts (q4_0 default, or q8_0).

    python scripts/qwen4exp-mtp-gguf.py --fetched /data/models/qwen3.8-flash-next-mtp \\
        --config /data/models/qwen3.8-flash-next-mtp/hf/config.json \\
        --target /srv/models/qwen3.8-flash-next-gguf/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \\
        --out /data/models/qwen3.8-flash-next-mtp/mtp-qwen3.8-flash-next-Q8_0-expQ4_0.gguf

Runs on the CPU only; needs torch (the converter's tensor type) and numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import gguf  # noqa: E402
from conversion.qwen4exp import Qwen4ExpTextModel  # noqa: E402

logger = logging.getLogger("qwen4exp-mtp-gguf")

T = gguf.MODEL_TENSOR
Q = gguf.GGMLQuantizationType

# the target GGUF's type for each role (UD-IQ4_XS: dense Q8_0, router/inject/norms F32, indexer BF16)
DENSE_TYPES: dict[T, Q] = {
    T.NEXTN_EH_PROJ:       Q.Q8_0,
    T.NEXTN_ENORM:         Q.F32,
    T.NEXTN_HNORM:         Q.F32,
    T.NEXTN_HC_HEAD_NORM:  Q.F32,
    T.NEXTN_HC_HEAD_DOWN:  Q.Q8_0,
    T.NEXTN_HC_HEAD_UP:    Q.Q8_0,
    T.HC_ATTN_NORM:        Q.F32,
    T.HC_ATTN_DOWN:        Q.Q8_0,
    T.HC_ATTN_UP:          Q.Q8_0,
    T.HC_ATTN_INJECT:      Q.F32,
    T.HC_FFN_NORM:         Q.F32,
    T.HC_FFN_DOWN:         Q.Q8_0,
    T.HC_FFN_UP:           Q.Q8_0,
    T.HC_FFN_INJECT:       Q.F32,
    T.ATTN_Q:              Q.Q8_0,
    T.ATTN_K:              Q.Q8_0,
    T.ATTN_V:              Q.Q8_0,
    T.ATTN_OUT:            Q.Q8_0,
    T.ATTN_Q_NORM:         Q.F32,
    T.ATTN_K_NORM:         Q.F32,
    T.INDEXER_Q_PROJ:      Q.BF16,
    T.INDEXER_K_PROJ:      Q.BF16,
    T.INDEXER_Q_NORM:      Q.F32,
    T.INDEXER_K_NORM:      Q.F32,
    T.FFN_GATE_INP:        Q.F32,
    T.FFN_GATE_INP_SHEXP:  Q.F32,
    T.FFN_GATE_SHEXP:      Q.Q8_0,
    T.FFN_UP_SHEXP:        Q.Q8_0,
    T.FFN_DOWN_SHEXP:      Q.Q8_0,
}
EXPERT_ROLES = (T.FFN_GATE_EXP, T.FFN_UP_EXP, T.FFN_DOWN_EXP)
SHARED_FROM_TARGET = ("token_embd.weight", "output.weight")
BF16_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_fetched(fetched: Path, check_sha: bool) -> dict[str, dict]:
    """mtp-manifest.json rows by name, each checked against its file's size (and sha256)."""
    rows = json.loads((fetched / "mtp-manifest.json").read_text())
    out = {}
    for r in rows:
        path = fetched / r["file"]
        if path.stat().st_size != r["bytes"]:
            raise SystemExit(f"{path}: {path.stat().st_size} bytes, manifest says {r['bytes']}")
        if check_sha and sha256_of(path) != r["sha256"]:
            raise SystemExit(f"{path}: sha256 differs from the manifest")
        out[r["name"]] = dict(r, path=path)
    if len(out) != 31:
        logger.warning("%d MTP tensors in the manifest (the pinned checkpoint has 31)", len(out))
    return out


def make_model(config: Path, fetched_rows: dict[str, dict], out: Path) -> Qwen4ExpTextModel:
    """The converter's own model class, fed the fetched tensors instead of the safetensors shards."""

    class FetchedMtp(Qwen4ExpTextModel):
        model_arch = gguf.MODEL_ARCH.QWEN4EXP

        def index_tensors(self, remote_hf_model_id=None):
            # the bookkeeping _QwenMtpMixin.index_tensors does, then the fetched files as the parts
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            type(self)._original_block_count = hp["num_hidden_layers"]
            type(self).opt_num_mtp_layers = 0
            tensors = {}
            for name, r in fetched_rows.items():
                def gen(r=r):
                    raw = np.fromfile(r["path"], dtype=np.uint8)
                    return torch.from_numpy(raw).view(BF16_DTYPES[r["dtype"]]).reshape(r["shape"])
                if titem := self.filter_tensors((name, gen)):
                    tensors[titem[0]] = titem[1]
            return tensors

    FetchedMtp.mtp_only = True
    hparams = json.loads(config.read_text())
    return FetchedMtp(config.parent, gguf.LlamaFileType.MOSTLY_Q8_0, out, eager=True, hparams=hparams)


def converted_tensors(model: Qwen4ExpTextModel):
    """(gguf name, float32 numpy) pairs, as ModelBase.prepare_tensors would produce them."""
    for name, gen in model.model_tensors.items():
        data = gen().to(torch.float32)
        bid = next((int(p) for p in name.split(".") if p.isdecimal()), None)
        for new_name, t in model.modify_tensors(data, name, bid):
            yield name, new_name, t.contiguous().numpy()


def role_of(tmap: dict[str, T], new_name: str) -> T:
    base = new_name.removesuffix(".weight")
    if base not in tmap:
        raise SystemExit(f"{new_name}: not a qwen4exp tensor name")
    return tmap[base]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetched", type=Path, required=True, help="mtp_fetch.py --out directory (tensors/, mtp-manifest.json)")
    ap.add_argument("--config", type=Path, required=True, help="the checkpoint's config.json (same pinned revision)")
    ap.add_argument("--target", type=Path, required=True, help="the target GGUF (first split)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experts", choices=["q4_0", "q8_0"], default="q4_0")
    ap.add_argument("--no-sha", action="store_true", help="skip re-hashing the fetched tensors")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = load_fetched(a.fetched, check_sha=not a.no_sha)
    model = make_model(a.config, rows, a.out)
    n_layer = model.hparams["num_hidden_layers"]
    il = n_layer                                   # the NextN block's index
    if model.block_count != n_layer + 1:
        raise SystemExit(f"block_count {model.block_count}: expected one MTP layer after {n_layer}")

    expert_type = Q.Q4_0 if a.experts == "q4_0" else Q.Q8_0
    # tensor name (no suffix) -> role, for the NextN block
    tmap = {gguf.TENSOR_NAMES[t].format(bid=il): t for t in gguf.MODEL_TENSORS[gguf.MODEL_ARCH.QWEN4EXP]}

    # the target's splits: its metadata, its shared tensors, and its last layer's types as a reference
    target_paths = sorted(a.target.parent.glob(a.target.name.replace("00001-of", "*-of"))) or [a.target]
    readers = [gguf.GGUFReader(p) for p in target_paths]
    target_tensors = {t.name: t for r in readers for t in r.tensors}
    meta = readers[0]
    arch = meta.fields["general.architecture"].contents()
    if arch != "qwen4exp":
        raise SystemExit(f"target architecture is {arch!r}")

    writer = gguf.GGUFWriter(a.out, arch)

    # ---- metadata: the target's, plus the MTP layer
    old_ratios = meta.fields[f"{arch}.attention.compress_ratios"].contents()
    if len(old_ratios) != n_layer:
        raise SystemExit(f"target compress_ratios has {len(old_ratios)} entries, expected {n_layer}")
    # as Qwen4ExpTextModel.set_gguf_parameters writes them with the MTP layer present
    ratio = model.hparams["indexer_compress_ratio"]
    layer_types = model.hparams["layer_types"]
    new_ratios = [ratio if i >= n_layer or layer_types[i] == "full_attention" else 0 for i in range(model.block_count)]
    if new_ratios[:n_layer] != list(old_ratios):
        raise SystemExit("the converter's compress ratios differ from the target's for the trunk layers")
    overrides = {
        f"{arch}.block_count": (model.block_count, gguf.GGUFValueType.UINT32, None),
        f"{arch}.attention.compress_ratios": (new_ratios, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32),
    }
    # general.description is replaced below; quantize.imatrix.* describe the target's own quantization
    skip = ("GGUF.", "split.", "quantize.", "general.architecture", "general.file_type", "general.description")
    for field in meta.fields.values():
        if field.name.startswith(skip):
            continue
        if field.name in overrides:
            val, vt, st = overrides.pop(field.name)
            writer.add_key_value(field.name, val, vt, sub_type=st)
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
    if overrides:
        raise SystemExit(f"target metadata lacks {sorted(overrides)}")
    writer.add_nextn_predict_layers(model.block_count - n_layer)
    writer.add_string("general.description", "MTP (NextN) draft head of Qwen3.8 Flash Next, for draft-mtp; "
                      "metadata and token_embd/output from the target GGUF")

    # ---- the shared embedding and head, as the target has them
    for name in SHARED_FROM_TARGET:
        t = target_tensors[name]
        writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
        logger.info("%-40s target %s %s", name, t.tensor_type.name, [int(x) for x in t.shape])

    # ---- the MTP block, through the converter
    seen = set()
    for src, new_name, data in converted_tensors(model):
        role = role_of(tmap, new_name)
        if role in EXPERT_ROLES:
            qtype = expert_type
        elif role in DENSE_TYPES:
            qtype = DENSE_TYPES[role]
        else:
            raise SystemExit(f"{new_name} ({role.name}): no type chosen for this role")
        if role == T.FFN_GATE_INP_SHEXP:
            data = data.reshape(-1)               # [1, n_embd] -> [n_embd], as in the target
        # the trunk's last full-attention layer has the same roles: its shapes must agree
        ref = target_tensors.get(new_name.replace(f"blk.{il}.", f"blk.{n_layer - 1}."))
        if ref is not None and [int(x) for x in ref.shape] != list(reversed(data.shape)):
            raise SystemExit(f"{new_name}: shape {list(reversed(data.shape))} != target's "
                             f"{[int(x) for x in ref.shape]}")
        qdata = gguf.quants.quantize(data, qtype)
        writer.add_tensor(new_name, qdata, raw_dtype=qtype)
        seen.add(new_name)
        logger.info("%-40s <- %-60s %s %s", new_name, src, qtype.name, list(reversed(data.shape)))
    if model._mtp_fc:
        raise SystemExit(f"unpaired MTP input projection: {sorted(model._mtp_fc)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    logger.info("wrote %s: %d MTP tensors + %d shared, %.3f GB", a.out, len(seen), len(SHARED_FROM_TARGET),
                a.out.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
