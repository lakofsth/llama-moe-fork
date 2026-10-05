#!/usr/bin/env python3
"""Check a qwen4exp MTP GGUF (scripts/qwen4exp-mtp-gguf.py) against its sources, on the CPU.

  * metadata: the target's keys and values, except block_count (+1), nextn_predict_layers (1) and the
    compress ratios (one more entry); the tokenizer arrays are the target's, element for element;
  * every MTP tensor dequantizes back to its fetched BF16 source through the transform the converter
    applies (F32/BF16: exactly; Q8_0/Q4_0: with no more error than quantizing that expected tensor on its
    own in the same type, so a misplaced row, half or expert cannot hide under a loose bound): +1 on the
    zero-centred norm gammas, the indexer projection split, the gate/up split, eh_proj = [fc_embedding |
    fc_hidden];
  * token_embd/output are the target's bytes;
  * the gamma means of the MTP layer and the trunk's last layer are printed for reference only: the
    (1 + w) fold for the MTP block rests on Strata's tools/mtp_rt.py (it folds every MTP norm the same way
    and drafts at 0.89 acceptance), not on these statistics.

    python scripts/qwen4exp-mtp-gguf-check.py --gguf MTP.gguf --fetched DIR --target TARGET-00001-of-N.gguf

Exit 0 when everything holds, 1 with the failures listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

Q = gguf.GGMLQuantizationType
LOSSLESS = (Q.F32, Q.BF16)


def bf16_file(path: Path, shape) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--fetched", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    a = ap.parse_args()

    fails: list[str] = []
    mtp = gguf.GGUFReader(a.gguf)
    tpaths = sorted(a.target.parent.glob(a.target.name.replace("00001-of", "*-of"))) or [a.target]
    treaders = [gguf.GGUFReader(p) for p in tpaths]
    tmeta = treaders[0]
    ttensors = {t.name: t for r in treaders for t in r.tensors}
    arch = "qwen4exp"

    # ---- metadata
    n_layer = tmeta.fields[f"{arch}.block_count"].contents()
    want_changed = {f"{arch}.block_count", f"{arch}.attention.compress_ratios", f"{arch}.nextn_predict_layers",
                    "general.description"}
    for f in tmeta.fields.values():
        if f.name.startswith(("GGUF.", "split.", "quantize.", "general.file_type")) or f.name in want_changed:
            continue
        g = mtp.fields.get(f.name)
        if g is None:
            fails.append(f"metadata {f.name}: missing")
        elif g.types != f.types or g.contents() != f.contents():
            fails.append(f"metadata {f.name}: differs from the target")
    bc = mtp.fields[f"{arch}.block_count"].contents()
    nn = mtp.fields[f"{arch}.nextn_predict_layers"].contents()
    cr = mtp.fields[f"{arch}.attention.compress_ratios"].contents()
    tcr = tmeta.fields[f"{arch}.attention.compress_ratios"].contents()
    if bc != n_layer + 1 or nn != 1:
        fails.append(f"block_count {bc} / nextn_predict_layers {nn}: expected {n_layer + 1} / 1")
    if cr[:n_layer] != tcr or len(cr) != n_layer + 1 or cr[-1] == 0:
        fails.append(f"compress_ratios {cr[-3:]} (len {len(cr)}): expected the target's plus a nonzero MTP entry")
    print(f"metadata: {len(mtp.fields)} keys, block_count {bc}, nextn {nn}, compress ratio of blk.{n_layer} = {cr[-1]}")

    # ---- shared tensors: the target's bytes
    m = {t.name: t for t in mtp.tensors}
    for name in ("token_embd.weight", "output.weight"):
        if m[name].tensor_type != ttensors[name].tensor_type or not np.array_equal(m[name].data, ttensors[name].data):
            fails.append(f"{name}: not the target's bytes")
        else:
            print(f"{name}: the target's {m[name].tensor_type.name} bytes")

    # ---- MTP tensors against their sources
    rows = {r["name"]: r for r in json.loads((a.fetched / "mtp-manifest.json").read_text())}
    src = lambda n: bf16_file(a.fetched / rows[n]["file"], rows[n]["shape"])  # noqa: E731
    L = f"blk.{n_layer}."
    p = "mtp.layers.0."
    qk = src(p + "self_attn.indexer.index_qk_proj.weight")
    gu = src(p + "mlp.experts.gate_up_proj")
    n_ff = gu.shape[1] // 2
    expect = {
        L + "nextn.eh_proj.weight": np.concatenate([src("mtp.fc_embedding.weight"), src("mtp.fc_hidden.weight")], axis=1),
        L + "nextn.enorm.weight": src("mtp.pre_fc_norm_embedding.weight") + 1,
        L + "nextn.hnorm.weight": src("mtp.pre_fc_norm_hidden.weight") + 1,
        L + "nextn.hc_head_norm.weight": src("mtp.hyper_connection_mixer.hc_norm.weight") + 1,
        L + "nextn.hc_head_down.weight": src("mtp.hyper_connection_mixer.input_mix_weight_down.weight"),
        L + "nextn.hc_head_up.weight": src("mtp.hyper_connection_mixer.input_mix_weight_up.weight"),
        L + "attn_q.weight": src(p + "self_attn.q_proj.weight"),
        L + "attn_k.weight": src(p + "self_attn.k_proj.weight"),
        L + "attn_v.weight": src(p + "self_attn.v_proj.weight"),
        L + "attn_output.weight": src(p + "self_attn.o_proj.weight"),
        L + "attn_q_norm.weight": src(p + "self_attn.q_norm.weight") + 1,
        L + "attn_k_norm.weight": src(p + "self_attn.k_norm.weight") + 1,
        L + "indexer.q_proj.weight": qk[:512],
        L + "indexer.k_proj.weight": qk[512:],
        L + "indexer.q_norm.weight": src(p + "self_attn.indexer.q_layernorm.weight") + 1,
        L + "indexer.k_norm.weight": src(p + "self_attn.indexer.k_layernorm.weight") + 1,
        L + "ffn_gate_inp.weight": src(p + "mlp.gate.weight"),
        L + "ffn_gate_inp_shexp.weight": src(p + "mlp.shared_expert_gate.weight").reshape(-1),
        L + "ffn_gate_shexp.weight": src(p + "mlp.shared_expert.gate_proj.weight"),
        L + "ffn_up_shexp.weight": src(p + "mlp.shared_expert.up_proj.weight"),
        L + "ffn_down_shexp.weight": src(p + "mlp.shared_expert.down_proj.weight"),
        L + "ffn_gate_exps.weight": gu[:, :n_ff, :],
        L + "ffn_up_exps.weight": gu[:, n_ff:, :],
        L + "ffn_down_exps.weight": src(p + "mlp.experts.down_proj"),
    }
    for kind in ("attn", "ffn"):
        hf = "attn_hyper_connection" if kind == "attn" else "mlp_hyper_connection"
        expect[L + f"hc_{kind}_norm.weight"] = src(p + f"{hf}.hc_norm.weight") + 1
        expect[L + f"hc_{kind}_down.weight"] = src(p + f"{hf}.input_mix_weight_down.weight")
        expect[L + f"hc_{kind}_up.weight"] = src(p + f"{hf}.input_mix_weight_up.weight")
        expect[L + f"hc_{kind}_inject.weight"] = src(p + f"{hf}.block_inject_weight.weight")

    extra = sorted(set(m) - set(expect) - {"token_embd.weight", "output.weight"})
    missing = sorted(set(expect) - set(m))
    if extra or missing:
        fails.append(f"tensor set: extra {extra}, missing {missing}")
    for name, want in expect.items():
        if name not in m:
            continue
        t = m[name]
        rel = lambda x: float(np.sqrt(np.mean((x - want) ** 2)) / (np.sqrt(np.mean(want ** 2)) + 1e-30))  # noqa: E731
        err = rel(dequantize(t.data, t.tensor_type).reshape(want.shape))
        if t.tensor_type in LOSSLESS:
            floor = 0.0 if t.tensor_type == Q.F32 else rel(dequantize(gguf.quants.quantize(want, Q.BF16), Q.BF16).reshape(want.shape))
        else:
            floor = rel(dequantize(gguf.quants.quantize(want, t.tensor_type), t.tensor_type).reshape(want.shape))
        ok = err <= floor * 1.0001 + 1e-12
        if not ok:
            fails.append(f"{name}: relative RMS error {err:.4g} above its own {t.tensor_type.name} floor {floor:.4g}")
        print(f"{'ok  ' if ok else 'FAIL'} {name:36s} {t.tensor_type.name:5s} rel-rms {err:.2e} (floor {floor:.2e})")

    # ---- the folded-norm convention: the trunk's last layer and the MTP layer agree
    for role in ("hc_attn_norm", "hc_ffn_norm", "attn_q_norm", "attn_k_norm"):
        ref = ttensors[f"blk.{n_layer - 1}.{role}.weight"]
        rm = float(np.mean(dequantize(ref.data, ref.tensor_type)))
        mm = float(np.mean(dequantize(m[L + role + ".weight"].data, m[L + role + ".weight"].tensor_type)))
        print(f"mean {role} (folded, reference only): trunk blk.{n_layer - 1} {rm:.3f}, MTP {mm:.3f}")

    for f in fails:
        print("FAIL:", f)
    print("all checks hold" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
