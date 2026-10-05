from __future__ import annotations

from typing import Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import ModelBase, TextModel
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # The MTP block (mtp.*, one layer) is a separate draft head: a plain conversion leaves it out
    # (vLLM's main model drops it too) and `--mtp` exports it alone, as blk.{n_layer} NextN tensors
    # next to token_embd/output, for common/speculative.cpp's draft-mtp. See docs/qwen4exp-mtp.md.
    supports_mtp_export = True
    no_mtp = True

    # mtp.* tensors outside mtp.layers.N, renamed onto the NextN block (index n_layer) so the
    # shared remap in _QwenMtpMixin.filter_tensors and the block tensor map see one layer.
    # fc_embedding/fc_hidden are joined into nextn.eh_proj by modify_tensors.
    _MTP_TOP_LEVEL = {
        "mtp.fc_embedding.weight":                          "fc_embedding.weight",
        "mtp.fc_hidden.weight":                             "fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight":                 "enorm.weight",
        "mtp.pre_fc_norm_hidden.weight":                    "hnorm.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight":        "nextn_hc_head.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": "nextn_hc_head.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight":   "nextn_hc_head.input_mix_weight_up.weight",
    }

    def __init__(self, *args, **kwargs):
        # --mtp sets mtp_only on the class; the block must then be kept, not dropped by the default
        if type(self).mtp_only:
            type(self).no_mtp = False
        super().__init__(*args, **kwargs)
        self._mtp_fc: dict[str, Tensor] = {}
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    # what --mtp still reads from the trunk: the PLE hash constants (set_gguf_parameters writes them
    # as metadata) and the PLE table shards (only their row width, for embedding_length_per_layer_input)
    _PLE_CONSTANTS = ("ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_offsets",
                      "ple_embedding.ngram_heads_vocab_sizes")

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if cls.mtp_only and (name.endswith(cls._PLE_CONSTANTS) or ".ngram_embedding.shard_" in name):
            # the draft's hparams must be the target's, PLE keys included (load_arch_hparams checks them)
            return TextModel.filter_tensors((name, gen))
        if name.startswith("model.mtp."):
            name = name.replace("model.", "", 1)
        if name in cls._MTP_TOP_LEVEL:
            if cls.no_mtp:
                return None
            assert cls._original_block_count is not None
            name = f"model.layers.{cls._original_block_count}.{cls._MTP_TOP_LEVEL[name]}"
            # past the mixin, whose --mtp filter keeps only mtp.* and the shared embedding/head
            return TextModel.filter_tensors((name, gen))
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        # the MTP layer (index n_layer, present only with --mtp) is a full-attention layer with an
        # indexer of its own, so it carries the same ratio
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if i >= n_layer or layer_types[i] == "full_attention" else 0 for i in range(self.block_count)]
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # MTP input projections: e = fc_embedding(enorm(embed)), h_s = fc_hidden(hnorm(R)_s) per stream s.
        # One [n_embd, 2*n_embd] matrix over concat(e_in, h_in_s) gives e + h_s exactly (it is linear),
        # which is the NextN eh_proj layout (columns: embedding half first, then hidden), so the existing
        # nextn.eh_proj detection in common/speculative.cpp finds this head.
        if name.endswith((".fc_embedding.weight", ".fc_hidden.weight")):
            assert bid is not None
            self._mtp_fc[name.rpartition(".")[0].rpartition(".")[2]] = data_torch
            if len(self._mtp_fc) < 2:
                return []
            e = self._mtp_fc.pop("fc_embedding")
            h = self._mtp_fc.pop("fc_hidden")
            if e.shape != h.shape or e.ndim != 2:
                raise ValueError(f"MTP fc_embedding {tuple(e.shape)} and fc_hidden {tuple(h.shape)} do not pair")
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, ".weight"),
                     torch.cat([e, h], dim=1))]

        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        if ".ngram_embedding.shard_" in name:
            if self.mtp_only:
                # the table is the trunk's; the MTP GGUF carries only its row width, as metadata
                self._ple_row_dim = int(data_torch.shape[-1])
                return []
            return self._place_ple_shard(data_torch, name)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._mtp_fc:
            raise ValueError(f"MTP input projection without its pair: {sorted(self._mtp_fc)}")
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
