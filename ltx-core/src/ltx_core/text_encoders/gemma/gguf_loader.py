"""Load a language-model-only Gemma3 GGUF checkpoint (e.g. Q4_0 quantized)
into a state dict keyed to match Gemma3ForConditionalGeneration, dequantizing
tensors on the fly.

Scope: only the language-model backbone (embed_tokens, transformer layers,
final norm) is loaded. Vision tower and lm_head are never populated here --
GemmaTextEncoder.encode() (the only method this codebase's inference path
calls) runs the inner Gemma3Model backbone directly with no pixel_values and
never calls generate(), so those parts are never touched at runtime. Confirmed
against real GGUF tensor data: 48 blocks x 13 tensors + 2 (embed/norm) = 626,
exactly matching the file's reported tensor count -- no vision tensors are
bundled into this specific file at all (they ship in a separate mmproj file
this loader does not use).

Mapping and shape-reversal rule below were verified against a real download
of google/gemma-3-12b-it-qat-q4_0-gguf, not inferred from documentation alone.

Tensors are yielded one at a time (not collected into a dict) -- the full
language-model backbone dequantized to bf16 is ~24GB, which alone can push a
Colab-class instance's system RAM over the edge before anything even reaches
the GPU. Streaming lets the caller quantize/move each tensor to GPU and drop
the CPU copy before dequantizing the next one, keeping peak CPU RAM to
roughly one tensor's size.
"""

from __future__ import annotations

from collections.abc import Iterator

import gguf
import torch

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX

LANGUAGE_MODEL_PREFIX = "model.model.language_model."
EMBED_TOKENS_KEY = f"{LANGUAGE_MODEL_PREFIX}embed_tokens.weight"

# GGUF's token_embd.weight has 262144 rows (pure text vocabulary) while our
# Gemma3TextConfig.vocab_size is 262208 -- the extra 64 rows are reserved for
# multimodal special tokens (config.image_token_index == 262144, i.e. right
# at this exact boundary). Verified against a real download: llama.cpp's
# conversion excludes these reserved rows since this is the language-model
# -only GGUF (no vision). Safe to zero-pad: GemmaTextEncoder.encode() only
# ever processes plain text token ids from the tokenizer, which never
# include image_token_index or other reserved ids beyond the base text
# vocabulary, so these padding rows are never looked up at runtime.
_TARGET_VOCAB_SIZE = GEMMA3_CONFIG_FOR_LTX.text_config.vocab_size

# GGUF tensor-name suffix (after "blk.N.") -> HF Gemma3 decoder-layer attribute path.
_PER_LAYER_MAP = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",  # named "ffn_norm" in GGUF but semantically pre-FFN
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "post_ffw_norm": "post_feedforward_layernorm",
}

_TOP_LEVEL_MAP = {
    "token_embd": "embed_tokens",
    "output_norm": "norm",
}

# 2D linear/embedding weights -- GGUF stores these with reversed shape vs
# PyTorch's (out_features, in_features) convention. Confirmed with real data:
# attn_q.weight gguf shape (3840, 4096) reversed -> (4096, 3840), matching
# Linear(in=3840, out=4096) (16 heads * 256 head_dim = 4096).
_2D_NAMES = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "token_embd"}


def _gguf_name_to_final_key(name: str) -> str | None:
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) != 3:
            return None
        _, block_idx, rest = parts
        rest = rest.removesuffix(".weight")
        mapped = _PER_LAYER_MAP.get(rest)
        if mapped is None:
            return None
        return f"{LANGUAGE_MODEL_PREFIX}layers.{block_idx}.{mapped}.weight"

    base = name.removesuffix(".weight")
    mapped = _TOP_LEVEL_MAP.get(base)
    if mapped is None:
        return None
    return f"{LANGUAGE_MODEL_PREFIX}{mapped}.weight"


def _tensor_base_name(gguf_name: str) -> str:
    if gguf_name.startswith("blk."):
        return gguf_name.split(".")[-2]
    return gguf_name.removesuffix(".weight")


def iter_gemma_gguf_tensors(gguf_path: str, dtype: torch.dtype = torch.bfloat16) -> Iterator[tuple[str, torch.Tensor]]:
    """Read a language-model-only Gemma3 GGUF file, dequantizing and yielding
    one (final_key, tensor) pair at a time -- see module docstring for scope
    and why this streams instead of returning a full state dict."""
    reader = gguf.GGUFReader(gguf_path)

    unmapped: list[str] = []

    for tensor in reader.tensors:
        final_key = _gguf_name_to_final_key(tensor.name)
        if final_key is None:
            unmapped.append(tensor.name)
            continue

        dequantized = gguf.dequantize(tensor.data, tensor.tensor_type)  # numpy float32

        if _tensor_base_name(tensor.name) in _2D_NAMES:
            true_shape = tuple(int(d) for d in reversed(tensor.shape))
        else:
            true_shape = tuple(int(d) for d in tensor.shape)
        dequantized = dequantized.reshape(true_shape)

        tensor_out = torch.from_numpy(dequantized.copy()).to(dtype=dtype)

        if final_key == EMBED_TOKENS_KEY and tensor_out.shape[0] < _TARGET_VOCAB_SIZE:
            pad_rows = _TARGET_VOCAB_SIZE - tensor_out.shape[0]
            padding = torch.zeros((pad_rows, tensor_out.shape[1]), dtype=tensor_out.dtype)
            tensor_out = torch.cat([tensor_out, padding], dim=0)

        yield final_key, tensor_out

    if unmapped:
        # Expected count here is zero -- this file only contains language
        # -model tensors (verified: 48*13+2 == reader tensor count). Any
        # name landing here means either an unexpected vision tensor or a
        # gap in the mapping table; surface it instead of silently
        # dropping weight data.
        raise ValueError(f"{len(unmapped)} GGUF tensors had no key mapping (first 5): {unmapped[:5]}")
