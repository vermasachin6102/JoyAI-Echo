"""Load Gemma3's language-model backbone directly from the bf16 safetensors
checkpoint (the one `gemma_root` already requires for ModelLedger regardless
of whether the GGUF path is used), instead of a GGUF file.

Motivation: `gguf_loader.py`'s GGUF path is CPU-bound because
`gguf.dequantize()` does real numpy math -- unpacking Q4_0's 4-bit packed
bytes into float32. Measured: 226-471s depending on host (see AGENT_HANDOFF.md
/ NOTES.md). safetensors has no equivalent decode step: it's a raw-bytes
format (mmap'd, no compression), so `safe_open(...).get_tensor(name)` is
essentially "hand back a view into the file" -- there is nothing to compute.
Since the bf16 shards are already on disk either way (ModelLedger.
build_model_builders globs model*.safetensors unconditionally), this trades
"download an extra 8GB GGUF and CPU-decode it" for "read what's already
there, no decode."

Bonus, not just speed: this also improves accuracy. The GGUF path quantizes
bf16 -> Q4_0 (llama.cpp's conversion, lossy) -> bf16 (dequant) -> NF4 (ours,
lossy again) -- two lossy quantization passes stacked. This path is
bf16 -> NF4 directly, one lossy pass. Not bit-identical to the GGUF path by
construction (real output DOES differ, in the direction of MORE accurate,
not less) -- needs the quality gate before being trusted over the GGUF path
for anything that matters, not just a hash comparison.

Streams tensors one at a time, same memory discipline as gguf_loader.py's
iter_gemma_gguf_tensors -- do NOT reuse SingleGPUModelBuilder /
SafetensorsStateDictLoader.load() directly, that accumulates every tensor
into one dict before returning, which reintroduces the ~24GB peak-RAM
problem this design exists to avoid (checked its source directly, see
gguf_builder.py commit history / AGENT_HANDOFF.md for why streaming exists
at all).
"""

from __future__ import annotations

from collections.abc import Iterator

import safetensors
import torch

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX

LANGUAGE_MODEL_RAW_PREFIX = "language_model.model."
LANGUAGE_MODEL_PREFIX = "model.model.language_model."
EMBED_TOKENS_KEY = f"{LANGUAGE_MODEL_PREFIX}embed_tokens.weight"

_TARGET_VOCAB_SIZE = GEMMA3_CONFIG_FOR_LTX.text_config.vocab_size


def _raw_key_to_final_key(name: str) -> str | None:
    """Only the language-model backbone -- vision_tower / multi_modal_projector
    / lm_head keys are skipped entirely (never read, never yielded), same
    scope as the GGUF path (see gguf_loader.py's module docstring for why
    that's safe: GemmaTextEncoder.encode() never touches those)."""
    if not name.startswith(LANGUAGE_MODEL_RAW_PREFIX):
        return None
    return LANGUAGE_MODEL_PREFIX + name[len(LANGUAGE_MODEL_RAW_PREFIX):]


def iter_gemma_safetensors_tensors(
    gemma_root: str, dtype: torch.dtype = torch.bfloat16
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read Gemma3's bf16 safetensors shards under `gemma_root`, yielding one
    (final_key, tensor) pair at a time -- mirrors iter_gemma_gguf_tensors'
    contract exactly so both can feed the same _assign_tensor unchanged."""
    from ltx_core.utils import find_matching_file

    first_shard = find_matching_file(gemma_root, "model*.safetensors")
    shard_paths = sorted(first_shard.parent.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No *.safetensors shards found under {gemma_root}")

    yielded_any_embed = False
    for shard_path in shard_paths:
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for raw_name in f.keys():
                final_key = _raw_key_to_final_key(raw_name)
                if final_key is None:
                    continue

                tensor = f.get_tensor(raw_name).to(dtype=dtype)

                if final_key == EMBED_TOKENS_KEY:
                    yielded_any_embed = True
                    # Defensive, not assumed: the GGUF conversion excludes 64
                    # multimodal special-token rows (see gguf_loader.py), but
                    # the ORIGINAL HF checkpoint this is read from very likely
                    # already has the full vocab_size rows since that's what
                    # the model was actually trained/saved with. Check the
                    # real shape rather than assume either way.
                    if tensor.shape[0] < _TARGET_VOCAB_SIZE:
                        pad_rows = _TARGET_VOCAB_SIZE - tensor.shape[0]
                        padding = torch.zeros((pad_rows, tensor.shape[1]), dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding], dim=0)
                    elif tensor.shape[0] > _TARGET_VOCAB_SIZE:
                        raise ValueError(
                            f"embed_tokens has {tensor.shape[0]} rows, more than "
                            f"config vocab_size {_TARGET_VOCAB_SIZE} -- unexpected, "
                            f"investigate before trusting this tensor silently."
                        )

                yield final_key, tensor

    if not yielded_any_embed:
        raise ValueError(
            f"No '{EMBED_TOKENS_KEY}' tensor found under {gemma_root} -- "
            f"raw key prefix assumption ('{LANGUAGE_MODEL_RAW_PREFIX}') may be "
            f"wrong for this checkpoint. Do not trust a build that hit this path."
        )
