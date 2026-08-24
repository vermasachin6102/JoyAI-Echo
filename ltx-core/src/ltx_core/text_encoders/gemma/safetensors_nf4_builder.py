"""Build a GemmaTextEncoder, NF4-quantized, sourced directly from the bf16
safetensors checkpoint instead of a GGUF file. See safetensors_loader.py for
the full rationale (no GGUF download, no CPU-bound dequant, more accurate
than the GGUF path's double-quantization).

Mirrors gguf_builder.py's build_gemma_text_encoder_from_gguf structure
closely on purpose -- same meta-model setup, same _assign_tensor (imported
unchanged, not reimplemented) for the actual NF4 quantization, same
leftover-meta safety check. Only the tensor SOURCE differs. No dequant-cache
wiring here: unlike GGUF decode, safetensors reads have no expensive CPU
step worth caching.
"""

from __future__ import annotations

import torch

from ltx_core.text_encoders.gemma.encoders.base_encoder import (
    GemmaTextEncoder,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    GEMMA_MODEL_OPS,
    GemmaTextEncoderConfigurator,
)
from ltx_core.text_encoders.gemma.gguf_builder import _assign_tensor
from ltx_core.text_encoders.gemma.safetensors_loader import iter_gemma_safetensors_tensors


def build_gemma_text_encoder_from_safetensors_nf4(
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaTextEncoder:
    """Construct an NF4-quantized GemmaTextEncoder reading weights straight
    from gemma_root's bf16 safetensors shards -- no GGUF file needed.

    Vision tower / multi-modal projector / lm_head are set to None, same
    scope and same justification as build_gemma_text_encoder_from_gguf.
    """
    with torch.device("meta"):
        text_encoder = GemmaTextEncoderConfigurator.from_config({})

    if GEMMA_MODEL_OPS.matcher(text_encoder):
        text_encoder = GEMMA_MODEL_OPS.mutator(text_encoder)

    inner = text_encoder.model
    inner.lm_head = None
    inner.model.vision_tower = None
    inner.model.multi_modal_projector = None

    tensor_count = 0
    for final_key, tensor in iter_gemma_safetensors_tensors(gemma_root, dtype=dtype):
        _assign_tensor(text_encoder, final_key, tensor, device=device, dtype=dtype)
        tensor_count += 1
    print(f"[GemmaSafetensorsNF4] loaded {tensor_count} tensors", flush=True)

    l_model = inner.model.language_model
    l_model.embed_tokens.to(device=device, dtype=dtype)
    l_model.rotary_emb_local.to(device=device, dtype=dtype)
    l_model.rotary_emb.to(device=device, dtype=dtype)

    leftover_meta = [name for name, p in text_encoder.named_parameters() if p.device.type == "meta"]
    if leftover_meta:
        raise RuntimeError(
            f"{len(leftover_meta)} parameters still on meta device after safetensors load "
            f"(first 5): {leftover_meta[:5]}"
        )

    for module_op in module_ops_from_gemma_root(gemma_root):
        if module_op.matcher(text_encoder):
            text_encoder = module_op.mutator(text_encoder)

    return text_encoder.eval()
