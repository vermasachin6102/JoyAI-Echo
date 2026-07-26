"""Build a GemmaTextEncoder with weights sourced from a language-model-only
GGUF checkpoint (e.g. Q4_0 quantized), instead of the full bf16 safetensors
checkpoint. See gguf_loader.py for tensor-format details and scope.

Motivation: the full bf16 Gemma-3-12B checkpoint (~24GB, including vision
tower + lm_head we never use) doesn't fit on 24GB-class GPUs like L4 --
confirmed OOM at ~22GB used, failing on the last ~30MB. A Q4_0 GGUF of just
the language-model backbone is ~6.9GB, which does fit.
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
from ltx_core.text_encoders.gemma.gguf_loader import load_gemma_gguf_state_dict


def build_gemma_text_encoder_from_gguf(
    gguf_path: str,
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaTextEncoder:
    """Construct a GemmaTextEncoder with the Gemma LLM's weights sourced
    from a language-model-only GGUF file.

    Vision tower / multi-modal projector / lm_head are set to None -- never
    populated with weights, never present in the returned model. Confirmed
    safe because GemmaTextEncoder.encode() (the only method this codebase's
    inference path calls) runs Gemma3Model's inner forward with no
    pixel_values and never calls generate(); Gemma3Model.forward() only
    touches vision_tower/multi_modal_projector when pixel_values is not
    None (checked against the installed transformers source directly).
    """
    with torch.device("meta"):
        text_encoder = GemmaTextEncoderConfigurator.from_config({})

    # GEMMA_MODEL_OPS (create_and_populate) reads vision_tower's meta-shape
    # position_ids to size a rope buffer -- this only needs shape metadata
    # (works fine on a meta tensor), but it must run BEFORE vision_tower is
    # nulled out below, matching the order the normal (non-GGUF) build path
    # already uses.
    if GEMMA_MODEL_OPS.matcher(text_encoder):
        text_encoder = GEMMA_MODEL_OPS.mutator(text_encoder)

    inner = text_encoder.model  # Gemma3ForConditionalGeneration
    # Zero real cost (these were meta tensors, i.e. no storage, either way):
    # never loaded, never executed for our text-only hidden-state use case.
    inner.lm_head = None
    inner.model.vision_tower = None
    inner.model.multi_modal_projector = None

    state_dict = load_gemma_gguf_state_dict(gguf_path, dtype=dtype)
    load_result = text_encoder.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[GemmaGGUF] loaded {len(state_dict)} tensors; "
        f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}",
        flush=True,
    )

    # After nulling the unused submodules and loading everything else, no
    # parameter should remain on the meta device. If any do, the mapping
    # above doesn't fully cover what GemmaTextEncoderConfigurator actually
    # builds -- moving to a real device at this point would otherwise
    # either silently no-op (see single_gpu_model_builder.py's
    # _return_model, which skips .to(device) entirely if any meta param
    # remains) or error outright.
    leftover_meta = [name for name, p in text_encoder.named_parameters() if p.device.type == "meta"]
    if leftover_meta:
        raise RuntimeError(
            f"{len(leftover_meta)} parameters still on meta device after GGUF load "
            f"(first 5): {leftover_meta[:5]}"
        )

    text_encoder = text_encoder.to(device=device, dtype=dtype)

    for module_op in module_ops_from_gemma_root(gemma_root):
        if module_op.matcher(text_encoder):
            text_encoder = module_op.mutator(text_encoder)

    return text_encoder.eval()
