"""Build a GemmaTextEncoder with weights sourced from a language-model-only
GGUF checkpoint (e.g. Q4_0 quantized), instead of the full bf16 safetensors
checkpoint. See gguf_loader.py for tensor-format details and scope.

Motivation: the full bf16 Gemma-3-12B checkpoint (~24GB, including vision
tower + lm_head we never use) doesn't fit on 24GB-class GPUs like L4 --
confirmed OOM at ~22GB used, failing on the last ~30MB. A Q4_0 GGUF of just
the language-model backbone is ~6.9GB on disk. Two separate memory limits
turned out to matter here, not just GPU:

1. GPU: dequantizing every tensor to bf16 (needed because plain nn.Linear
   can't run matmul against packed Q4_0 data) reproduces the same ~24GB
   footprint as the full checkpoint if it's all moved to GPU at once. Fixed
   by requantizing each decoder Linear layer to bitsandbytes NF4 as it's
   assigned, one at a time.
2. CPU: collecting every dequantized tensor into one big state dict before
   loading it (the original approach) *also* materializes ~24GB in system
   RAM, which alone can exceed a Colab-class instance's RAM before the GPU
   is ever touched. Fixed by streaming tensors one at a time from the GGUF
   reader and assigning/quantizing each immediately (see
   `iter_gemma_gguf_tensors` and `_assign_tensor` below) instead of
   building a full state dict.

Net effect: peak CPU and GPU memory both stay close to "one tensor at a
time" instead of "the whole 24GB model at once" on either side. Note NF4 is
a different scheme than the GGUF's QAT-trained Q4_0 -- this trades away
that QAT-specific quality benefit for actually fitting in 22GB.
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
from ltx_core.text_encoders.gemma.gguf_loader import iter_gemma_gguf_tensors


def _assign_tensor(
    root: torch.nn.Module,
    final_key: str,
    tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Assign one dequantized GGUF tensor into the model, in place. Linear
    weights are requantized to bitsandbytes NF4 and moved to `device`
    immediately; everything else (embed_tokens, norms) is just moved to
    device/dtype. Either way, `tensor` (the bf16 CPU copy) is not referenced
    after this call returns, so it's free to be garbage-collected before the
    next tensor is even dequantized."""
    import bitsandbytes as bnb

    *module_path, leaf = final_key.split(".")
    parent = root
    for part in module_path[:-1]:
        parent = getattr(parent, part)
    module_name = module_path[-1]
    target = getattr(parent, module_name)

    if isinstance(target, torch.nn.Linear) and leaf == "weight":
        quantized = bnb.nn.Linear4bit(
            target.in_features,
            target.out_features,
            bias=target.bias is not None,
            compute_dtype=dtype,
            quant_type="nf4",
        )
        quantized.weight = bnb.nn.Params4bit(tensor, requires_grad=False, quant_type="nf4")
        if target.bias is not None:
            quantized.bias = torch.nn.Parameter(target.bias.data.to(device=device, dtype=dtype))
        quantized.to(device)  # triggers the actual NF4 quantization
        setattr(parent, module_name, quantized)
    else:
        setattr(target, leaf, torch.nn.Parameter(tensor.to(device=device, dtype=dtype)))


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

    # Stream tensors one at a time: dequantize, assign (quantizing Linear
    # weights to NF4 on GPU immediately), then drop the CPU bf16 copy before
    # the next tensor is even read. Never materializes the full ~24GB model
    # in system RAM the way a batch load_state_dict would.
    tensor_count = 0
    for final_key, tensor in iter_gemma_gguf_tensors(gguf_path, dtype=dtype):
        _assign_tensor(text_encoder, final_key, tensor, device=device, dtype=dtype)
        tensor_count += 1
    print(f"[GemmaGGUF] loaded {tensor_count} tensors", flush=True)

    # embed_scale / rotary inv_freq buffers aren't GGUF tensors -- they're
    # created by GEMMA_MODEL_OPS.mutator above and never touched by
    # _assign_tensor, so without this they'd be stranded on CPU while
    # everything else moves to GPU. Targeted moves only (not a blanket
    # text_encoder.to()) so the already-quantized NF4 Linear layers aren't
    # touched again.
    l_model = inner.model.language_model
    l_model.embed_tokens.to(device=device, dtype=dtype)
    l_model.rotary_emb_local.to(device=device, dtype=dtype)
    l_model.rotary_emb.to(device=device, dtype=dtype)

    # After nulling the unused submodules and streaming everything else in,
    # no parameter should remain on the meta device. If any do, the mapping
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

    for module_op in module_ops_from_gemma_root(gemma_root):
        if module_op.matcher(text_encoder):
            text_encoder = module_op.mutator(text_encoder)

    return text_encoder.eval()
