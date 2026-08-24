"""
Gemma Text Encoder Wrapper for DMD distillation.

Provides a simple interface for text encoding without prompt enhancement.
Just pure text -> context embedding conversion.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from ltx_core.loader.registry import Registry


class GemmaTextEncoderWrapper(nn.Module):
    """
    Wrapper for Gemma text encoder to provide DMD-compatible interface.

    This wrapper:
    - Takes raw text prompts (no enhancement needed)
    - Returns conditional_dict with video_context and audio_context
    - Handles batched encoding
    """

    def __init__(
        self,
        text_encoder,
        embeddings_processor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            text_encoder: GemmaTextEncoder instance
            embeddings_processor: EmbeddingsProcessor instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def forward(
        self,
        text_prompts: List[str],
        padding_side: str = "left",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Encode text prompts to conditioning embeddings.

        Args:
            text_prompts: List of text prompts (already processed, no enhancement)
            padding_side: Padding side for tokenizer

        Returns:
            Dictionary containing:
                - video_context: [B, seq_len, dim] video conditioning
                - audio_context: [B, seq_len, dim] audio conditioning
                - attention_mask: [B, seq_len] attention mask
        """
        batch_size = len(text_prompts)

        # Encode each prompt
        video_contexts = []
        audio_contexts = []
        attention_masks = []

        for prompt in text_prompts:
            # 1) Run Gemma LLM to get raw hidden states + attention mask
            hidden_states, attn_mask = self.text_encoder.encode(prompt, padding_side=padding_side)
            # 2) Process hidden states to obtain final embeddings
            output = self.embeddings_processor.process_hidden_states(
                hidden_states, attn_mask, padding_side=padding_side
            )

            video_contexts.append(output.video_encoding)
            audio_contexts.append(output.audio_encoding)
            attention_masks.append(output.attention_mask)

        # Stack batch
        video_context = torch.cat(video_contexts, dim=0) if len(video_contexts) > 0 else None
        # Handle optional audio connector (may be None depending on config)
        if any(ac is None for ac in audio_contexts):
            audio_context = None
        else:
            audio_context = torch.cat(audio_contexts, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0) if len(attention_masks) > 0 else None

        return {
            "video_context": video_context,
            "audio_context": audio_context,
            "attention_mask": attention_mask,
        }

    def encode_batch(
        self,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward() with default padding."""
        return self.forward(text_prompts)


def create_text_encoder_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
) -> GemmaTextEncoderWrapper:
    """
    Factory function to create GemmaTextEncoderWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype

    Returns:
        Configured GemmaTextEncoderWrapper
    """
    from ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )

    text_encoder = ledger.text_encoder().to(device=device, dtype=dtype)
    embeddings_processor = ledger.gemma_embeddings_processor().to(device=device, dtype=dtype)

    wrapper = GemmaTextEncoderWrapper(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        dtype=dtype,
    )

    return wrapper


def create_text_encoder_wrapper_from_gguf(
    gguf_path: str,
    checkpoint_path: str,
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
) -> GemmaTextEncoderWrapper:
    """Same as create_text_encoder_wrapper, but sources the Gemma LLM's
    weights from a language-model-only GGUF file (e.g. Q4_0 quantized)
    instead of the full bf16 safetensors checkpoint -- see
    ltx_core.text_encoders.gemma.gguf_builder for scope and details.

    embeddings_processor (video/audio connectors) is unaffected -- it's
    built from the main LTX-2/JoyAI-Echo checkpoint, unrelated to gemma_root.
    """
    from ltx_core.text_encoders.gemma.gguf_builder import build_gemma_text_encoder_from_gguf
    from ltx_pipelines.utils.model_ledger import ModelLedger

    text_encoder = build_gemma_text_encoder_from_gguf(
        gguf_path=gguf_path,
        gemma_root=gemma_root,
        device=device,
        dtype=dtype,
    )

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        registry=registry,
    )
    embeddings_processor = ledger.gemma_embeddings_processor().to(device=device, dtype=dtype)

    return GemmaTextEncoderWrapper(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        dtype=dtype,
    )


def create_text_encoder_wrapper_from_safetensors_nf4(
    checkpoint_path: str,
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
) -> GemmaTextEncoderWrapper:
    """Same as create_text_encoder_wrapper_from_gguf, but sources NF4 weights
    directly from gemma_root's bf16 safetensors instead of a separate GGUF
    file -- see ltx_core.text_encoders.gemma.safetensors_nf4_builder for
    scope and details, and safetensors_loader.py for why this exists (no
    GGUF download, no CPU-bound dequant, more accurate: single bf16->NF4
    quantization pass instead of the GGUF path's bf16->Q4_0->bf16->NF4).

    UNVERIFIED end-to-end as of introduction -- built and syntax-checked,
    not yet run against a live checkpoint. Do not treat as a drop-in
    replacement for the GGUF path until confirmed: real encode() output,
    no NaN/Inf, and a side-by-side comparison against the GGUF path's
    output (expect close but NOT bit-identical -- this path is meant to be
    MORE accurate, so "different" is the correct outcome, not a bug).
    """
    from ltx_core.text_encoders.gemma.safetensors_nf4_builder import (
        build_gemma_text_encoder_from_safetensors_nf4,
    )
    from ltx_pipelines.utils.model_ledger import ModelLedger

    text_encoder = build_gemma_text_encoder_from_safetensors_nf4(
        gemma_root=gemma_root,
        device=device,
        dtype=dtype,
    )

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        registry=registry,
    )
    embeddings_processor = ledger.gemma_embeddings_processor().to(device=device, dtype=dtype)

    return GemmaTextEncoderWrapper(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        dtype=dtype,
    )
