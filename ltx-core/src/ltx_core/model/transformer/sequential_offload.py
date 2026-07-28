"""Stream transformer blocks to the GPU one at a time during the forward pass,
keeping them on CPU otherwise.

Motivation: the generator's weights and its activations compete for the same
VRAM. On a 22GB L4 the fp8 generator loads at ~18.1GB, leaving ~2.3GB -- not
enough for the AdaLN timestep projection alone, which needs
tokens * 9 * 4096 * 2 bytes (3.98GB at 497 frames / 736x1280). Since the
blocks are only used one at a time (model.py's _process_transformer_blocks
iterates a plain ModuleList), keeping all 48 resident is pure waste: this
drops block residency to ~1/48th and hands the freed ~17GB to activations.

Cost is one host-to-device weight copy per block per forward pass. That copy
is a constant ~40-65ms per block while per-block compute scales with token
count, so the relative overhead shrinks as the workload grows -- roughly 3%
at 58k tokens, but over 100% below ~3k tokens. That is the right shape: the
regime where offloading is needed (large activations) is the regime where
it's nearly free, and small workloads that would be transfer-bound don't
need offloading in the first place. Hence caller-gated, not always-on.

Deliberately no prefetch stream: at the sizes where this is enabled the
transfer is already ~3% of block time, so double-buffering would add
CUDA-stream and event-sync complexity to reclaim almost nothing.
"""

from __future__ import annotations

import torch

CPU = torch.device("cpu")


def _find_transformer_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the block ModuleList without hardcoding the wrapper nesting
    (X0Model -> velocity_model -> ...), which differs between call paths."""
    for module in model.modules():
        blocks = getattr(module, "transformer_blocks", None)
        if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0:
            return blocks
    raise ValueError(
        "No non-empty 'transformer_blocks' ModuleList found -- sequential offload "
        "cannot be installed on this model."
    )


def install_sequential_offload(model: torch.nn.Module, device: torch.device) -> int:
    """Move every transformer block to CPU and register hooks that bring each
    one to `device` for its forward pass and return it to CPU afterwards.

    Returns the number of blocks offloaded. Everything outside the blocks
    (embeddings, AdaLN, final layers) is left wherever it already is -- those
    are small and used throughout the forward pass.
    """
    blocks = _find_transformer_blocks(model)

    def to_device(module: torch.nn.Module, _args) -> None:
        module.to(device, non_blocking=True)

    def to_cpu(module: torch.nn.Module, _args, _output) -> None:
        module.to(CPU, non_blocking=True)

    for block in blocks:
        block.to(CPU)
        block.register_forward_pre_hook(to_device)
        block.register_forward_hook(to_cpu)

    if device.type == "cuda":
        # The blocks' GPU copies are gone now; release the cached blocks so the
        # freed memory is actually available to the activation allocator rather
        # than sitting in PyTorch's reserved pool.
        torch.cuda.empty_cache()

    return len(blocks)
