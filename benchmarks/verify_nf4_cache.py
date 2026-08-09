"""Verify the NF4 cache: build once (cache miss, writes cache), build again
(cache hit), confirm bit-identical weights and measure the real time saved.

This is the whole correctness argument for landing the cache without a full
quality-metric gate -- see nf4_cache.py's docstring.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = "/content/JoyAI-Echo"
for sub in ["ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"]:
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO)

import torch
import bitsandbytes as bnb

from ltx_core.text_encoders.gemma.gguf_builder import build_gemma_text_encoder_from_gguf
from ltx_core.text_encoders.gemma.nf4_cache import default_cache_dir

gguf_path = Path("/content/gguf_path.txt").read_text().strip()
gemma_root = "/content/JoyAI-Echo/checkpoints/gemma-3-12b"
device = torch.device("cuda")
dtype = torch.bfloat16

cache_dir = default_cache_dir(gguf_path)
if cache_dir.exists():
    import shutil
    shutil.rmtree(cache_dir)
    print(f"[verify] removed pre-existing cache at {cache_dir} for a clean test", flush=True)


def extract_state(model) -> dict:
    """Same extraction shape as nf4_cache.save_nf4_cache -- weight_data for
    Linear4bit, tensor for plain params. Used purely for comparison here."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            state[name + ".weight_data"] = module.weight.data.clone().cpu()
            if module.bias is not None:
                state[name + ".bias"] = module.bias.data.clone().cpu()
    l_model = model.model.model.language_model
    state["embed_tokens.weight"] = l_model.embed_tokens.weight.data.clone().cpu()
    state["norm.weight"] = l_model.norm.weight.data.clone().cpu()
    for i, layer in enumerate(l_model.layers):
        for attr in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm",
                     "self_attn.q_norm", "self_attn.k_norm"):
            sub = layer
            for part in attr.split("."):
                sub = getattr(sub, part)
            state[f"layers.{i}.{attr}.weight"] = sub.weight.data.clone().cpu()
    return state


print("[verify] === BUILD 1 (expect cache MISS, writes cache) ===", flush=True)
t0 = time.perf_counter()
model1 = build_gemma_text_encoder_from_gguf(gguf_path, gemma_root, device=device, dtype=dtype)
build1_s = time.perf_counter() - t0
print(f"[verify] BUILD1_SECONDS={build1_s:.1f}", flush=True)
state1 = extract_state(model1)
del model1
import gc
gc.collect()  # force-break any reference cycle BEFORE empty_cache -- del alone
torch.cuda.empty_cache()  # doesn't guarantee the refcount hits zero immediately
freed_gb = torch.cuda.memory_allocated() / 1024**3
print(f"[verify] after freeing model1: {freed_gb:.2f}GB still allocated (want ~0)", flush=True)

print("\n[verify] === BUILD 2 (expect cache HIT) ===", flush=True)
t0 = time.perf_counter()
model2 = build_gemma_text_encoder_from_gguf(gguf_path, gemma_root, device=device, dtype=dtype)
build2_s = time.perf_counter() - t0
print(f"[verify] BUILD2_SECONDS={build2_s:.1f}", flush=True)
state2 = extract_state(model2)

print(f"\n[verify] speedup: {build1_s:.1f}s -> {build2_s:.1f}s "
      f"({build1_s/max(build2_s,1e-6):.1f}x, saved {build1_s-build2_s:.1f}s)", flush=True)

# --- bit-identical check ---
assert state1.keys() == state2.keys(), (
    f"KEY MISMATCH missing={state2.keys()-state1.keys()} extra={state1.keys()-state2.keys()}"
)
mismatches = []
for k in state1:
    if not torch.equal(state1[k], state2[k]):
        mismatches.append((k, float((state1[k].float() - state2[k].float()).abs().max())))
if mismatches:
    print(f"\n[verify] FAIL: {len(mismatches)} tensors differ:", flush=True)
    for k, d in mismatches[:10]:
        print(f"[verify]   {k}: max_abs_diff={d}", flush=True)
    print("[verify] BIT_IDENTICAL=False", flush=True)
else:
    print(f"\n[verify] PASS: all {len(state1)} tensors bit-identical", flush=True)
    print("[verify] BIT_IDENTICAL=True", flush=True)

# --- smoke test: cached model actually runs ---
# torch.no_grad() matters here, not just for speed: the real pipeline's
# GemmaTextEncoderWrapper.forward() is @torch.no_grad()-decorated, but this
# script calls .encode() directly, bypassing that. Without it, autograd
# builds a full graph over a 12B-param forward -- the likely cause of the
# prior OOM at 22GB (model alone is ~13GB resident).
print(f"\n[verify] model2 resident before smoke test: "
      f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
print("[verify] === smoke test: encode with the cache-loaded model ===", flush=True)
with torch.no_grad():
    hidden_states, attn_mask = model2.encode("A cat sitting on a red chair.", padding_side="left")
last_layer = hidden_states[-1]  # hidden_states is a tuple of per-layer tensors
print(f"[verify] num_layers={len(hidden_states)} last_layer_shape={tuple(last_layer.shape)} "
      f"has_nan={torch.isnan(last_layer).any().item()} has_inf={torch.isinf(last_layer).any().item()}", flush=True)

print("\n[verify] VERIFY_DONE", flush=True)
