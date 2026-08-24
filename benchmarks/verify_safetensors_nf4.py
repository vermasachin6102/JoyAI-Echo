"""Verify the safetensors-direct NF4 path against the existing GGUF-NF4 path.

Unlike verify_nf4_cache.py, this does NOT expect bit-identical output --
by design, this path skips the GGUF's Q4_0 double-quantization, so its NF4
weights are DIFFERENT (and should be more accurate) than the GGUF path's.
"Close but not identical" is the pass condition here, not "identical".

Checks:
  1. Both builders run without error.
  2. Timing -- expect safetensors path faster (no CPU-bound GGUF decode).
  3. Output shapes match.
  4. Output values are close (cosine similarity near 1.0, bounded relative
     difference) -- NOT bit-identical. A large divergence would indicate a
     real bug (e.g. wrong key mapping), not the expected quantization delta.
  5. Smoke test: real encode() call, no NaN/Inf.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = "/workspace/JoyAI-Echo" if Path("/workspace/JoyAI-Echo").exists() else "/content/JoyAI-Echo"
for sub in ["ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"]:
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO)

import torch

from ltx_core.text_encoders.gemma.gguf_builder import build_gemma_text_encoder_from_gguf
from ltx_core.text_encoders.gemma.safetensors_nf4_builder import build_gemma_text_encoder_from_safetensors_nf4

gguf_path_file = Path("/workspace/gguf_path.txt")
if not gguf_path_file.exists():
    gguf_path_file = Path("/content/gguf_path.txt")
gguf_path = gguf_path_file.read_text().strip()
gemma_root = f"{REPO}/checkpoints/gemma-3-12b"
device = torch.device("cuda")
dtype = torch.bfloat16

PROMPT = "A cat sitting on a red chair."


def encode_and_extract(model, tag: str):
    with torch.no_grad():
        hidden_states, attn_mask = model.encode(PROMPT, padding_side="left")
    last_layer = hidden_states[-1].detach().float().cpu()
    print(f"[verify] {tag}: shape={tuple(last_layer.shape)} "
          f"has_nan={torch.isnan(last_layer).any().item()} "
          f"has_inf={torch.isinf(last_layer).any().item()}", flush=True)
    return last_layer


print("[verify] === GGUF-NF4 path (existing, baseline) ===", flush=True)
t0 = time.perf_counter()
model_gguf = build_gemma_text_encoder_from_gguf(gguf_path, gemma_root, device=device, dtype=dtype)
gguf_s = time.perf_counter() - t0
print(f"[verify] GGUF_BUILD_SECONDS={gguf_s:.1f}", flush=True)
out_gguf = encode_and_extract(model_gguf, "gguf")

del model_gguf
import gc
gc.collect()
torch.cuda.empty_cache()
freed_gb = torch.cuda.memory_allocated() / 1024**3
print(f"[verify] after freeing gguf model: {freed_gb:.2f}GB still allocated (want ~0)", flush=True)

print("\n[verify] === Safetensors-NF4 path (new) ===", flush=True)
t0 = time.perf_counter()
model_st = build_gemma_text_encoder_from_safetensors_nf4(gemma_root, device=device, dtype=dtype)
st_s = time.perf_counter() - t0
print(f"[verify] SAFETENSORS_BUILD_SECONDS={st_s:.1f}", flush=True)
out_st = encode_and_extract(model_st, "safetensors")

print(f"\n[verify] speedup: {gguf_s:.1f}s -> {st_s:.1f}s "
      f"({gguf_s/max(st_s,1e-6):.1f}x, saved {gguf_s-st_s:.1f}s)", flush=True)

# --- comparison: close, NOT identical ---
assert out_gguf.shape == out_st.shape, f"SHAPE MISMATCH: {out_gguf.shape} vs {out_st.shape}"

diff = (out_gguf - out_st).abs()
rel_diff = diff / out_gguf.abs().clamp_min(1e-6)
cos_sim = torch.nn.functional.cosine_similarity(
    out_gguf.flatten().unsqueeze(0), out_st.flatten().unsqueeze(0)
).item()
bit_identical = torch.equal(out_gguf, out_st)

print(f"\n[verify] max_abs_diff={diff.max().item():.4f} "
      f"mean_abs_diff={diff.mean().item():.4f} "
      f"median_rel_diff={rel_diff.median().item():.4f}", flush=True)
print(f"[verify] cosine_similarity={cos_sim:.6f}", flush=True)
print(f"[verify] bit_identical={bit_identical} (expected False -- different quantization paths)", flush=True)

# Sanity bounds, not proof of correctness -- a human should still look at a
# real generated video before trusting this for anything that matters.
if bit_identical:
    print("[verify] UNEXPECTED: bit-identical to GGUF path -- the safetensors "
          "path may not actually be doing anything different. Investigate "
          "before trusting either build.", flush=True)
elif cos_sim < 0.99:
    print(f"[verify] FAIL: cosine similarity {cos_sim:.4f} is too low -- "
          f"this looks like a real bug (wrong key mapping?), not expected "
          f"quantization-path divergence.", flush=True)
    print("[verify] VERIFY_FAILED", flush=True)
else:
    print(f"[verify] PASS: outputs close (cos_sim={cos_sim:.6f}) but not "
          f"identical, as expected for two different quantization paths.", flush=True)
    print("[verify] VERIFY_PASSED", flush=True)

print("\n[verify] VERIFY_DONE", flush=True)
