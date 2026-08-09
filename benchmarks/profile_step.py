"""Profile ONE denoise step -- splits the ~99s into transfer / attention /
MLP / other, per the course-correction §2.

Nothing gets optimized until this exists. Order-of-magnitude sanity: 18.5GB
of fp8 weights swept once from VRAM at ~300GB/s is ~60ms, so ~99s/step is
~1000x that -- it is either PCIe offload traffic (48 blocks streamed CPU->GPU
every step), genuinely expensive attention over a large token count, or
something else nobody has looked at.

Runs inside the normal inference path so the numbers are real, not a
synthetic microbenchmark. Uses torch.profiler over exactly one step.
"""

from __future__ import annotations

import json
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
from torch.profiler import ProfilerActivity, profile

OFFLOAD = os.environ.get("PROFILE_OFFLOAD", "true").lower() == "true"
FRAMES = int(os.environ.get("PROFILE_FRAMES", "497"))
HEIGHT = int(os.environ.get("PROFILE_HEIGHT", "736"))
WIDTH = int(os.environ.get("PROFILE_WIDTH", "1280"))
OUT = os.environ.get("PROFILE_OUT", "/content/profile_result.json")

from inference import InferenceConfig, InferenceEngine

gguf = Path("/content/gguf_path.txt").read_text().strip()

print(f"[profile] offload={OFFLOAD} frames={FRAMES} {HEIGHT}x{WIDTH}", flush=True)

cfg = InferenceConfig(
    f"{REPO}/configs/inference.yaml",
    num_frames=FRAMES, video_height=HEIGHT, video_width=WIDTH,
    gemma_gguf_path=gguf,
    quantization_fp8_enabled=True,
    sequential_offload_enabled=OFFLOAD,
    prompts_glob="test_001.json",
)
engine = InferenceEngine(cfg)

t0 = time.perf_counter()
cached = engine.encode_all_prompts([Path(f"{REPO}/prompts/test_001.json")])
stage1_s = time.perf_counter() - t0
print(f"[profile] STAGE1_SECONDS={stage1_s:.1f}", flush=True)

t0 = time.perf_counter()
engine.load_generator()
stage2_s = time.perf_counter() - t0
print(f"[profile] STAGE2_SECONDS={stage2_s:.1f}", flush=True)

# Token count -- the number that decides whether attention is the honest answer.
tokens = ((FRAMES - 1) // 8 + 1) * (HEIGHT // 32) * (WIDTH // 32)
print(f"[profile] LATENT_TOKENS={tokens}", flush=True)

engine._stage_for_denoise()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
resident_gb = torch.cuda.memory_allocated() / 1024**3
print(f"[profile] RESIDENT_AFTER_STAGE_FOR_DENOISE_GB={resident_gb:.2f}", flush=True)

# Build the exact inputs one denoise step sees, then run a single generator
# forward under the profiler. Reaching into the pipeline rather than calling
# .generate() so we time ONE step, not all 8.
pipe = engine.base_pipeline
cond = {k: (v.to(engine.device) if isinstance(v, torch.Tensor) else v)
        for k, v in list(cached.values())[0][0].items()}

from ltx_distillation.utils import compute_latent_shapes
video_shape, audio_shape = compute_latent_shapes(
    num_frames=FRAMES, video_height=HEIGHT, video_width=WIDTH, batch_size=1,
    video_fps=float(cfg.video_fps),
)
print(f"[profile] video_shape={video_shape} audio_shape={audio_shape}", flush=True)

device, dtype = engine.device, engine.dtype
video = torch.randn(video_shape, device=device, dtype=dtype)
audio = torch.randn(audio_shape, device=device, dtype=dtype)
sigma = pipe.denoising_sigmas[0]
B, F_v, F_a = video_shape[0], video_shape[1], audio_shape[1]
v_sigma = sigma * torch.ones([B, F_v], device=device)
a_sigma = sigma * torch.ones([B, F_a], device=device)

# Warm-up step (first step pays lazy-init / autotune costs -- never time it).
print("[profile] warm-up step...", flush=True)
with torch.no_grad():
    pipe.generator(noisy_image_or_video=video, conditional_dict=cond,
                   timestep=v_sigma, noisy_audio=audio, audio_timestep=a_sigma)
torch.cuda.synchronize()

print("[profile] profiled step...", flush=True)
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, profile_memory=False, with_stack=False) as prof:
    with torch.no_grad():
        pipe.generator(noisy_image_or_video=video, conditional_dict=cond,
                       timestep=v_sigma, noisy_audio=audio, audio_timestep=a_sigma)
    torch.cuda.synchronize()
step_s = time.perf_counter() - t0
print(f"[profile] STEP_SECONDS={step_s:.2f}", flush=True)
print(f"[profile] PEAK_VRAM_GB={torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)

# Bucket the kernels. Names are matched loosely on purpose -- report the
# residual as "other" rather than force-fitting everything into a bucket.
events = prof.key_averages()
buckets = {"transfer": 0.0, "attention": 0.0, "mlp_linear": 0.0, "other": 0.0}
rows = []
for e in events:
    cuda_us = getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0) or 0
    cpu_us = e.cpu_time_total or 0
    name = e.key.lower()
    t_us = max(cuda_us, 0)
    if any(k in name for k in ("memcpy", "copy_", "to_copy", "pin_memory", "htod", "dtoh")):
        b = "transfer"
    elif any(k in name for k in ("attention", "sdpa", "flash", "bmm", "softmax", "scaled_dot")):
        b = "attention"
    elif any(k in name for k in ("gemm", "linear", "addmm", "matmul", "mm_", "cutlass", "cublas")):
        b = "mlp_linear"
    else:
        b = "other"
    buckets[b] += t_us / 1e6
    rows.append((e.key, cuda_us / 1e6, cpu_us / 1e6))

total_bucketed = sum(buckets.values())
print("\n[profile] ===== BUCKETS (CUDA time, seconds) =====", flush=True)
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"[profile]   {k:12s} {v:8.2f}s  ({v/max(total_bucketed,1e-9)*100:5.1f}%)", flush=True)
print(f"[profile]   {'SUM':12s} {total_bucketed:8.2f}s  (wall step: {step_s:.2f}s)", flush=True)

print("\n[profile] ===== TOP 25 KERNELS BY CUDA TIME =====", flush=True)
for name, cu, cp in sorted(rows, key=lambda r: -r[1])[:25]:
    print(f"[profile]   {cu:8.3f}s cuda | {cp:8.3f}s cpu | {name[:80]}", flush=True)

json.dump({
    "offload": OFFLOAD, "frames": FRAMES, "height": HEIGHT, "width": WIDTH,
    "latent_tokens": tokens, "stage1_s": stage1_s, "stage2_s": stage2_s,
    "step_s": step_s, "resident_gb": resident_gb,
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    "buckets": buckets,
    "top_kernels": [{"name": n, "cuda_s": c, "cpu_s": p}
                     for n, c, p in sorted(rows, key=lambda r: -r[1])[:40]],
}, open(OUT, "w"), indent=2)
print(f"\n[profile] wrote {OUT}", flush=True)
print("[profile] PROFILE_DONE", flush=True)
