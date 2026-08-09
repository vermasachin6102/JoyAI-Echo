"""A/B: torch.compile on the transformer blocks, measured in ONE process.

Why this shape: loading the engine costs ~6 min, so running baseline and
candidate as separate processes doubles that for nothing. Both variants share
one load, one set of inputs, one seed -- which also makes the output
comparison exact rather than approximate.

Targets the launch-overhead finding from profile_step.py, not attention:
  attention  42.0s  -- flash kernel at ~52% of L4 peak, near practical limit
  GPU busy   ~81s of 97s wall  -> ~16s of launch gaps
  rms_norm   54.0s CPU vs 16.1s CUDA   } classic symptoms of thousands of
  Command Buffer Full  88.2s CPU       } small kernel launches

torch.compile fuses elementwise chains and cuts launch count, which is
exactly that bucket. Expected to be output-invariant (may reorder float ops),
so it is gated by a direct output comparison here rather than the full
LPIPS/PESQ harness -- per course-correction §8.
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

FRAMES = int(os.environ.get("AB_FRAMES", "497"))
HEIGHT = int(os.environ.get("AB_HEIGHT", "736"))
WIDTH = int(os.environ.get("AB_WIDTH", "1280"))
OFFLOAD = os.environ.get("AB_OFFLOAD", "true").lower() == "true"
OUT = os.environ.get("AB_OUT", "/content/ab_compile.json")

from inference import InferenceConfig, InferenceEngine
from ltx_distillation.utils import compute_latent_shapes

gguf = Path("/content/gguf_path.txt").read_text().strip()
print(f"[ab] frames={FRAMES} {HEIGHT}x{WIDTH} offload={OFFLOAD}", flush=True)

cfg = InferenceConfig(
    f"{REPO}/configs/inference.yaml",
    num_frames=FRAMES, video_height=HEIGHT, video_width=WIDTH,
    gemma_gguf_path=gguf, quantization_fp8_enabled=True,
    sequential_offload_enabled=OFFLOAD, prompts_glob="test_001.json",
)
engine = InferenceEngine(cfg)
cached = engine.encode_all_prompts([Path(f"{REPO}/prompts/test_001.json")])
engine.load_generator()
engine._stage_for_denoise()
torch.cuda.synchronize()
print("[ab] engine ready", flush=True)

pipe = engine.base_pipeline
cond = {k: (v.to(engine.device) if isinstance(v, torch.Tensor) else v)
        for k, v in list(cached.values())[0][0].items()}
video_shape, audio_shape = compute_latent_shapes(
    num_frames=FRAMES, video_height=HEIGHT, video_width=WIDTH,
    batch_size=1, video_fps=float(cfg.video_fps),
)
device, dtype = engine.device, engine.dtype
B, F_v, F_a = video_shape[0], video_shape[1], audio_shape[1]
sigma = pipe.denoising_sigmas[0]
v_sigma = sigma * torch.ones([B, F_v], device=device)
a_sigma = sigma * torch.ones([B, F_a], device=device)


def make_inputs(seed: int = 1234):
    """Identical inputs for both variants -- fixed seed, regenerated fresh so
    neither variant can mutate what the other sees."""
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(video_shape, device=device, dtype=dtype, generator=g)
    a = torch.randn(audio_shape, device=device, dtype=dtype, generator=g)
    return v, a


def timed_step(tag: str, warmups: int = 1, reps: int = 2):
    for _ in range(warmups):
        v, a = make_inputs()
        with torch.no_grad():
            pipe.generator(noisy_image_or_video=v, conditional_dict=cond,
                           timestep=v_sigma, noisy_audio=a, audio_timestep=a_sigma)
        torch.cuda.synchronize()
    times = []
    out = None
    for _ in range(reps):
        v, a = make_inputs()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipe.generator(noisy_image_or_video=v, conditional_dict=cond,
                                 timestep=v_sigma, noisy_audio=a, audio_timestep=a_sigma)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        # Clone immediately: if any compiled path ever hands back a buffer it
        # may reuse, holding the raw reference across runs reads garbage.
        out = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in out)
    best = min(times)
    print(f"[ab] {tag}: times={[f'{t:.2f}' for t in times]} best={best:.2f}s "
          f"peak_vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True)
    return best, out


# ---- A: baseline, as the pipeline ships today ----
torch.cuda.reset_peak_memory_stats()
base_s, base_out = timed_step("BASELINE")
base_vram = torch.cuda.max_memory_allocated() / 1024**3
base_video = base_out[0].detach().float().cpu().clone()
base_audio = base_out[1].detach().float().cpu().clone()

# ---- B: torch.compile on the transformer blocks ----
# Compiling per-block rather than the whole model: the sequential-offload
# forward hooks move each block CPU<->GPU, which would force graph breaks at
# every block boundary anyway. All 48 blocks share a class and shapes, so
# inductor should compile once and reuse.
from ltx_core.model.transformer.sequential_offload import _find_transformer_blocks

#
# mode="default", NOT "reduce-overhead": the latter wraps each compiled block
# in a CUDA graph, and CUDA graphs reuse output buffers. Blocks chain
# (transformer.py:392 `vx = vx + self.ff(vx_scaled) * vgate_mlp` consumes the
# previous block's output), so block N's graph output is overwritten by block
# N+1's run -> "accessing tensor output of CUDAGraphs that has been
# overwritten by a subsequent run". Measured, not predicted -- reduce-overhead
# was tried first and failed exactly this way.
# `cudagraph_mark_step_begin()` does not fix it: the conflict is BETWEEN
# blocks inside one model invocation, not between invocations.
#
# Expect limited fusion regardless: the block forward has heavy Python control
# flow (`if self.idx >= num_layers*0.7`, `if run_vx`/`run_ax`,
# `getattr(audio, "v2a_grad_scale")`, `torch.is_grad_enabled()`) which forces
# dynamo graph breaks -- confirmed by the `torch_dynamo_resume_in_forward_at_368`
# frame in the traceback.
blocks = _find_transformer_blocks(engine.generator)
print(f"[ab] compiling {len(blocks)} transformer blocks (mode=default)...", flush=True)
t0 = time.perf_counter()
for blk in blocks:
    blk.compile(mode="default")
print(f"[ab] compile() call took {time.perf_counter()-t0:.1f}s "
      f"(actual compilation happens on first forward)", flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
comp_s, comp_out = timed_step("COMPILED", warmups=2)  # extra warm-up: 1st triggers compile
comp_total = time.perf_counter() - t0
comp_vram = torch.cuda.max_memory_allocated() / 1024**3
comp_video = comp_out[0].detach().float().cpu()
comp_audio = comp_out[1].detach().float().cpu()

# ---- determinism / output-invariance gate ----
v_max = (base_video - comp_video).abs().max().item()
a_max = (base_audio - comp_audio).abs().max().item()
v_rel = v_max / max(base_video.abs().max().item(), 1e-12)
print(f"\n[ab] OUTPUT DELTA video_max_abs={v_max:.3e} (rel {v_rel:.3e}) "
      f"audio_max_abs={a_max:.3e}", flush=True)
print(f"[ab] bitwise identical: {torch.equal(base_video, comp_video)}", flush=True)

print(f"\n[ab] ===== RESULT =====", flush=True)
print(f"[ab] baseline : {base_s:.2f}s  peak {base_vram:.2f}GB", flush=True)
print(f"[ab] compiled : {comp_s:.2f}s  peak {comp_vram:.2f}GB", flush=True)
speedup = base_s / comp_s if comp_s > 0 else 0
print(f"[ab] speedup  : {speedup:.3f}x  ({(1-comp_s/base_s)*100:+.1f}% step time)", flush=True)
print(f"[ab] compile+warmup one-off cost: {comp_total:.1f}s", flush=True)

json.dump({
    "frames": FRAMES, "height": HEIGHT, "width": WIDTH, "offload": OFFLOAD,
    "baseline_s": base_s, "compiled_s": comp_s, "speedup": speedup,
    "baseline_peak_gb": base_vram, "compiled_peak_gb": comp_vram,
    "video_max_abs_delta": v_max, "video_rel_delta": v_rel,
    "audio_max_abs_delta": a_max,
    "compile_warmup_total_s": comp_total,
}, open(OUT, "w"), indent=2)
print(f"[ab] wrote {OUT}", flush=True)
print("[ab] AB_DONE", flush=True)
