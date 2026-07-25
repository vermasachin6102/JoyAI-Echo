# Inference pipeline — changes and why

Log of every change made to `inference.py`, `utils.py`, `vae_wrapper.py`,
`configs/inference.yaml`, and the Colab notebook while diagnosing and fixing
inference speed. Ordered chronologically. Each entry: what changed, why.

## 1. Seed-video support (`inference.py`, `configs/inference.yaml`)

Added `paths.seed_video` config + `--seed-video` CLI flag. Reads an existing
mp4 (e.g. a Veo 3 clip), encodes its frames + audio, and pre-populates the
memory bank with it via `_seed_memory_from_video()` before the first shot —
so generation conditions on that clip from shot 0 instead of starting from
pure noise. Currently unused in the notebook (removed from the notebook's
UI, still available via CLI/config) — kept in `inference.py` since it's a
generic capability, not Colab-specific.

## 2. Checkpoint moved off Google Drive (Colab notebook)

`drive.mount()` + Drive-based HF cache removed. Checkpoints now download to
local Colab disk (`/content/hf_cache_local`, 256GB) instead of a Drive FUSE
mount. **Why:** reading the 46GB Echo checkpoint through Drive's FUSE mount
was the actual load-time bottleneck — GPU sat idle while it crawled through
network I/O. Local disk costs a fresh download every new Colab runtime
instead, but every load *within* a session is fast local I/O. Confirmed via
timing: checkpoint load dropped to ~4s once moved to local disk.

## 3. Bottleneck-diagnosis logging (`inference.py`, `utils.py`)

Added `time.perf_counter()`-based logging at every phase boundary that had
none before, since the pipeline was a black box with only per-shot totals:
- CPU↔GPU model stage-swap timing + GPU memory (`_stage_for_denoise` etc.)
- `memory_video`/`memory_audio` tensor shapes logged before each `generate()`
  call (this is what later proved the memory bank's shape churns every shot)
- Per-denoising-step timing inside `BasePipeline`/`MemoryPipeline`
- Stage 2 checkpoint-load timing, split into `create_ltx2_wrapper` vs
  `create_vae_wrappers`
- Decode-phase split: `video_vae.decode_to_pixel`, `audio_vae.decode_to_waveform`,
  postprocess, `write_video`/`torchaudio.save` — previously one unbroken
  ~230s black box

**Why:** every optimization decision below came directly from these numbers,
not guesswork. This logging is why "compile the generator" and "GPU-encode
with NVENC" were correctly ruled out with real data before writing any code
for them, and why tiled decode was correctly identified as the actual win.

## 4. Tiled video-VAE decode — the real fix (`vae_wrapper.py`, `utils.py`, `inference.py`, `configs/inference.yaml`)

**The finding:** logging showed `video_vae.decode_to_pixel` alone was
228.9s — 62% of a 369s total run, bigger than all 8 denoising steps
combined. `VideoDecoder.forward()` runs one monolithic full-resolution pass
(63 latent frames → 497×736×1280×3 pixels) with no chunking. The codebase
already ships a complete tiled-decode system (`TilingConfig`,
`tiled_decode()` in `ltx_core.model.video_vae`) that `VideoVAEWrapper.decode()`
never called.

**The fix:** threaded an optional `tiling_config` through
`VideoVAEWrapper.decode()`/`decode_to_pixel()` → `decode_benchmark_sample()`
→ `InferenceEngine`, built from a new `decode:` YAML section. Chose
`tile_size_frames=32` (not the repo's own `TilingConfig.default()` of 64) —
the run's 63-latent-frame video would make a 64-frame tile a silent
temporal no-op (`63 ≤ 64` triggers the single-tile fallback), so 64 was the
wrong default for this workload.

**Result — confirmed by rerun, not assumed:** `decode_to_pixel` dropped
from 228.9s to **12.45s** (18.4x). Total run: 369.3s → 151.3s (2.4x). This
is the single highest-value change in this whole session. Kept **on by
default** (`tiled_decode_enabled: true`) since the payoff is verified, not
speculative.

## 5. `torch.compile` on the generator — tried, reverted (was `abce958`, reverted in `8dd89e1`)

**Reasoning at the time:** after fix #4, denoising became the new bottleneck
(83% of the shorter run). `torch.compile` reuses fast kernels only when
input shape stays fixed across calls — true within one shot's 8 denoising
steps, but broken across a multi-shot run because the memory bank's shape
grows every shot until it hits `memory.max_size`. Added as an **opt-in**
config flag (`compile_enabled: false` default) specifically so it wouldn't
silently regress the normal multi-shot case.

**Why reverted:** tested it. Real numbers:
- Step 1 (compile): **154.6s** (not the ~30-60s estimated — `max-autotune`
  runs an expensive Triton kernel autotuning search per matmul shape, and
  several candidate kernels failed with `OutOfMemoryError: out of resource`
  against Colab's Triton shared-memory limit, burning search time for
  nothing)
- Step 2: **16.8s** — *worse* than the uncompiled baseline (~15.7-15.9s)
- Step 3: 14.5s — only ~1.4s better than baseline

Root cause, not just "cold start": `torch._dynamo` logs showed
`hit config.recompile_limit (8)` on the transformer's per-layer `forward()`,
because a `self.idx` (layer index) attribute is treated as a compile-time
constant — Dynamo tries to compile a *separate specialized graph per
transformer layer*, hits its cache limit after 8 layers, and silently falls
back to eager execution for the rest. Most of the network was never
actually running compiled code. This is an architectural mismatch in the
model's forward pass, not something the `compile_enabled` toggle can fix by
itself — proper fix would mean patching the model
(`torch._dynamo.config.allow_unspec_int_on_nn_module = True`) or restructuring
the per-layer indexing, out of scope for a config flag.

**Net effect if left on:** worst-realistic total denoise ≈ 258s vs 125.7s
uncompiled — more than 2x *slower*. Reverted cleanly (`git revert abce958`);
tiled decode (#4) and every other change in this doc are unaffected and
still active.

## 6. Colab notebook housekeeping

- Split the setup cell into a heavy one-time row (deps + checkpoint
  downloads) and a fast `git pull`-only row, so code-only iteration doesn't
  require re-running the multi-minute setup every time.
- Streamed `inference.py`'s subprocess output live (`Popen` + line-by-line
  read) instead of `subprocess.run(capture_output=True)`, which buffered
  everything until exit and only printed the last 15 lines — all the
  bottleneck logs from #3 were invisible in Colab until this fix.
- Notebook prompt trimmed to a single short shot + `num_frames` tuned for
  ~10s/~20s test clips, specifically for fast iteration while diagnosing
  the above.

## Current state

Tiled decode: **on**. `torch.compile`: **off**. Net result from session
start: ~369s → ~151s per single-shot test run (2.4x), driven almost
entirely by #4.
