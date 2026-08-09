# Latency Optimization — NOTES

## Environment

- Target: NVIDIA L4, 24GB, Colab CLI (see COLAB_CLI_HANDOFF.md for the exact
  invocation patterns and the `jupyter_kernel_client` pin fix -- read that
  file instead of re-deriving CLI usage).
- Repo: this checkout. Entrypoint: `python inference.py --config configs/inference.yaml`
  (or CLI overrides -- see inference.py's argparse).

## Colab CLI (see COLAB_CLI_HANDOFF.md -- not duplicating here)

## Pipeline map (from local repo reading -- Phase 1, no GPU cost)

Entry: `inference.py` `InferenceEngine`, two-stage design (see class docstring).

1. **Stage 1 -- text encode** (`encode_all_prompts`, inference.py:~254)
   - Gemma-3-12B text encoder. Two build paths: full bf16 safetensors, or
     GGUF Q4_0 requantized to bitsandbytes NF4 (`--gemma-gguf-path`).
   - Encoder fully released (`del text_encoder; gc.collect(); empty_cache()`)
     before stage 2 loads. Confirmed in logs: releases cleanly, ~13GB peak
     via GGUF path.
2. **Stage 2 -- generator + VAEs load** (`load_generator`, inference.py:~271)
   - Generator: LTX DiT (`create_ltx2_wrapper`), optional fp8 weight-only
     quantization (`QuantizationPolicy.fp8_cast()`, `--quantization-fp8-enabled`).
     Confirmed working (float8_e4m3fn: 18.5B params in real run logs).
   - Optional sequential block-offload (`--sequential-offload-enabled`,
     `ltx_core.model.transformer.sequential_offload`): streams the 48
     transformer blocks CPU<->GPU one at a time via forward hooks. Confirmed
     working, ~2GB resident vs ~18.1GB.
   - Video VAE, audio VAE, vocoder also loaded (`create_vae_wrappers`).
3. **Per-shot stage-swap** (inference.py `_stage_for_denoise`,
   `_stage_for_video_encode`, `_stage_for_decode`, `_stage_after_video_encode`)
   - Manual CPU<->GPU juggling between generator and VAE pieces to fit 22GB.
   - This IS this repo's offload strategy -- there are no diffusers-style
     `enable_model_cpu_offload`/attention-slicing/vae-tiling hooks anywhere in
     the codebase (grepped, zero hits). Nothing to "turn off" per the mission's
     Tier 1 item -- the custom stage-swap already serves that role and is
     already minimal (only moves what's needed, confirmed via the
     `move_excluding_blocks` fix that had to preserve it against a blanket
     `.to()`).
4. **Denoise loop** (`BidirectionalAVInferencePipeline.generate`,
   ltx-distillation/.../bidirectional_pipeline.py:156)
   - Few-step DMD-**distilled** model. denoising.sigmas in configs/inference.yaml
     has 9 entries -> 8 actual denoise steps (matches real log:
     `denoise_step=1/8`...`8/8`).
   - **No CFG anywhere in this path** -- `BidirectionalAVInferencePipeline`
     never receives/passes guidance_scale or cfg_scale (confirmed: grepped
     inference.py's construction call, ltx-pipelines/*.py's cfg_scale usage is
     a DIFFERENT set of pipelines -- a2vid_two_stage.py, ti2vid_one_stage.py,
     keyframe_interpolation.py -- not what inference.py calls). Tier-2's
     "batch CFG cond+uncond" and "skip late-timestep CFG" items are **N/A for
     this model** -- recording as dead-end-before-trying, not an oversight.
   - `torch.cuda.synchronize()` called every step purely to print step timing
     (bidirectional_pipeline.py:206). Real stall, but likely near-zero net
     cost given each step is ~99s of actual compute (launch-overhead-bound
     vs compute-bound analysis says this should be negligible) -- MUST
     measure before/after removing, not assume (mission rule).
5. **Decode + mux**
   - `_stage_for_decode`: generator->CPU, VAE decoders + vocoder->GPU.
   - ffmpeg: **two call sites use software `libx264`**, confirmed the target
     Colab L4 image's ffmpeg (4.4.2) HAS `h264_nvenc`/`hevc_nvenc` compiled in
     (verified via `ffmpeg -encoders`, not assumed). Real Tier-0/1 candidate:
     - `ltx-distillation/src/ltx_distillation/utils.py:229` (`-c:v libx264 -preset medium -crf 18`,
       used for concatenating multi-shot output)
     - `ltx-pipelines/src/ltx_pipelines/utils/media_io.py:205,358` (`add_stream("libx264", ...)`)

## Precision / compile flags -- NOT SET anywhere in repo (grepped, zero hits)

- No `torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32` / `cudnn.benchmark`.
- No `torch.compile` actually used (one hit is a comment referencing it, not
  a real call -- inference.py:683).
- Real candidates, both need on-VM verification before landing: tf32 only
  matters for fp32 matmuls -- need to confirm which submodules (VAE? vocoder?)
  actually run any fp32 compute in this bf16/fp8 pipeline before claiming a
  win. `torch.compile` needs shape-stability check (fixed num_frames/height/
  width per run already true here -- one shape per `inference.py` invocation,
  so recompilation risk is low WITHIN a run, but the mission's own inductor
  cache setup makes cross-run compile cost reusable too).

## Environment facts (from live Colab L4 VM, 2026-07-31)

- Base image: torch 2.11.0+cu128, transformers 5.13.1, diffusers 0.39.0,
  accelerate 1.14.0, torchao 0.10.0 -- ALL get replaced by this repo's pinned
  `torch==2.8.0` + requirements.txt install (repo has a documented, painful
  history requiring the exact pins -- see numpy_debugging_report.md). Do not
  skip the pinned reinstall to save time.
- `h264_nvenc`, `hevc_nvenc` confirmed present in `ffmpeg -encoders` (ffmpeg 4.4.2).
- Disk: 236GB total, 189GB free before any downloads -- plenty for the ~78GB
  one-time checkpoint set.
- No pre-existing HF cache -- clean slate, cache dirs set before first import
  (see setup script).

## Confirmed Tier-0/1 candidates (verified against real installed source, not assumed)

- **NVENC swap**: `torchvision.io.write_video`'s real signature (pinned
  0.23.0, checked via `inspect.signature`/`inspect.getsource` on the live
  VM): `video_codec: str = "libx264"`, passed straight through to
  PyAV/ffmpeg. `h264_nvenc` confirmed available in this ffmpeg build (see
  above). One-line change at both call sites in
  `ltx-distillation/src/ltx_distillation/utils.py` (~line 158, 177,
  `write_benchmark_media`). NOT YET APPLIED -- needs a before/after
  measurement + quality gate first per mission rule.
  (Note: this whole API is deprecated in torchvision >=0.22, removal in
  0.24 -- out of scope to migrate off it for a latency task, noting only.)

- **tf32 / cudnn.benchmark flags**: natural insertion point is inference.py
  right after `import torch` (line 20), before `main()` (line 882) --
  executes once at process start regardless of CLI args, matches every run.
  `cudnn.benchmark=True` is the more likely real win here (auto-tunes conv
  algorithm selection; VAE/vocoder are conv-heavy, shape is FIXED for the
  duration of a single inference.py invocation -- no repeated re-benchmark
  cost across the run, only paid once per distinct shape at first use).
  `allow_tf32` only affects fp32 matmuls -- given this pipeline is
  bf16/fp8 nearly everywhere (confirmed: fp8 generator, bf16 VAE/vocoder/
  text-encoder), likely a near-total no-op UNLESS some specific submodule
  runs fp32 internally -- needs checking before claiming any win from it,
  not assumed. NOT YET APPLIED, needs baseline first.

## Session gotcha (real, hit and fixed 2026-07-31)

`colab exec` reuses the SAME persistent kernel across calls within a
session. My first smoke-test script did a bare `import torch` BEFORE the
pinned torch==2.8.0 reinstall ran (which only rewrites files on disk via a
separate subprocess) -- the kernel had 2.11.0 cached in `sys.modules` from
that point on. Every subsequent `import torchvision` (reading the correctly
-pinned 0.23.0 from disk) then broke with
`RuntimeError: operator torchvision::nms does not exist` -- classic
torch/torchvision ABI mismatch, and exactly the failure class this repo's
own `numpy_debugging_report.md` and the browser notebook's "deliberately no
`import torch` here" comment already warned about, just via the CLI path
instead of the notebook. Fix: `colab restart-kernel -s <session>` -- clears
the kernel's Python process (fresh `sys.modules`) without tearing down the
VM/session, so the ~78GB of already-downloaded checkpoints on disk survive.
Lesson: never run ad-hoc `import torch`/`import torchvision` smoke tests in
a session BEFORE its pinned-version install step has completed.

## Harness bug (found live, not yet fixed -- not urgent)

`run_eval.py`'s `run_job()` uses `subprocess.run(cmd, capture_output=True)`
to invoke inference.py -- this BUFFERS the child's entire stdout/stderr and
only returns it when the process exits. Every inference.py progress line
(`[GemmaGGUF]`, `denoise_step=`, stage timings) is invisible to log-tailing
until each job completes, even though inference.py itself flushes every
print immediately. Looks identical to a hang from the outside (confirmed:
3 identical 60s-apart polls with zero new content, direct process check
showed it alive and progressing normally underneath). Fix for next
iteration: `Popen` + stream to the log file line-by-line as it's produced,
matching the pattern the browser notebook's `generate_video()` already
uses (`Popen(..., stdout=PIPE) ; for line in proc.stdout: ...`).

## Stage 1 (GGUF dequant) timing -- MEASURED 211s, earlier estimate was WRONG

**Correction.** Instrumented measurement: `STAGE1_SECONDS=211.0` (~3.5 min),
covering GGUF dequant + NF4 quantize + encoding 15 prompts. My earlier
"10-12 min" figure below was inferred from `nvidia-smi` polling gaps and is
WRONG -- polling only told me GPU util was 0%, not when stage 1 started or
ended. Never estimate a duration from sampled utilization again; add a timer.

This materially shrinks the NF4-cache prize (course-correction §4 assumed
~11 min). Against a ~25 min run, caching stage 1 saves ~3.5 min at most,
not ~11. Still worth doing (it is deterministic recomputation of a fixed
transform = textbook Tier 0 waste) but it is NOT the dominant cost and
should be sequenced after the denoise profile, which is ~13 min of the run.

Original (incorrect) note kept below for the record:

First-ever cold-cache run: stage 1 (CPU-bound GGUF dequantization, all 626
tensors via `iter_gemma_gguf_tensors`) took roughly 10-12 min before GPU
utilization ever left 0%. Confirmed via direct `nvidia-smi` polling, not
assumed -- looked exactly like a hang from log output alone (matches the
`subprocess.run(capture_output=True)` blind-spot already noted), but
`ps` showed sustained 80-95% CPU the whole time and GPU util went 0% ->
100% right after. This is much slower than casual impressions from earlier
interactive runs suggested (those never isolated stage-1 duration alone).
Real Tier-2/3 candidate worth investigating later: is single-tensor-at-a-
time CPU dequant+NF4-quantize (added for the system-RAM fix) now the
dominant cost for SHORT jobs specifically, since it's a fixed cost
independent of video length/resolution? Needs a real per-stage timer
added to gguf_builder.py to confirm before touching anything -- not
assumed. Also observed RSS spiking to ~36GB during this -- unclear yet
whether that's genuine peak usage or first-touch page-cache inflation from
reading the freshly-downloaded 78GB for the first time; will check if it
drops between jobs.

## SESSION LOST 2026-07-31 (constraint #1 writeup -- why nothing else worked)

Session `perf-opt` (gpu-l4-s-kkb-ass1b0-3ueckq6cmjn9p) died mid-run during
the ~15min kernel-resident engine load. `colab status` -> "No active
sessions"; `colab sessions` listed `gpu-l4-s-kkb-ass1a2-2aca9aollrbrw` with
status `[?]`, but both `status -s` and `exec -s` against it returned
"Session not found" -- stale registry artifact, not a recoverable VM.
Nothing to reconnect to; this was not a recoverable-by-restart-kernel case
(that trick was already used successfully earlier for the sys.modules
issue -- see above -- and does not apply when the VM itself is gone).

LOST: all 78GB of checkpoints, the pinned torch 2.8.0 env, the HF token
file, /content entirely.

Cause: not determinable post-mortem (VM gone with its logs). Candidates,
none confirmed:
  - VM OOM. Observed RSS ~36GB earlier on a 53GB box; the persistent-engine
    load does CPU-side GGUF dequant AND then reads the 46GB safetensors.
  - Colab preemption / idle timeout.
  - Compute-unit exhaustion (user had ~34 units, burn ~1.5-2/hr).

ARCHITECTURAL LESSON (this is the actionable part):
I put a ~15-minute blocking load inside a single `colab exec` call, making
model residency depend on the kernel + websocket staying healthy for the
whole duration. Every long-running thing that SURVIVED this session was a
detached `subprocess.Popen(..., start_new_session=True)` writing to a log
file; the one thing that died was the kernel-resident one.
Mission §3d offers two options and I picked the weaker one for this
environment: "module-level singleton" (kernel-resident objects) couples
weight residency to a fragile transport. The "persistent model server"
option (detached process + socket/watched-dir, clients are short-lived)
survives kernel restarts, client timeouts, AND websocket drops. If this is
retried: detached server process, never a long kernel exec.

## PROFILE RESULT (1 denoise step, 497f 736x1280, offload ON) -- 2026-08-04

Wall 96.97s. Peak VRAM 12.96GB. Resident weights 0.80GB. 57,960 latent tokens.
video_shape=[1,63,128,23,40] audio_shape=[1,497,128].

HARNESS CAVEAT: `prof.key_averages()` aggregates NESTED ops, so naive bucket
sums are inflated (my buckets summed 376s against a 97s wall --
`scaled_dot_product_attention` 46.1s contains `_flash_attention_forward`
42.7s contains `flash_fwd_kernel` 42.0s). Use LEAF CUDA kernels only:

| leaf kernel            | CUDA s | % wall |
|------------------------|--------|--------|
| flash_fwd_kernel       | 42.0   | 43%    |
| cutlass bf16 GEMM      | 22.1   | 23%    |
| elementwise kernels    | ~14.5  | 15%    |
| bfloat16_copy          | 2.4    | 2%     |
| **identified GPU work**| **~81**| **84%**|

### Attention is at its practical limit -- do NOT chase it with kernel swaps
4*N^2*d * 48 layers = 2642 TFLOP over 42.0s = **63 TFLOPS achieved**, i.e.
~52% of L4's ~121 TFLOPS bf16 peak. That is normal-to-good for flash
attention. It is already using `pytorch_flash::flash_fwd_kernel`. The only
real lever is a QUANTIZED attention kernel (SageAttention supports sm_89),
which changes numerics and needs the full quality gate.

### Offload removal: DISPROVED as the big win (course-correction §3 answered)
- Transfer bucket <= 8.75s (`aten::copy_`), and that figure ALSO includes the
  fp8->bf16 weight upcasts, so true PCIe traffic is less. <= 9% of the step.
- Full residency would OOM: activations = 12.96 peak - 0.80 resident =
  ~12.2GB; weights 18.1GB; total ~30.3GB vs 22GB available. **Does not fit.**
- Partial residency ceiling: ~8.8GB headroom = ~23 of 48 blocks pinned =>
  ~half of <=9% => **~4% best case.** Not a priority.
- So the 48-block streaming is cheap *because* the step is genuinely
  compute-heavy. The hand-rolled offload was the right call for this model at
  this resolution, not the mistake it looked like.

### NEW finding not in either plan: launch/CPU overhead
```
Command Buffer Full   88.2s CPU
cudaLaunchKernel      83.8s CPU
aten::rms_norm        54.0s CPU vs 16.1s CUDA   (3.4x more CPU than GPU)
aten::mul             45.2s CPU vs 10.5s CUDA
```
GPU busy only ~84% of wall => ~16s of launch gaps per step. Symptoms of
thousands of small kernel launches. This is the torch.compile / CUDA-graphs
target and is the largest remaining output-invariant lever.

### fp8 upcast tax
`aten::to`/`aten::_to_copy` ~8.7s -- every Linear upcasts fp8->bf16 every
step (fp8_cast.py:80 `_upcast_and_round`). Inherent to upcast-during-
inference. Noted, not yet attacked.

## Dead ends

- **CFG batching / adaptive guidance skip (Tier 2)**: N/A. This pipeline's
  actual runtime path (`BidirectionalAVInferencePipeline`) never uses CFG at
  all -- it's a few-step distilled model. Confirmed by reading the real
  construction call in inference.py, not assumed from the mission's generic
  playbook.
- **Diffusers-style offload/tiling/slicing removal (Tier 1)**: N/A, nothing
  to remove. This codebase never used those hooks -- its own hand-rolled
  stage-swap + block-offload already serves that role and is already minimal.

  **REOPENED then RE-CLOSED with data (2026-08-04).** The reviewer correctly
  called out that my original dismissal was pattern-matching on an API name
  (`enable_model_cpu_offload` absent) rather than on behavior (48 blocks
  crossing PCIe per step), and that "nothing should cross PCIe inside the
  denoise loop if it can stay resident" was the real mission item. That
  criticism was right about my REASONING. But the profile then showed the
  original conclusion was accidentally correct: transfer is <=9% of the step
  and full residency OOMs by ~8GB (see PROFILE RESULT above). Keeping both
  the correction and this resolution recorded -- the lesson is about how the
  first conclusion was reached, not about its outcome.

- **allow_tf32**: dropped without measuring, on the reviewer's instruction and
  my own prior reasoning -- pipeline is bf16/fp8 nearly everywhere, so it is
  a near-certain no-op and not worth a benchmark run against a ~25min job.

- **Step-count reduction 8->6 (Tier 2)**: not attempted yet. Reviewer's
  calibration: DMD-distilled schedules have very little slack, expect it to
  fail the quality gate, timebox to ONE experiment. Sequenced after the
  quality harness exists.

## Phase 2 plan (quality harness -- real gap, not yet built)

inference.py ALREADY logs per-stage wall time via print statements
(`stage_for_denoise`, per-step `denoise_step=i/N ... step_time=`, decode
timing, `run_total=`) -- Phase 2's timing-instrumentation requirement is
largely already satisfied, nothing to build there. Real gap: no automated
quality-metric harness exists (no PSNR/SSIM/LPIPS/audio-metric code anywhere
in the repo). Needed before ANY Tier 1+ change can be gated:
- `colab install lpips` (video perceptual metric) -- not in the base image's
  pip list (checked, absent) nor requirements.txt.
- PESQ/STOI/audio metrics -- also absent, need install if used.
- PSNR/SSIM can come from `torchmetrics` (need to check if present) or a
  small numpy implementation (cheap, no install needed) -- prefer avoiding
  a new dependency if a few lines of code does it, matching this repo's
  existing minimal-dependency style.
- Eval set: reusing `prompts/french_lesson.json` (already exercised
  end-to-end this session) as one fixed case. Need 2-4 more small/fast
  ones for iteration speed -- keep at least one short (few-second, low-res)
  case so the eval loop itself doesn't cost 10+ min per run.
