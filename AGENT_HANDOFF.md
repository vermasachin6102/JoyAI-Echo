# AGENT HANDOFF — Video Pipeline Latency Optimization

**Read this first. Then NOTES.md (raw findings), REPORT.md (results),
COLAB_CLI_HANDOFF.md (how to get a working Colab session).**

Mission: minimize end-to-end wall-clock of the JoyAI-Echo video+audio
generation pipeline on an NVIDIA L4 (22GB, Colab), with zero perceptible
quality loss. Target was ≥2×.

**Status: profiling complete. ZERO optimizations landed.** Everything below
is measured, not guessed, unless explicitly flagged as unverified.

---

## 0. TL;DR for whoever picks this up

1. Do **not** re-run recon. The pipeline is mapped and profiled. Numbers in §2.
2. The two "obvious" wins are **disproved** — offload removal OOMs, attention
   is already near hardware limit. Don't repeat that work (§4).
3. Highest-value untried item: **native fp8 GEMM already exists in this repo
   and is unused** (§3, item 2). Second: **NF4 cache** (§3, item 1).
4. Getting a Colab session to survive is genuinely hard here. Read
   COLAB_CLI_HANDOFF.md §"CRITICAL: session lifetime" before you burn an hour
   like I did.
5. The honest ceiling: ~15-20% per software item. **≥2× is not reachable on
   an L4 at 497 frames / 736×1280 without a spec change.** A100 gets you 3-4×
   for free.

---

## 1. How to run anything at all

Environment: Colab L4 via `google-colab-cli` from WSL. Full details in
COLAB_CLI_HANDOFF.md. Minimum you must know:

```bash
# 1. TERMINATE ALL BROWSER COLAB SESSIONS FIRST. A live browser session
#    causes the CLI session to be pruned within ~2 minutes. Measured.
wsl -d Ubuntu -- bash -lc "colab sessions"          # must be empty
wsl -d Ubuntu -- bash -lc "colab new --gpu L4 --session <name>"

# 2. HF auth: token file is uploaded, never pasted (gated gemma repos).
#    One-time on the machine (user runs, INSIDE WSL not cmd.exe):
#      read -rsp "token: " t; printf "%s" "$t" > ~/.hf_token; chmod 600 ~/.hf_token
#    Then per session:
wsl -d Ubuntu -- bash -lc "colab upload ~/.hf_token /content/hf_cache/token -s <name>"

# 3. Long jobs MUST be detached; a long `colab exec` dies on client timeout:
#      subprocess.Popen([...], stdout=log, start_new_session=True)
#    then poll the log with short exec calls.

# 4. `colab exec -f <path>` reads <path> LOCALLY and ships the source.
#    It does NOT run a path that exists on the VM.
```

Setup script that works: `scratch_setup_full.sh` (in repo root, uncommitted).
~55GB of downloads, ~10 min. **Download the FULL gemma snapshot** — see §5.

---

## 2. Measured baseline (L4 22GB, 497 frames, 736×1280, 8 denoise steps)

| stage | time | share |
|-------|------|-------|
| Stage 1 (GGUF→NF4 text encoder) | 211–233 s | ~19% |
| Stage 2 (generator + VAE load)  | 134–139 s | ~11% |
| Denoise (8 × ~95 s)             | ~760 s    | ~63% |
| **Run total**                   | **~19–21 min** | |

Denoise step reproducibility: `94.93 / 94.94 / 95.01 s` across separate
processes. **Noise floor ≈ 0.** Any real change is unambiguous — you do not
need many reps.

### Inside one 95s denoise step (torch.profiler, LEAF CUDA kernels)

| leaf kernel | CUDA s | % wall |
|-------------|--------|--------|
| `flash_fwd_kernel` (attention) | 42.0 | **43%** |
| `cutlass` bf16 GEMM | 22.1 | 23% |
| elementwise | ~14.5 | 15% |
| `bfloat16_copy` | 2.4 | 2% |
| **identified GPU work** | **~81** | **84% busy** |

Latent tokens **57,960** (63 × 23 × 40). Peak VRAM 12.97 GB. Resident
weights 0.80 GB (offload ON).

> **PROFILER TRAP:** `prof.key_averages()` aggregates NESTED ops. Naive
> bucket sums come out ~4× the wall time (`scaled_dot_product_attention`
> 46.1s ⊃ `_flash_attention_forward` 42.7s ⊃ `flash_fwd_kernel` 42.0s).
> Use leaf `void ...` kernels only. Harness: `benchmarks/profile_step.py`.

### CPU-side (launch overhead — real, partially untapped)
```
Command Buffer Full   88.2 s CPU
cudaLaunchKernel      83.8 s CPU
aten::rms_norm        54.0 s CPU vs 16.1 s CUDA   (3.4× more CPU than GPU)
aten::mul             45.2 s CPU vs 10.5 s CUDA
```
GPU 84% busy ⇒ ~15 s/step of launch gaps.

---

## 3. GENUINE remaining work, ranked

### 1. NF4 quantization cache (stage 1) — ~17% of run, ZERO risk
211 s recomputing a deterministic transform of fixed input files, every run.
Serialize the bitsandbytes NF4 state dict + quant state after first build;
key the cache on a hash of the source GGUF so a changed input invalidates.
Output is **bit-identical** — no quality gate needed, just hash-compare the
reconstructed tensors against a fresh dequant once.
Code: `ltx-core/src/ltx_core/text_encoders/gemma/gguf_builder.py`
(`_assign_tensor` / `build_gemma_text_encoder_from_gguf`).

### 2. Native fp8 GEMM — ALREADY IN THE REPO, UNUSED — ~20% of step
**The most interesting finding.** `QuantizationPolicy.fp8_scaled_mm()`
(`ltx-core/src/ltx_core/quantization/policy.py:29`) uses
`torch.ops.trtllm.cublas_scaled_mm` — real fp8 tensor cores. But
`inference.py:316` hardcodes `QuantizationPolicy.fp8_cast()`, the *upcast*
path, which every step:
- upcasts fp8→bf16 (`_upcast_and_round`, `fp8_cast.py:80`) ≈ **8.7 s**
- then runs a **bf16** GEMM ≈ **22.1 s**, at half Ada's fp8 throughput

Caveats, all real, all verified by reading the source:
- requires `tensorrt_llm` installed (heavy dep, not currently present)
- requires calibration scales (`input_scale`, `weight_scale`) — from a
  pre-quantized checkpoint or an amax calibration pass
- **the only existing call site is stale**: `ltx-pipelines/.../args.py:143`
  calls `fp8_scaled_mm(amax_path)` but the signature is `def fp8_scaled_mm(cls)`
  — takes no args. Signature mismatch ⇒ almost certainly never run in this fork.
- changes numerics ⇒ **full quality gate required**
- note `EXCLUDED_LAYER_SUBSTRINGS` already excludes block 0, blocks 43-47,
  adaln/patchify/proj_out — sensible, keep it.

### 3. Stage 2 load (134 s, ~11%) — NEVER PROFILED
I profiled the denoise step and never instrumented this. The fp8 cast is
deterministic ⇒ same cacheability argument as #1. Unknown split between
disk read / fp8 cast / offload-hook install. **Instrument before assuming.**

### 4. Graph-break removal → torch.compile — single digits, medium effort
The block forward is dense with Python control flow that forces dynamo
breaks: `if self.idx >= int(self.num_layers * 0.7)`, `if run_vx`/`if run_ax`,
`getattr(audio, "v2a_grad_scale", 1.0)`, `torch.is_grad_enabled()`
(`ltx-core/src/ltx_core/model/transformer/transformer.py` ~line 358-400).
Hoisting those out of the hot path would let fusion work.
**CUDA graphs stay blocked regardless** — see §4.

### 5. SageAttention (int8/fp8 attention, sm_89) — up to ~20% of step, HIGH risk
Only lever on the 42 s / 43% attention bucket. Needs a fundamentally cheaper
kernel — attention already runs at 52% of peak on flash, which is normal, so
there's nothing to *tune*. Full quality gate required.

### 6. Partial block residency — ~4% measured ceiling, low risk
~8.8 GB headroom ⇒ ~23 of 48 blocks pinnable. Cuts ~half of a ≤9% bucket.

---

## 4. DEAD ENDS — do not repeat these

| item | verdict | evidence |
|------|---------|----------|
| **Remove sequential block offload** | **OOMs by ~8 GB** | activations 12.2 GB + weights 18.1 GB = 30.3 GB vs 22 GB. Transfer is only ≤8.75 s (≤9% of step) — and that figure already includes fp8 upcasts, so real PCIe traffic is lower. The offload is cheap *because* the step is compute-heavy. |
| **Faster attention kernel** | no headroom | 2642 TFLOP / 42.0 s = **63 TFLOPS = 52% of L4 peak**. Normal-to-good for flash. Already `pytorch_flash::flash_fwd_kernel`. |
| **`torch.compile(mode="reduce-overhead")`** | **structurally impossible** | CUDA graphs reuse output buffers; blocks chain (`vx = vx + self.ff(vx_scaled) * vgate_mlp`, transformer.py:392) so block N's output is overwritten by block N+1. `cudagraph_mark_step_begin()` does NOT fix it — conflict is *between* blocks inside one invocation. |
| **`allow_tf32`** | no-op | pipeline is bf16/fp8 throughout; tf32 only affects fp32 matmul |
| **NVENC swap, `cudnn.benchmark`** | Amdahl-irrelevant | worth seconds against a ~20 min run. Land as a free batch, never as a priority. |
| **CFG batching / adaptive guidance** | **N/A to this model** | `BidirectionalAVInferencePipeline` never uses CFG — it's DMD-distilled few-step. The `cfg_scale` hits in `ltx-pipelines/*.py` belong to *different* pipelines inference.py doesn't call. |
| **diffusers-style offload/tiling/slicing removal** | nothing to remove | repo never used those hooks; its hand-rolled stage-swap + block-offload already serves that role |
| **Step count 8→6** | untried, expect failure | DMD-distilled schedules have minimal slack. Timebox to ONE experiment, after the quality harness exists. |

---

## 5. Traps that cost me time — avoid

1. **Never `import torch` in a Colab kernel before the pinned reinstall.**
   `colab exec` reuses ONE persistent kernel; an early import caches torch
   2.11 in `sys.modules` while disk gets 2.8.0, giving
   `RuntimeError: operator torchvision::nms does not exist` much later.
   Fix is cheap: `colab restart-kernel -s <name>` (keeps the VM + downloads).
2. **Download the FULL gemma snapshot.** Tokenizer-only *looks* sufficient
   (`module_ops_from_gemma_root` reads just `tokenizer.model` +
   `preprocessor_config.json`), but `ModelLedger.build_model_builders`
   (`model_ledger.py:172`) globs `model*.safetensors` **unconditionally**
   whenever `gemma_root_path` is set — including from `create_ltx2_wrapper`,
   which never calls `.text_encoder()`. So the GGUF path cannot run without
   ~24GB it never reads. **Latent bug, left unfixed deliberately** (out of
   scope for latency; fixing it risks the working path).
3. **Use the repo's proven notebook setup** (`seed_veo_3__joy_ai_echo_v1.ipynb`
   cell 3) as the source of truth for install order. I wrote bespoke scripts
   and hit two avoidable failures.
4. **Never estimate duration from sampled `nvidia-smi`.** I inferred "stage 1
   = 10-12 min" from polling gaps; instrumented measurement said **211 s**.
   Add a timer.
5. **`subprocess.run(capture_output=True)` buffers everything** until exit —
   a running job looks identical to a hang. Use `Popen` + line-streaming.
   (`benchmarks/run_eval.py` still has this bug.)

---

## 6. Assets in this repo (all uncommitted as of handoff)

| file | what |
|------|------|
| `NOTES.md` | raw findings, dead ends with reasoning, corrections |
| `REPORT.md` | baseline, profile, hypotheses tested, recommendation |
| `COLAB_CLI_HANDOFF.md` | Colab CLI usage, session-lifetime root cause, HF token setup |
| `benchmarks/profile_step.py` | one-step profiler (works; produced §2) |
| `benchmarks/ab_compile.py` | in-process A/B harness, one model load for both arms |
| `benchmarks/quality_metrics.py` | PSNR/SSIM/LPIPS/LSD/MCD/SI-SDR, self-tested |
| `benchmarks/run_eval.py` | eval-set runner — **has the buffering bug, and reloads the model per job. Rewrite before use.** |
| `scratch_*.py`, `scratch_*.sh` | Colab driver scripts (setup, launch, tail) |

**Quality harness is NOT yet exercised against real output.** Items §3.1 and
§3.3 are output-invariant and can be verified by hashing instead. Items §3.2
and §3.5 genuinely move numerics — build and calibrate the harness (including
a noise-floor run: same seed twice, measure the metric between them) before
gating those.

---

## 7. Suggested order for the next agent

1. Get a session alive (§1). Verify with `colab sessions` that nothing else runs.
2. Land the **NF4 cache** (§3.1). Safe, certain, ~17%. Verify by hash.
3. Instrument **stage 2** (§3.3). 134 s is unexamined; may be another easy cache.
4. Build the **quality harness + noise floor** (needed for anything below).
5. Attempt **native fp8 GEMM** (§3.2). Biggest software lever. Expect to fight
   `tensorrt_llm` install and calibration-scale plumbing; the existing call
   site is stale so treat it as new work, not a config flip.
6. Re-profile and re-plan from whatever dominates then.

Be honest in the report about what did not work. Most of the value in this
handoff is the disproved hypotheses in §4 — they are what stops the next
person spending a day on the offload.
