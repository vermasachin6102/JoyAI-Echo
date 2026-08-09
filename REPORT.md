# Latency Optimization — Report (interim)

**Status: profiling complete, zero optimizations landed.** Read §4 before
deciding on further spend — the headline finding is that this pipeline is
much closer to the L4's floor than either the original mission brief or the
course correction assumed.

## 1. Measured baseline (NVIDIA L4 22GB, 497 frames, 736×1280)

| stage | time | notes |
|-------|------|-------|
| Stage 1 (GGUF→NF4 text encoder) | **211–233 s** | CPU-bound dequant + encode |
| Stage 2 (generator + VAE load)  | **134–139 s** | fp8 cast, 48-block offload install |
| Denoise                          | **~95 s × 8 steps ≈ 760 s** | dominant cost |
| **Run total**                    | **~19–21 min** | |

Denoise step reproducibility is excellent — `94.93 / 94.94 / 95.01 s` across
separate processes. Noise floor is effectively zero, so any real change is
unambiguous.

## 2. Where the 95 s/step actually goes

`torch.profiler`, leaf CUDA kernels only (aggregated `key_averages()` totals
are inflated ~4× by op nesting — see NOTES.md).

| leaf kernel | CUDA s | % of wall |
|-------------|--------|-----------|
| `flash_fwd_kernel` (attention) | 42.0 | **43%** |
| `cutlass` bf16 GEMM            | 22.1 | 23% |
| elementwise kernels            | ~14.5 | 15% |
| `bfloat16_copy`                | 2.4  | 2% |
| **identified GPU work**        | **~81** | **84%** |

Latent tokens: **57,960** (63 × 23 × 40). Peak VRAM 12.97 GB, resident
weights 0.80 GB.

## 3. Three hypotheses tested — two disproved

### Offload removal — DISPROVED (was believed to be "the whole task")
- Transfer ≤ 8.75 s (≤9% of step), and that figure also includes fp8→bf16
  weight upcasts, so true PCIe traffic is lower still.
- **Full residency does not fit**: activations 12.2 GB + weights 18.1 GB =
  30.3 GB vs 22 GB available. OOM by ~8 GB.
- Partial residency ceiling ≈ 23 of 48 blocks pinned ⇒ **~4% best case.**
- The hand-rolled block streaming is cheap *because* the step is genuinely
  compute-heavy. It was the right design, not the bug it looked like.

### Attention optimization — LITTLE HEADROOM
2642 TFLOP over 42.0 s = **63 TFLOPS achieved ≈ 52% of L4's ~121 TFLOPS
bf16 peak**. Normal-to-good for flash attention, and it is already on
`pytorch_flash::flash_fwd_kernel`. No free kernel swap exists.

### torch.compile — INCONCLUSIVE, likely low-yield
- `mode="reduce-overhead"` **fails structurally**: CUDA graphs reuse output
  buffers, but blocks chain (`vx = vx + self.ff(vx_scaled) * vgate_mlp`,
  transformer.py:392), so block N's output is overwritten by block N+1.
  `cudagraph_mark_step_begin()` cannot fix it — the conflict is *between*
  blocks inside one invocation.
- `mode="default"` run was **lost to session death mid-compilation**.
- Expected yield is low regardless: the block forward is dense with Python
  control flow (`if self.idx >= num_layers*0.7`, `if run_vx`/`run_ax`,
  `getattr(audio, "v2a_grad_scale")`, `torch.is_grad_enabled()`) forcing
  dynamo graph breaks — confirmed by `torch_dynamo_resume_in_forward_at_368`
  in the traceback.

## 4. The honest conclusion

**GPU is 84% busy; attention runs at ~52% of peak; VRAM blocks the one
placement change available.** There is no large, safe win left at this
resolution on this hardware. Remaining candidates, ranked:

| candidate | est. gain | risk | notes |
|-----------|-----------|------|-------|
| NF4 cache (stage 1) | ~3.9 min of ~20 (**~19%**) | none — deterministic, bit-identical | Best remaining ratio. Cache the NF4 state dict keyed on GGUF hash. |
| SageAttention (int8, sm_89) | up to ~20% of step | **high** — changes numerics | Only lever on the 43% attention bucket. Needs full quality harness. |
| torch.compile `mode="default"` | single digits | low | Unfinished; graph breaks cap it. |
| Partial block residency | ~4% | low | Measured ceiling. |
| `cudnn.benchmark` + NVENC | seconds | none | Batch these; Amdahl-irrelevant alone. |

**Recommendation:** land the NF4 cache (best gain-to-risk by a wide margin),
batch the free flags, and treat the denoise loop as near-floor unless
quantized attention is worth the quality risk. A ≥2× overall target is not
reachable on an L4 at 497 frames / 736×1280 without changing the spec.

## 5. Infrastructure findings (cost the majority of the session)

Five Colab sessions died. Root causes, from the CLI's own event log at
`~/.config/colab-cli/history/<session>.jsonl` — **read this file first next
time**:

| session | lifetime | execs | workload |
|---------|----------|-------|----------|
| perf2   | 3h07m    | 4     | downloads only |
| perf-opt| 69 min   | 42    | heavy model load |
| perf5   | 61 min   | 53    | heavy load + torch.compile |
| perf3   | 2 min    | 0     | idle, **browser session concurrent** |
| perf4   | 2.5 min  | 1     | idle, **browser session concurrent** |

Two distinct causes:
1. **Concurrency** — a live browser Colab session appears to displace the
   frontend-less CLI session within ~2 min. Terminate all browser sessions
   before creating a CLI one.
2. **Resource exhaustion under heavy compute** — the two sessions doing real
   model work both died at ~60-70 min with no keep-alive errors, while the
   download-only session lived 3h. Inverse correlation with workload, not
   with idleness.

See COLAB_CLI_HANDOFF.md for the full writeup and the HF-token setup that
removes the repeated manual login.

## 6. What I got wrong

- Estimated stage 1 at 10-12 min from `nvidia-smi` polling gaps. Instrumented
  measurement: **211 s**. Never infer duration from sampled utilization.
- Dismissed the block offload by pattern-matching an absent API name rather
  than the behavior. The conclusion happened to be right; the reasoning was
  not.
- Initially picked NVENC and `allow_tf32` as top candidates — worth seconds
  against a ~20 min run. Chose what was cheap to verify over what was large.
- Wrote bespoke setup scripts instead of reusing the repo's proven notebook
  cell, causing two avoidable failures (missing `HF_TOKEN`; a gemma
  tokenizer-only download that tripped `ModelLedger`'s unconditional
  `model*.safetensors` requirement).
