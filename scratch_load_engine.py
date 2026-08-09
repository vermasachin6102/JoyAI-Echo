import sys, time, os
for sub in ["ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"]:
    p = os.path.join("/content/JoyAI-Echo", sub)
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir("/content/JoyAI-Echo")

from inference import InferenceConfig, InferenceEngine

gguf_path = open("/content/gguf_path.txt").read().strip()

t0 = time.perf_counter()
cfg = InferenceConfig(
    "/content/JoyAI-Echo/configs/inference.yaml",
    num_frames=49, video_height=320, video_width=576,
    gemma_gguf_path=gguf_path,
    quantization_fp8_enabled=True,
    sequential_offload_enabled=True,
)
engine = InferenceEngine(cfg)
print(f"[persist] InferenceEngine constructed in {time.perf_counter()-t0:.1f}s", flush=True)

t0 = time.perf_counter()
prompt_files = [
    __import__("pathlib").Path("/content/JoyAI-Echo/prompts/test_001.json"),
    __import__("pathlib").Path("/content/JoyAI-Echo/prompts/test_002.json"),
    __import__("pathlib").Path("/content/JoyAI-Echo/prompts/test_003.json"),
]
cached = engine.encode_all_prompts(prompt_files)
print(f"[persist] Stage 1 (encode all 3 prompt files) took {time.perf_counter()-t0:.1f}s", flush=True)

t0 = time.perf_counter()
engine.load_generator()
print(f"[persist] Stage 2 (generator+VAEs load) took {time.perf_counter()-t0:.1f}s", flush=True)

print("[persist] ENGINE READY -- resident in kernel as `engine`, `cached`, `cfg`", flush=True)
