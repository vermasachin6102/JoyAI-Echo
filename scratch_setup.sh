#!/bin/bash
set -e
cd /content

# Persistent cache -- local disk (not Drive: this repo's own notebook already
# documented Drive-FUSE reads as the bottleneck for these checkpoint sizes).
# We're staying in ONE session per the mission's hard constraint, so local
# disk survives for the campaign's duration.
export HF_HOME=/content/hf_cache
export HUGGINGFACE_HUB_CACHE=/content/hf_cache/hub
export TORCH_HOME=/content/torch_cache
export TORCHINDUCTOR_CACHE_DIR=/content/inductor_cache
export TRITON_CACHE_DIR=/content/triton_cache
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "export HF_HOME=/content/hf_cache" >> ~/.bashrc
echo "export HUGGINGFACE_HUB_CACHE=/content/hf_cache/hub" >> ~/.bashrc
echo "export TORCH_HOME=/content/torch_cache" >> ~/.bashrc
echo "export TORCHINDUCTOR_CACHE_DIR=/content/inductor_cache" >> ~/.bashrc
echo "export TRITON_CACHE_DIR=/content/triton_cache" >> ~/.bashrc

REPO_ROOT=/content/JoyAI-Echo
if [ ! -d "$REPO_ROOT" ]; then
  git clone https://github.com/vermasachin6102/JoyAI-Echo.git "$REPO_ROOT"
fi
cd "$REPO_ROOT"
git pull origin main

echo "=== [1/4] pinned torch stack ==="
pip install --quiet --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo "=== [2/4] requirements.txt ==="
pip install --quiet -r requirements.txt

echo "=== [3/4] huggingface_hub ==="
pip install --quiet "huggingface_hub[cli]>=0.34.0,<1.0"

python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__; print('torch OK:', torch.__version__)"

echo "=== [4/4] checkpoint downloads (this is the ~78GB step) ==="
python3 <<'PYEOF'
import os, glob
os.environ.setdefault("HF_HOME", "/content/hf_cache")
from huggingface_hub import snapshot_download, hf_hub_download

os.makedirs("/content/JoyAI-Echo/checkpoints", exist_ok=True)

print("Downloading JoyAI-Echo checkpoint...")
echo_dir = snapshot_download(repo_id="jdopensource/JoyAI-Echo",
                              allow_patterns=["*.safetensors", "*.json", "*.md"])
cand = glob.glob(os.path.join(echo_dir, "**", "*.safetensors"), recursive=True)
dst_test = "/content/JoyAI-Echo/checkpoints/test.safetensors"
if os.path.exists(dst_test) or os.path.islink(dst_test): os.remove(dst_test)
os.symlink(cand[0], dst_test)
print("Echo ->", os.path.realpath(dst_test))

print("Downloading gemma-3-12b-it (bf16, tokenizer/config only needed but full snapshot for now)...")
gemma_dir = snapshot_download(repo_id="google/gemma-3-12b-it")
dst_gemma = "/content/JoyAI-Echo/checkpoints/gemma-3-12b"
if os.path.islink(dst_gemma): os.remove(dst_gemma)
elif os.path.isdir(dst_gemma):
    import shutil; shutil.rmtree(dst_gemma)
os.symlink(gemma_dir, dst_gemma)
print("Gemma ->", os.path.realpath(dst_gemma))

print("Downloading gemma GGUF Q4_0...")
gguf_path = hf_hub_download(repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
                             filename="gemma-3-12b-it-q4_0.gguf")
print("GGUF ->", gguf_path)
with open("/content/gguf_path.txt", "w") as f:
    f.write(gguf_path)

print("ALL DOWNLOADS COMPLETE")
PYEOF

echo "=== SETUP COMPLETE ==="
