#!/bin/bash
# Full setup: env + pinned torch + ALL downloads. Idempotent/resumable.
# Token is expected at /content/hf_cache/token (uploaded before this runs).
set -x
cd /content

export HF_HOME=/content/hf_cache
export HUGGINGFACE_HUB_CACHE=/content/hf_cache/hub
export TORCH_HOME=/content/torch_cache
export TORCHINDUCTOR_CACHE_DIR=/content/inductor_cache
export TRITON_CACHE_DIR=/content/triton_cache
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
for v in "export HF_HOME=/content/hf_cache" \
         "export HUGGINGFACE_HUB_CACHE=/content/hf_cache/hub" \
         "export TORCH_HOME=/content/torch_cache" \
         "export TORCHINDUCTOR_CACHE_DIR=/content/inductor_cache" \
         "export TRITON_CACHE_DIR=/content/triton_cache"; do
  grep -qxF "$v" ~/.bashrc || echo "$v" >> ~/.bashrc
done

REPO_ROOT=/content/JoyAI-Echo
[ -d "$REPO_ROOT" ] || git clone https://github.com/vermasachin6102/JoyAI-Echo.git "$REPO_ROOT"
cd "$REPO_ROOT" && git pull origin main

echo "=== [1/4] pinned torch stack ==="
pip install --quiet --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo "=== [2/4] requirements + hub ==="
pip install --quiet -r requirements.txt
pip install --quiet "huggingface_hub[cli]>=0.34.0,<1.0"
python -c "import torch, torchvision; assert torch.__version__.startswith('2.8.0'); print('torch', torch.__version__, 'tv', torchvision.__version__)"

echo "=== [3/4] downloads ==="
python3 <<'PYEOF'
import os, glob, shutil
os.makedirs("/content/JoyAI-Echo/checkpoints", exist_ok=True)
from huggingface_hub import snapshot_download, hf_hub_download

dst_test = "/content/JoyAI-Echo/checkpoints/test.safetensors"
if os.path.exists(dst_test):
    print("Echo already present, skipping")
else:
    d = snapshot_download(repo_id="jdopensource/JoyAI-Echo",
                          allow_patterns=["*.safetensors", "*.json", "*.md"])
    c = glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True)
    os.symlink(c[0], dst_test)
    print("Echo ->", os.path.realpath(dst_test))

# Tokenizer/config files ONLY -- module_ops_from_gemma_root reads just
# tokenizer.model + preprocessor_config.json; the 24GB of safetensors shards
# are never touched on the GGUF path. Verified by reading base_encoder.py.
print("Gemma tokenizer/config files only (NOT the 24GB shards)...")
gemma_dir = snapshot_download(
    repo_id="google/gemma-3-12b-it",
    allow_patterns=["*.json", "*.model", "*.txt"],
)
dst_gemma = "/content/JoyAI-Echo/checkpoints/gemma-3-12b"
if os.path.islink(dst_gemma): os.remove(dst_gemma)
elif os.path.isdir(dst_gemma): shutil.rmtree(dst_gemma)
os.symlink(gemma_dir, dst_gemma)
print("Gemma ->", os.path.realpath(dst_gemma))
print("  files:", sorted(os.listdir(gemma_dir)))

gguf_path = hf_hub_download(repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
                             filename="gemma-3-12b-it-q4_0.gguf")
print("GGUF ->", gguf_path)
open("/content/gguf_path.txt", "w").write(gguf_path)
PYEOF

echo "=== [4/4] verify ==="
ls -la /content/JoyAI-Echo/checkpoints/
du -sh /content/hf_cache

echo "=== SETUP COMPLETE ==="
