#!/bin/bash
# Phase A: everything that needs NO HuggingFace auth.
# Idempotent/resumable -- safe to re-run after a session loss.
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

echo "=== [1/3] pinned torch stack ==="
pip install --quiet --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo "=== [2/3] requirements + hub ==="
pip install --quiet -r requirements.txt
pip install --quiet "huggingface_hub[cli]>=0.34.0,<1.0"

python -c "import torch, torchvision; assert torch.__version__.startswith('2.8.0'); print('torch', torch.__version__, 'tv', torchvision.__version__)"

echo "=== [3/3] Echo checkpoint (PUBLIC, no auth needed, ~46GB) ==="
python3 <<'PYEOF'
import os, glob
from huggingface_hub import snapshot_download
os.makedirs("/content/JoyAI-Echo/checkpoints", exist_ok=True)
d = snapshot_download(repo_id="jdopensource/JoyAI-Echo",
                      allow_patterns=["*.safetensors", "*.json", "*.md"])
c = glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True)
dst = "/content/JoyAI-Echo/checkpoints/test.safetensors"
if os.path.exists(dst) or os.path.islink(dst): os.remove(dst)
os.symlink(c[0], dst)
print("Echo ->", os.path.realpath(dst))
PYEOF

echo "=== PHASE A COMPLETE (awaiting HF auth for phase B) ==="
