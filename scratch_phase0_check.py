import subprocess, sys, shutil, os

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()

print("=== nvidia-smi ===")
out, _ = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
print(out)

print("\n=== torch / cuda ===")
try:
    import torch
    print("torch:", torch.__version__, "| torch.cuda:", torch.version.cuda,
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
except Exception as e:
    print("torch not importable yet:", e)

print("\n=== relevant installed packages ===")
out, _ = sh("pip list 2>/dev/null | grep -iE 'torch|xformers|flash.attn|torchao|lpips|pesq|pystoi|transformers|bitsandbytes|gguf|diffusers|accelerate'")
print(out)

print("\n=== ffmpeg encoders (nvenc?) ===")
out, _ = sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -i nvenc")
print(out if out else "(no nvenc encoders found / ffmpeg not installed yet)")

print("\n=== ffmpeg version ===")
out, _ = sh("ffmpeg -version 2>/dev/null | head -1")
print(out if out else "(ffmpeg not installed)")

print("\n=== disk ===")
out, _ = sh("df -h /content 2>/dev/null || df -h /")
print(out)

print("\n=== existing HF cache ===")
out, _ = sh("du -sh ~/.cache/huggingface 2>/dev/null || echo 'no cache yet'")
print(out)
