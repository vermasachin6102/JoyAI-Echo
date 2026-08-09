import subprocess, os

from google.colab import userdata
hf_token = userdata.get("hf")
print("HF_TOKEN fetched in-kernel, len=", len(hf_token))

env = os.environ.copy()
env["HF_TOKEN"] = hf_token
env.setdefault("HF_HOME", "/content/hf_cache")

script = r'''
import os, glob, shutil
from huggingface_hub import snapshot_download, hf_hub_download

dst_test = "/content/JoyAI-Echo/checkpoints/test.safetensors"
if os.path.exists(dst_test):
    print("Echo checkpoint already present, skipping:", dst_test)
else:
    echo_dir = snapshot_download(repo_id="jdopensource/JoyAI-Echo",
                                  allow_patterns=["*.safetensors", "*.json", "*.md"])
    cand = glob.glob(os.path.join(echo_dir, "**", "*.safetensors"), recursive=True)
    os.symlink(cand[0], dst_test)
    print("Echo ->", os.path.realpath(dst_test))

print("Downloading gemma-3-12b-it (bf16 dir)...")
gemma_dir = snapshot_download(repo_id="google/gemma-3-12b-it")
dst_gemma = "/content/JoyAI-Echo/checkpoints/gemma-3-12b"
if os.path.islink(dst_gemma): os.remove(dst_gemma)
elif os.path.isdir(dst_gemma): shutil.rmtree(dst_gemma)
os.symlink(gemma_dir, dst_gemma)
print("Gemma ->", os.path.realpath(dst_gemma))

print("Downloading gemma GGUF Q4_0...")
gguf_path = hf_hub_download(repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
                             filename="gemma-3-12b-it-q4_0.gguf")
print("GGUF ->", gguf_path)
with open("/content/gguf_path.txt", "w") as f:
    f.write(gguf_path)

print("ALL DOWNLOADS COMPLETE")
'''
with open("/content/scratch_resume2_inner.py", "w") as f:
    f.write(script)

log = open("/content/setup2.log", "w")  # fresh log -- avoid stale-content ambiguity from before
p = subprocess.Popen(["python3", "/content/scratch_resume2_inner.py"],
                      stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
with open("/content/setup2.pid", "w") as f:
    f.write(str(p.pid))
print("resumed downloads (fresh log), pid=", p.pid)
