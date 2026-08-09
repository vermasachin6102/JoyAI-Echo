import subprocess, os

script = r'''
import os, glob, shutil
os.environ.setdefault("HF_HOME", "/content/hf_cache")

from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("hf")
print("HF_TOKEN loaded from Colab secret.")

from huggingface_hub import snapshot_download, hf_hub_download

dst_test = "/content/JoyAI-Echo/checkpoints/test.safetensors"
if os.path.exists(dst_test):
    print("Echo checkpoint already present, skipping (cached from prior run):", dst_test)
else:
    echo_dir = snapshot_download(repo_id="jdopensource/JoyAI-Echo",
                                  allow_patterns=["*.safetensors", "*.json", "*.md"])
    cand = glob.glob(os.path.join(echo_dir, "**", "*.safetensors"), recursive=True)
    os.symlink(cand[0], dst_test)
    print("Echo ->", os.path.realpath(dst_test))

print("Downloading gemma-3-12b-it (bf16 dir -- now with token)...")
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

with open("/content/scratch_resume_downloads_inner.py", "w") as f:
    f.write(script)

log = open("/content/setup.log", "a")
log.write("\n=== RESUMED after HF_TOKEN fix ===\n")
log.flush()
p = subprocess.Popen(["python3", "/content/scratch_resume_downloads_inner.py"],
                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
with open("/content/setup.pid", "w") as f:
    f.write(str(p.pid))
print("resumed downloads, pid=", p.pid)
