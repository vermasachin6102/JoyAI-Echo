import subprocess, os
env = os.environ.copy()
env["HF_HOME"] = "/content/hf_cache"

script = r'''
import os, shutil
from huggingface_hub import snapshot_download
# Full snapshot -- exactly what the working notebook (seed_veo_3__joy_ai_echo_v1.ipynb
# cell 3) does. ModelLedger.build_model_builders globs model*.safetensors
# unconditionally whenever gemma_root_path is set, even on the GGUF path, so the
# shards must be present regardless of which text-encoder path is used.
d = snapshot_download(repo_id="google/gemma-3-12b-it")
dst = "/content/JoyAI-Echo/checkpoints/gemma-3-12b"
if os.path.islink(dst): os.remove(dst)
elif os.path.isdir(dst): shutil.rmtree(dst)
os.symlink(d, dst)
print("Gemma full ->", os.path.realpath(dst))
import glob
print("safetensors shards:", len(glob.glob(os.path.join(d, "*.safetensors"))))
print("GEMMA FIX COMPLETE")
'''
open("/content/fix_gemma_inner.py", "w").write(script)
log = open("/content/fix_gemma.log", "w")
p = subprocess.Popen(["python3", "/content/fix_gemma_inner.py"],
                      stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
open("/content/fix_gemma.pid", "w").write(str(p.pid))
print("gemma full download launched, pid=", p.pid)
