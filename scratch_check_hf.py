import os, subprocess
# Deliberately NOT importing torch here -- phase A's pinned reinstall may still
# be running, and caching a stale torch in this kernel is exactly the trap that
# already cost a debugging cycle this session.
os.environ.setdefault("HF_HOME", "/content/hf_cache")

r = subprocess.run(["bash", "-c",
    "ls -la /content/hf_cache/token /content/hf_cache/stored_tokens 2>&1 | head -5"],
    capture_output=True, text=True)
print("token files:\n", r.stdout)

try:
    from huggingface_hub import whoami
    info = whoami()
    print("WHOAMI OK ->", info.get("name"), "| type:", info.get("type"))
except Exception as e:
    print("WHOAMI FAILED ->", type(e).__name__, str(e)[:200])
