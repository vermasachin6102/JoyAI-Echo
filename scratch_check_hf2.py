import subprocess
r = subprocess.run(["bash", "-c",
    "echo TOKENFILES:; ls /content/hf_cache/token /content/hf_cache/stored_tokens 2>&1; "
    "echo; echo WHOAMI:; HF_HOME=/content/hf_cache huggingface-cli whoami 2>&1 | tail -3"],
    capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-300:] if r.stderr else "")
