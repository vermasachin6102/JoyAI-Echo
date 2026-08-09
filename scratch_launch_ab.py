import subprocess, os
env = os.environ.copy()
env.update({
    "HF_HOME": "/content/hf_cache",
    "AB_FRAMES": "497", "AB_HEIGHT": "736", "AB_WIDTH": "1280",
    "AB_OFFLOAD": "true",
    "AB_OUT": "/content/ab_compile.json",
    "TORCHINDUCTOR_CACHE_DIR": "/content/inductor_cache",
    "TRITON_CACHE_DIR": "/content/triton_cache",
})
log = open("/content/ab_compile.log", "w")
p = subprocess.Popen(["python3", "/content/JoyAI-Echo/benchmarks_ab_compile.py"],
                      cwd="/content/JoyAI-Echo", stdout=log, stderr=subprocess.STDOUT,
                      env=env, start_new_session=True)
open("/content/ab.pid", "w").write(str(p.pid))
print("A/B compile test launched, pid=", p.pid)
