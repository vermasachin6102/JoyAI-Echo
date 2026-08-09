import subprocess, os
env = os.environ.copy()
env.update({
    "HF_HOME": "/content/hf_cache",
    "PROFILE_OFFLOAD": "true",     # baseline: as the pipeline ships today
    "PROFILE_FRAMES": "497",
    "PROFILE_HEIGHT": "736",
    "PROFILE_WIDTH": "1280",
    "PROFILE_OUT": "/content/profile_offload_on.json",
})
log = open("/content/profile_on.log", "w")
p = subprocess.Popen(["python3", "/content/JoyAI-Echo/benchmarks_profile_step.py"],
                      cwd="/content/JoyAI-Echo", stdout=log, stderr=subprocess.STDOUT,
                      env=env, start_new_session=True)
open("/content/profile.pid", "w").write(str(p.pid))
print("profile launched, pid=", p.pid)
