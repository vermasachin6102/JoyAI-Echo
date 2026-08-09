import subprocess
r = subprocess.run(["bash", "-c",
    "tail -n 45 /content/profile_on.log 2>/dev/null; echo ---PS---; "
    "ps -p $(cat /content/profile.pid) -o etimes,stat --no-headers 2>&1 || echo PROCESS_ENDED; "
    "echo ---GPU---; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"],
    capture_output=True, text=True)
print(r.stdout)
