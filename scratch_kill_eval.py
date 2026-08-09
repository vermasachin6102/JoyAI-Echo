import subprocess
r = subprocess.run(["bash", "-c",
    "kill -9 $(cat /content/eval_baseline.pid) 2>&1; "
    "pkill -9 -f 'inference.py --prompts-glob' 2>&1; "
    "sleep 2; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"],
    capture_output=True, text=True)
print(r.stdout, r.stderr)
