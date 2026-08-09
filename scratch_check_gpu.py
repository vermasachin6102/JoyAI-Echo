import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", r.stdout.strip())
r2 = subprocess.run(["bash", "-c", "ps aux | grep inference.py | grep -v grep"],
                     capture_output=True, text=True)
print("inference.py process:", r2.stdout.strip() or "(none running)")
