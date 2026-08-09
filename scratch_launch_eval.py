import subprocess, os

log = open("/content/eval_baseline.log", "w")
p = subprocess.Popen(
    ["python3", "/content/JoyAI-Echo/benchmarks_run_eval.py", "baseline"],
    cwd="/content/JoyAI-Echo",
    stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/eval_baseline.pid", "w") as f:
    f.write(str(p.pid))
print("launched baseline eval, pid=", p.pid)
