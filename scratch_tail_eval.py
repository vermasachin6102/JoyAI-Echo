import subprocess
r = subprocess.run(["bash", "-c",
    "tail -n 20 /content/eval_baseline.log; echo ---PS---; "
    "ps -p $(cat /content/eval_baseline.pid) -o pid,etimes,stat,cmd --no-headers 2>&1 || echo PROCESS_ENDED"],
    capture_output=True, text=True)
print(r.stdout)
