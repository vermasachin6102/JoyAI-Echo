import subprocess
r = subprocess.run(["bash", "-c",
    "tail -n 30 /content/setup3.log; echo ---PS---; "
    "ps -p $(cat /content/setup3.pid) -o pid,etimes,stat,cmd --no-headers 2>&1 || echo PROCESS_ENDED"],
    capture_output=True, text=True)
print(r.stdout)
