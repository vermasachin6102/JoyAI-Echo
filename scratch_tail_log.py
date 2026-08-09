import subprocess
r = subprocess.run(["tail", "-n", "40", "/content/setup.log"], capture_output=True, text=True)
print(r.stdout)
r2 = subprocess.run(["bash", "-c", "ps -p $(cat /content/setup.pid) -o pid,etimes,cmd --no-headers 2>&1 || echo PROCESS_ENDED"],
                     capture_output=True, text=True)
print("--- process state ---")
print(r2.stdout)
