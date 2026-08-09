import subprocess
r = subprocess.run(["bash", "-c",
    "tail -n 25 /content/ab_compile.log 2>/dev/null; echo ---PS---; "
    "ps -p $(cat /content/ab.pid) -o etimes,stat --no-headers 2>&1 || echo PROCESS_ENDED"],
    capture_output=True, text=True)
print(r.stdout)
