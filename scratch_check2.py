import subprocess
r = subprocess.run(["bash", "-c",
    "tail -n 30 /content/setup.log; echo ---PS---; "
    "ps -p $(cat /content/setup.pid) -o pid,etimes,stat,cmd --no-headers 2>&1 || echo PROCESS_ENDED; "
    "echo ---DU---; du -sh /content/hf_cache 2>/dev/null"],
    capture_output=True, text=True)
print(r.stdout)
