import subprocess
r = subprocess.run(["bash", "-c",
    "tail -n 6 /content/phaseA.log 2>/dev/null; echo ---PS---; "
    "ps -p $(cat /content/phaseA.pid) -o etimes,stat --no-headers 2>&1 || echo PROCESS_ENDED; "
    "echo ---DU---; du -sh /content/hf_cache 2>/dev/null"],
    capture_output=True, text=True)
print(r.stdout)
