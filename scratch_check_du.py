import subprocess
r = subprocess.run(["bash", "-c", "du -sh /content/hf_cache 2>/dev/null; echo ---; ps -p $(cat /content/setup.pid) -o pid,etimes,cmd --no-headers 2>&1 || echo PROCESS_ENDED; echo ---; tail -c 800 /content/setup.log"],
                    capture_output=True, text=True)
print(r.stdout)
