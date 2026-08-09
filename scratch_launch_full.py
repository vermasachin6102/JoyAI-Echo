import subprocess
log = open("/content/setup_full.log", "w")
p = subprocess.Popen(["bash", "/content/scratch_setup_full.sh"],
                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open("/content/setup_full.pid", "w").write(str(p.pid))
print("full setup launched, pid=", p.pid)
