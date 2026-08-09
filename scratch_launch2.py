import subprocess
log = open("/content/phaseA.log", "w")
p = subprocess.Popen(["bash", "/content/scratch_setup2.sh"],
                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open("/content/phaseA.pid", "w").write(str(p.pid))
print("phase A launched, pid=", p.pid)
