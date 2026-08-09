import subprocess, os

os.makedirs("/content", exist_ok=True)
log = open("/content/setup.log", "w")
p = subprocess.Popen(
    ["bash", "/content/scratch_setup.sh"],
    stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True,   # detach from this exec's process group -- survives
)
with open("/content/setup.pid", "w") as f:
    f.write(str(p.pid))
print("launched setup, pid=", p.pid)
