# Running Colab from this repo — general handoff

General-purpose guide to driving a Google Colab GPU session from the command
line, for anyone/anything doing it — not tied to any one task. For the full
discovery narrative (session-death investigation, dated findings) see
`COLAB_CLI_HANDOFF.md`; this file is the clean cookbook version.

## What this is

`google-colab-cli` (pip package, binary name `colab`) drives a real Colab
runtime from a terminal — create a GPU session, upload/run scripts, download
results, all without opening a browser. Runs inside WSL on this machine
(the package needs `termios`, POSIX-only, doesn't work under native Windows
Python).

## One-time setup

Already done on this machine. If setting up fresh:

```bash
wsl -d Ubuntu -- bash -lc "pipx install google-colab-cli"
```

**Immediately apply this fix** — 0.6.0 ships broken against the `pip`-latest
`jupyter_kernel_client` (renamed a class the CLI still calls by its old
name; every `exec` fails with `AttributeError: ... no attribute
'KernelClient'`):

```bash
wsl -d Ubuntu -- bash -lc "~/.local/share/pipx/venvs/google-colab-cli/bin/python -m pip install --quiet 'jupyter_kernel_client==0.15.0'"
```

Verify: `python -c 'import jupyter_kernel_client as j; print(j.__version__, hasattr(j, "KernelClient"))'` should print `0.15.0 True`.

**HF token, if you'll touch any gated HuggingFace repo** (once per machine,
run this yourself — the token goes straight from your terminal into a file,
never through an agent's tool calls):

```bash
wsl -d Ubuntu -- bash -c 'read -rsp "Paste HF token: " t; printf "%s" "$t" > ~/.hf_token; chmod 600 ~/.hf_token; echo; echo saved'
```

Why not `google.colab.userdata` / Colab Secrets: unreachable from a CLI
session, full stop — `Requesting secret hf timed out. Secrets can only be
fetched when running from the Colab UI.` No workaround exists; it requires
an actual browser frontend.

## Every command goes through a login shell

```bash
wsl -d Ubuntu -- bash -lc "colab <subcommand> ..."
```

Not `wsl -d Ubuntu -- colab ...` — that's a non-login shell, `~/.bashrc`'s
PATH export never loads, `colab: command not found`.

## The one rule that matters most: no competing browser session

**Before creating any CLI session, close every browser tab on
colab.research.google.com and terminate any session shown there (Runtime →
Manage sessions).** A live browser-created session displaces a CLI-created
one within about 2 minutes — Colab appears to prune whichever session has no
attached UI to stay under the account's concurrent-session cap. This is not
about keeping something "warm"; a session with no competing browser tab has
been observed running for hours doing real work. Check first:

```bash
wsl -d Ubuntu -- bash -lc "colab sessions"    # should say "No active sessions found"
```

## Core workflow

```bash
# 1. Create a session (persists across many commands, unlike browser Colab)
wsl -d Ubuntu -- bash -lc "colab new --gpu L4 --session mywork"
#   --gpu: T4, L4, G4, H100, A100 -- availability depends on subscription tier
#   omit --gpu entirely for a CPU-only runtime

# 2. Push the HF token if needed (this upload IS the login -- no interactive step)
wsl -d Ubuntu -- bash -lc "colab exec -s mywork -f setup_mkdir.py --timeout 20"   # mkdir /content/hf_cache first
wsl -d Ubuntu -- bash -lc "colab upload ~/.hf_token /content/hf_cache/token -s mywork"

# 3. Upload any local files the run needs
wsl -d Ubuntu -- bash -lc "colab upload /mnt/c/path/to/script.sh /content/script.sh -s mywork"

# 4. Run code
wsl -d Ubuntu -- bash -lc "colab exec -s mywork -f /mnt/c/path/to/driver.py --timeout 90"

# 5. Pull results back
wsl -d Ubuntu -- bash -lc "colab download /content/output.mp4 /mnt/c/path/to/local.mp4 -s mywork"

# 6. Always stop when done -- GPU sessions are quota/unit-limited
wsl -d Ubuntu -- bash -lc "colab stop -s mywork"
```

## `colab exec -f` — the one semantic that trips everyone up

`-f <path>` reads `<path>` on the **local (WSL) side** and sends its
*content* as the code to run. It does **not** execute a path that already
exists on the remote VM — passing a remote-only path fails with a local
`FileNotFoundError`, not a remote one.

To run something that needs to exist as a real file on the VM (e.g. a shell
script a subprocess opens by path): `colab upload <local> <remote>` it
there first, then separately `colab exec -f <a small local launcher>` that
references the *remote* path.

`--timeout` (seconds) defaults to **30** — far too short for anything but a
smoke test. Downloads and real compute need this set explicitly, often into
the hundreds or thousands.

## Long-running jobs: always detach

`colab exec` is a synchronous RPC — if the client-side `--timeout` is hit,
the *client* gives up waiting, but the code may keep running server-side
with no way to reattach to it cleanly. For anything longer than ~30s, launch
detached and poll a log file instead:

```python
# launcher.py — run via a short `colab exec`, itself does the real work detached
import subprocess
log = open("/content/job.log", "w")
p = subprocess.Popen(["python3", "/content/real_script.py"],
                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open("/content/job.pid", "w").write(str(p.pid))
print("launched, pid=", p.pid)
```

Then poll with short `colab exec` calls reading `tail -n N /content/job.log`
and checking `ps -p $(cat job.pid)`. Note: a subprocess launched this way
still needs its own environment set explicitly (`env=` in `Popen`) — it does
not automatically inherit variables exported only in `~/.bashrc`.

`subprocess.run(cmd, capture_output=True)` **buffers everything until the
process exits** — a genuinely running job looks identical to a hang from the
log. Use `Popen` + read `.stdout` line by line if you need live progress.

## Command reference (real flags, not assumed)

```bash
colab new --gpu <T4|L4|G4|H100|A100> --session <name>
colab sessions                              # list all, including browser-created ([?] prefix)
colab status -s <name>                      # one session's status
colab exec -s <name> -f <local_path> --timeout <secs>
colab upload <local_path> <remote_path> -s <name>
colab download <remote_path> <local_path> -s <name>
colab restart-kernel -s <name>              # fresh Python process, same VM/disk -- see below
colab stop -s <name>
```

Less obvious, real subcommands: `ls`/`rm` (remote file ops), `edit` (edit a
remote file), `console -s <name>` (raw interactive TTY on the VM — use this
for anything genuinely interactive, e.g. `huggingface-cli login`'s masked
prompt), `repl`, `log --output x.ipynb` (save a replayable notebook),
`skill` (prints the CLI's own bundled agent-context doc — read it before
re-deriving behavior from trial and error), `run` (fresh VM, executes,
releases — NOT for anything persistent).

`colab sessions` lists browser-created sessions too, but `colab exec`
**cannot drive them** — only sessions in the CLI's own local
`~/.config/colab-cli/sessions.json`. `colab new` does not adopt an existing
unassigned runtime either; it always creates an additional one.

## `colab restart-kernel` — cheap fix for a stale-import trap

If code in an early `colab exec` call does `import torch` *before* a later
step pins a specific torch version (`pip install --force-reinstall
torch==X`), the already-running kernel keeps the old version cached in
`sys.modules` — the reinstall only rewrites files on disk, it can't touch a
process that already imported the old one. Symptom: version-mismatched
C-extension errors (e.g. `torchvision::nms does not exist`) much later,
looking unrelated to the actual cause.

Fix: `colab restart-kernel -s <name>` — clears the Python process only,
keeps the VM and its disk (so nothing downloaded gets lost). Cheaper than
`colab stop` + `colab new`. Lesson: never run an ad-hoc `import torch`
smoke-test in a session before its pinned-version install step has run.

## Debugging a session that died unexpectedly

Read the CLI's own event log before theorizing:

```bash
cat ~/.config/colab-cli/history/<session-name>.jsonl
```

Records `session_created`, `keep_alive_started`, `keep_alive_error` (with
HTTP status), and `session_terminated` with a `reason`. This answers "why
did it die" directly, in seconds, instead of guessing.

Do **not** `pkill` anything by a loose pattern match on this VM — the
session's own keep-alive daemon process has "keep-alive" and the session's
endpoint string in its command line, so a pattern aimed at some *other*
stuck process can accidentally kill the daemon that's holding the session
alive. If you must kill something, get its exact PID first and kill only
that PID.

## Other gotchas

- `colab auth` alone errors `No active sessions found` — auth is a side
  effect of `colab new`, not a standalone step.
- Files pulled via `download` land in the WSL filesystem
  (`/home/<user>/...`). To get them into a Windows-side repo:
  `wsl -d Ubuntu -- cp ... /mnt/c/path/to/repo/...` or via
  `\\wsl$\Ubuntu\home\<user>\...`.
- `drivemount` works but needs a **fresh, single-use interactive OAuth grant
  every session** (a URL to visit + Enter to confirm) — not a one-time
  account-level grant. Each attempt gets its own token in the auth URL;
  completing an earlier attempt's URL does not satisfy a later retry. Not
  worth the friction for caching a few GB; only reach for it if the
  alternative is re-downloading something genuinely large and slow on every
  session.
- Source: https://github.com/googlecolab/google-colab-cli
