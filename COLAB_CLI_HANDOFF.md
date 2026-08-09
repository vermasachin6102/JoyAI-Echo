# Colab CLI — Handoff

Remote GPU execution via Google Colab, driven from Windows through WSL. Replaces manual browser Colab usage.

## Environment

- WSL distro: `Ubuntu` (already installed, running)
- Installed via `pipx install google-colab-cli` (package: `google-colab-cli` 0.6.0) inside WSL, not Windows — the package imports `termios`, a POSIX-only module, so it cannot run under native Windows Python.
- Binary: `~/.local/bin/colab` inside WSL user `sachin`. Not on PATH in non-login shells.
- Auth: one-time Google OAuth already completed (`colab new` prompted a browser URL + code exchange). Token is cached in WSL; no need to re-auth unless it expires or is revoked.

## Invocation pattern (from Windows/PowerShell/Bash tool)

Must go through a login shell so `.bashrc` PATH export loads — a bare `wsl -d Ubuntu -- colab ...` fails with `colab: command not found`.

```bash
wsl -d Ubuntu -- bash -lc "colab <subcommand> ..."
```

## Status: verified working (2026-07-31)

Full round-trip confirmed from this repo via the Bash tool:
`colab new --gpu L4` → session READY → `colab exec -f <script>` cloned the
repo and reported `nvidia-smi` (L4, 23034 MiB) → landed on latest `main`
(`46ae776`, includes the sequential-offload stage-swap fix) → `colab stop`.
Did not run a full generation (78GB checkpoint downloads + ~15min compute) —
that's still unverified end-to-end via this CLI path specifically, though it's
the same command the browser Colab notebook already runs successfully.

## Known-good commands (real flags, verified against 0.6.0's actual `--help`)

```bash
wsl -d Ubuntu -- bash -lc "colab new --gpu L4 --session <name>"     # provision GPU session
                                                                     # --gpu: T4, L4, G4, H100, A100 (availability
                                                                     # varies by Colab subscription tier)
wsl -d Ubuntu -- bash -lc "colab status"                            # check active session(s)
wsl -d Ubuntu -- bash -lc "colab install -s <name> <pkgs>"          # pip install into remote runtime
                                                                     # or: -r requirements.txt
wsl -d Ubuntu -- bash -lc "colab exec -s <name> -f script.py --timeout 90"
                                                                     # run a LOCAL FILE remotely.
                                                                     # No -c / inline-code flag exists.
                                                                     # --timeout default is 30s -- too short for
                                                                     # anything beyond a smoke test; pass a real
                                                                     # value for actual work (checkpoint downloads,
                                                                     # denoising loops run into the hundreds/thousands
                                                                     # of seconds).
wsl -d Ubuntu -- bash -lc "colab download <remote_path> <local_path> -s <name>"  # both paths required, positional
wsl -d Ubuntu -- bash -lc "colab log --output x.ipynb"              # save replayable notebook
wsl -d Ubuntu -- bash -lc "colab stop -s <name>"                    # kill session (always do this -- GPU quota)
```

`-f` takes a path resolved by whatever filesystem the WSL command runs
against -- a Windows repo file must be passed as `/mnt/c/Users/verma/video/ai_video/<file>`,
not the Windows path.

**Correction (2026-07-31, found the hard way):** `colab exec -f <path>` reads
`<path>` on the **local (WSL) side** and sends its *content* as the code to
execute -- it does NOT execute a path that already exists on the remote VM.
Passing a remote-only path (e.g. one placed there via `colab upload`) fails
with `FileNotFoundError` on the WSL side, not the VM side. To run a script
that needs to exist as a real file on the VM (e.g. because a subprocess
inside it opens `/content/foo.sh` by path), `colab upload <local> <remote>`
it there first, then separately `colab exec -f <local wrapper>` a small
Python script that references the *remote* path.

Also undocumented in `--help` at a glance but real subcommands worth
knowing: `upload` (`local_path remote_path` -- the missing half of
`download`), `edit` (edit a remote file), `ls`/`rm` (remote file ops),
`run` (fresh VM, executes, releases -- NOT for a persistent session), `repl`,
`console`, `restart-kernel`, `sessions` (list all), `skill` (prints the
bundled `COLAB_SKILL.md` agent reference -- read this before re-deriving
CLI behavior from trial and error).

## HuggingFace auth (gated repos: gemma-3-12b-it, its GGUF)

Colab secrets (`google.colab.userdata`) are **unreachable from a CLI
session** -- confirmed error: `Requesting secret hf timed out. Secrets can
only be fetched when running from the Colab UI.` No workaround; the secrets
vault requires a browser frontend. Does not matter which value is behind
the secret.

Tokens live on the VM's disk, so every new session starts unauthenticated
and a lost session loses the login.

**Setup (once per machine, user runs it -- token never passes through the
agent's tool calls):**
```bash
wsl -d Ubuntu -- bash -c 'read -rsp "Paste HF token: " t; printf "%s" "$t" > ~/.hf_token; chmod 600 ~/.hf_token; echo; echo saved'
```

**Then per session (agent can run this unattended):**
```bash
wsl -d Ubuntu -- bash -lc "colab upload ~/.hf_token /content/hf_cache/token -s <session>"
```
`huggingface_hub` reads `$HF_HOME/token`, so with `HF_HOME=/content/hf_cache`
that upload IS the login -- no `huggingface-cli login`, no interactive paste.
Requires `/content/hf_cache` to exist first (the setup script creates it).

Security: plaintext at `~/.hf_token`, chmod 600 -- same exposure model as
`huggingface-cli login`'s own `~/.cache/huggingface/token`. Kept in WSL home,
deliberately NOT in the repo, so it can never be committed. Revoke at
https://huggingface.co/settings/tokens if the machine is compromised.

## CRITICAL: session lifetime / why CLI sessions get pruned (2026-08-04)

Four sessions died during one campaign. Root cause found in the CLI's own
event log at `~/.config/colab-cli/history/<session>.jsonl` -- READ THAT FILE
FIRST next time, it records `session_created`, `keep_alive_started`,
`keep_alive_error` (with HTTP status), and `session_terminated` with a
`reason`. It answered in one minute what an hour of theorising did not.

Measured:

| session | lifetime | execs | a browser Colab session alive concurrently? |
|---------|----------|-------|--------------------------------------------|
| perf-opt| 69 min   | 42    | no  |
| perf2   | ~3 hours | 4     | no  |
| perf3   | 2 min    | 0     | YES |
| perf4   | 2.5 min  | 1     | YES |

**Concurrency is the variable, not activity.** perf2 survived ~3h on only 4
exec calls; perf3/perf4 died in ~2 min while a browser-created session was
live. Colab appears to prune the frontend-less CLI session to stay under the
account's session cap -- the browser session wins because it has an attached
UI.

**Therefore: terminate ALL browser Colab sessions (Runtime > Manage sessions)
before creating a CLI session.** Do NOT keep a browser tab open "to keep it
alive" -- that is exactly backwards and was a wrong guess made earlier in
this campaign.

Also learned:
- The CLI DOES spawn a keep-alive daemon (`spawn_keep_alive`, pid recorded in
  `SessionState.keep_alive_pid`). Its endpoint
  (`/tun/m/<endpoint>/keep-alive/`) sometimes returns 404, logged as
  `keep_alive_error`. Present in perf2/3/4, ABSENT in perf-opt (which still
  got pruned) -- so 404s are correlated but not the sole cause.
- The CLI can LIST browser-created sessions (`colab sessions` shows them with
  `[?]`) but CANNOT `exec` against them -- it only drives sessions in its own
  `~/.config/colab-cli/sessions.json`. `SessionState` needs name/token/url/
  endpoint; the list API only returns endpoint/accelerator/variant, so
  adopting a browser session is not straightforward.
- `colab new` does NOT adopt an existing unassigned runtime of the same type;
  it creates an additional one (verified -- ended up with two L4s billing
  simultaneously).

## Gotchas

- `colab auth` alone errors `No active sessions found` — auth happens as a side effect of `colab new`, not standalone.
- Downloaded artifacts land in the WSL filesystem (`/home/sachin/...`). To get them into this repo, copy via `\\wsl$\Ubuntu\home\sachin\...` or `wsl -d Ubuntu -- cp ... /mnt/c/Users/verma/video/ai_video/...`.
- Always `colab stop` when done — free Colab GPU sessions are quota-limited and idle sessions burn it.
- **`colab exec` broken out of the box (hit and fixed 2026-07-31):** the CLI's
  pipx venv resolves `jupyter_kernel_client` to its latest release (1.0.0),
  which renamed the class `google-colab-cli` 0.6.0 still calls by its old
  name. Every `exec` fails with
  `AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'`.
  Fix (one-time per WSL install, does not persist across a `pipx reinstall`):
  ```bash
  wsl -d Ubuntu -- bash -lc "~/.local/share/pipx/venvs/google-colab-cli/bin/python -m pip install --quiet 'jupyter_kernel_client==0.15.0'"
  ```
  Verify before relying on it:
  ```bash
  wsl -d Ubuntu -- bash -lc "~/.local/share/pipx/venvs/google-colab-cli/bin/python -c 'import jupyter_kernel_client as j; print(j.__version__, hasattr(j, \"KernelClient\"))'"
  # want: 0.15.0 True
  ```
- Source: https://github.com/googlecolab/google-colab-cli — includes a `COLAB_SKILL.md` agent-context file upstream if deeper reference needed.

## Suggested next step

A real end-to-end generation via this CLI (not just the smoke test above)
would need: `colab new --gpu L4`, `colab install -r requirements.txt`
(or replicate the browser notebook's pinned-torch install sequence), the
~78GB checkpoint downloads (Echo 46GB + Gemma bf16 24GB + Gemma GGUF 8GB),
then `colab exec -f inference.py`-equivalent with a `--timeout` in the
thousands of seconds, then `colab download` the output `.mp4`. Not yet
attempted — real time and Colab compute-unit cost, deferred until actually
needed.
