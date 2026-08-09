"""Baseline / optimized-run harness for the latency-optimization campaign.

Runs a fixed eval set through inference.py, capturing per-stage timing
(parsed from inference.py's own existing log lines -- it already instruments
stage_for_denoise, per-step denoise timing, decode timing, run_total) and
saving outputs for the quality gate. Warms up once before the timed runs
per the mission's Phase 2 guidance (GPU throttles from idle; first run is
never representative).

Eval set: two speed tiers.
  - "fast" tier: real production prompts (test_001-003) at reduced settings
    (49 frames, 320x576) so iteration doesn't cost 10+ min per attempt.
  - "full" tier: test_001 at real production settings (497 frames,
    736x1280) -- run this before accepting anything into the final report,
    not on every single Tier-0 micro-change.

This is a TEST METHODOLOGY choice, not a pipeline change -- the "fast" tier
never touches what inference.py ships by default (see configs/inference.yaml,
untouched). Mission constraint #6 is about not silently shipping degraded
settings as the optimization; using smaller settings to iterate faster is
the mission's own explicit Phase 2 guidance.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path("/content/JoyAI-Echo")
OUT_ROOT = Path("/content/eval_runs")

FAST_SETTINGS = dict(num_frames=49, video_height=320, video_width=576)
FULL_SETTINGS = dict(num_frames=497, video_height=736, video_width=1280)

EVAL_JOBS = [
    {"name": "fast_001", "prompt_file": "test_001.json", "seed": 42, **FAST_SETTINGS},
    {"name": "fast_002", "prompt_file": "test_002.json", "seed": 42, **FAST_SETTINGS},
    {"name": "fast_003", "prompt_file": "test_003.json", "seed": 42, **FAST_SETTINGS},
    {"name": "full_001", "prompt_file": "test_001.json", "seed": 42, **FULL_SETTINGS},
]

_STAGE_PATTERNS = {
    "stage1_load_s": re.compile(r"\[Stage 1\].*?(\d+\.?\d*)s"),  # best-effort; refine once real log seen
    "stage2_load_s": re.compile(r"total_load_time=([\d.]+)s"),
    "denoise_step_s": re.compile(r"denoise_step=(\d+)/(\d+).*?step_time=([\d.]+)s"),
    "shot_total_s": re.compile(r"shot=\d+.*?denoise=([\d.]+)s decode=([\d.]+)s total=([\d.]+)s"),
    "run_total_s": re.compile(r"run_total=([\d.]+)s"),
}


@dataclass
class RunResult:
    name: str
    wall_time_s: float
    returncode: int
    log_path: Path
    output_mp4: Path | None
    denoise_step_times: list[float] = field(default_factory=list)
    decode_s: float | None = None
    run_total_s: float | None = None


def _parse_log(log_text: str) -> dict:
    step_times = [float(m.group(3)) for m in _STAGE_PATTERNS["denoise_step_s"].finditer(log_text)]
    shot_m = _STAGE_PATTERNS["shot_total_s"].search(log_text)
    run_m = _STAGE_PATTERNS["run_total_s"].search(log_text)
    return {
        "denoise_step_times": step_times,
        "decode_s": float(shot_m.group(2)) if shot_m else None,
        "run_total_s": float(run_m.group(1)) if run_m else None,
    }


def run_job(job: dict, gguf_path: str, fp8: bool, offload: bool, out_dir: Path) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{job['name']}.log"

    cmd = [
        "python", "inference.py",
        "--prompts-glob", job["prompt_file"],
        "--num-frames", str(job["num_frames"]),
        "--video-height", str(job["video_height"]),
        "--video-width", str(job["video_width"]),
        "--seed", str(job["seed"]),
        "--gemma-gguf-path", gguf_path,
        "--quantization-fp8-enabled", "true" if fp8 else "false",
        "--sequential-offload-enabled", "true" if offload else "false",
        "--output-root", str(out_dir),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    log_path.write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    parsed = _parse_log(proc.stdout)

    mp4s = sorted(out_dir.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime)
    return RunResult(
        name=job["name"], wall_time_s=wall, returncode=proc.returncode,
        log_path=log_path, output_mp4=mp4s[-1] if mp4s else None,
        denoise_step_times=parsed["denoise_step_times"],
        decode_s=parsed["decode_s"], run_total_s=parsed["run_total_s"],
    )


def run_eval_set(tag: str, gguf_path: str, fp8: bool = True, offload: bool = True,
                  jobs: list[dict] | None = None, warm_up: bool = True) -> list[RunResult]:
    """tag: label for this run set, e.g. 'baseline' or 'change_03_nvenc'."""
    jobs = jobs if jobs is not None else EVAL_JOBS
    out_dir = OUT_ROOT / tag
    results = []

    if warm_up:
        print(f"[eval] warm-up run (fast_001, not timed for the record)...", flush=True)
        run_job(jobs[0], gguf_path, fp8, offload, out_dir / "_warmup")

    for job in jobs:
        print(f"[eval] running {job['name']} (tag={tag})...", flush=True)
        r = run_job(job, gguf_path, fp8, offload, out_dir)
        results.append(r)
        print(f"[eval]   -> wall={r.wall_time_s:.1f}s rc={r.returncode} "
              f"steps={len(r.denoise_step_times)} run_total={r.run_total_s}", flush=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps([{
        "name": r.name, "wall_time_s": r.wall_time_s, "returncode": r.returncode,
        "output_mp4": str(r.output_mp4) if r.output_mp4 else None,
        "denoise_step_times": r.denoise_step_times, "decode_s": r.decode_s,
        "run_total_s": r.run_total_s,
    } for r in results], indent=2))
    print(f"[eval] summary written to {summary_path}", flush=True)
    return results


if __name__ == "__main__":
    import sys
    gguf = Path("/content/gguf_path.txt").read_text().strip()
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    jobs = EVAL_JOBS if len(sys.argv) <= 2 else [j for j in EVAL_JOBS if j["name"] == sys.argv[2]]
    run_eval_set(tag, gguf, jobs=jobs)
