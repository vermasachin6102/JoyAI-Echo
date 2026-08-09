"""Cache the dequantized-from-GGUF bf16 tensors so `iter_gemma_gguf_tensors`'s
CPU-bound Q4_0 decode doesn't get redone from scratch on every process start.

Motivation (measured, see AGENT_HANDOFF.md / NOTES.md from the latency
campaign): stage 1 of inference.py takes ~211-233s, almost entirely CPU-bound
GGUF dequantization -- a deterministic transform of a fixed input file,
recomputed from scratch every single run.

DESIGN NOTE -- this is the second implementation. The first cached
bitsandbytes' already-quantized NF4 state directly, with hand-rolled
module-tree replacement logic mirroring `_assign_tensor`. That hit two
distinct bugs in the reimplemented replacement logic on live testing. Rather
than debug a third variant, this version caches `iter_gemma_gguf_tensors`'s
own unmodified bf16 output and replays it through `_assign_tensor` -- the
same, already-proven function the live streaming path uses. Zero new
tree-navigation code.

ONE CACHE FILE PER TENSOR, not one big file -- gguf_builder.py's own
docstring explains why the streaming design exists at all: materializing the
full ~24GB dequantized state in one Python dict/file is exactly the CPU RAM
problem that was fixed once already. Writing (and later reading) one tensor
at a time, immediately, keeps peak RAM at "one tensor" on both save and load,
matching the live GGUF-streaming path's memory profile.

Cache validity is marked by a manifest file written LAST (after every
tensor's own file), so a save interrupted partway through (crash, OOM kill)
never leaves a directory that looks valid but isn't -- load only trusts a
manifest that's actually there.

This is intentionally NOT a quality-risk change: the cached tensors are
`iter_gemma_gguf_tensors`'s own output, byte for byte -- caching and
replaying them through `_assign_tensor` must produce a bit-identical model.
Verified by `verify_nf4_cache.py` (tensor-by-tensor hash comparison against a
fresh build), not assumed.

Cache key: sha256 of the GGUF file's actual bytes, not path/mtime -- a
changed file at the same path invalidates correctly, a copied/renamed file at
a different path with identical content still hits the cache.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import torch

_HASH_CHUNK = 64 * 1024 * 1024  # 64MB chunks -- large enough to not be syscall-bound
_MANIFEST_NAME = "manifest.json"


def gguf_sha256(gguf_path: str) -> str:
    """Full-file hash, not a cheap proxy (size/mtime) -- correctness of cache
    invalidation matters more than shaving a few seconds off this."""
    h = hashlib.sha256()
    with open(gguf_path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def default_cache_dir(gguf_path: str) -> Path:
    return Path(gguf_path).with_suffix(Path(gguf_path).suffix + ".dequant_cache")


class DequantCacheWriter:
    """Incremental writer -- call `add(key, tensor)` once per tensor as it's
    produced by the live streaming loop, then `finalize()` once at the end.
    A cache directory with no manifest.json is an incomplete/abandoned write
    and `load_dequant_cache` will correctly ignore it."""

    def __init__(self, cache_dir: Path, gguf_hash: str):
        self.cache_dir = cache_dir
        self.gguf_hash = gguf_hash
        self.keys: list[str] = []
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Clear any stale per-tensor files from a prior incomplete/different
        # attempt so a finalized cache never has extra, unreferenced files.
        for f in cache_dir.glob("*.pt"):
            f.unlink()
        manifest = cache_dir / _MANIFEST_NAME
        if manifest.exists():
            manifest.unlink()

    def add(self, key: str, tensor: torch.Tensor) -> None:
        torch.save(tensor, self.cache_dir / f"{key}.pt")
        self.keys.append(key)

    def finalize(self) -> None:
        manifest = {"gguf_sha256": self.gguf_hash, "keys": self.keys}
        tmp_path = self.cache_dir / f"{_MANIFEST_NAME}.tmp"
        tmp_path.write_text(json.dumps(manifest))
        tmp_path.replace(self.cache_dir / _MANIFEST_NAME)  # atomic
        print(f"[DequantCache] saved {len(self.keys)} tensors to {self.cache_dir}", flush=True)


def load_dequant_cache(cache_dir: Path, gguf_hash: str) -> Iterator[tuple[str, torch.Tensor]] | None:
    """Returns None (caller must fall back to the normal GGUF stream) if no
    usable cache exists or the hash doesn't match -- never raises on a stale/
    missing cache, that's an expected, cheap-to-recover-from case. Otherwise
    returns a generator yielding one (key, tensor) at a time, read from disk
    lazily -- same "one tensor in memory at a time" profile as the live path."""
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        print(f"[DequantCache] manifest at {manifest_path} unreadable ({e!r}), rebuilding", flush=True)
        return None
    if manifest.get("gguf_sha256") != gguf_hash:
        print(f"[DequantCache] cache at {cache_dir} is for a different GGUF, rebuilding", flush=True)
        return None

    def _gen() -> Iterator[tuple[str, torch.Tensor]]:
        for key in manifest["keys"]:
            yield key, torch.load(cache_dir / f"{key}.pt", map_location="cpu", weights_only=True)

    return _gen()
