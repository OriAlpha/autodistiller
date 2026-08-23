"""Stable fingerprints for configs, datasets and model weights.

Every number AutoDistiller reports is only trustworthy if we can say exactly
what produced it. These helpers turn "what produced it" into short, comparable
hex digests that survive being written to JSON and compared across machines.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DIGEST_LENGTH = 16
"""Truncated digest length. 16 hex chars = 64 bits, plenty for cache keys."""


def _canonical(obj: Any) -> Any:
    """Recursively normalise an object into something json.dumps can order."""
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_canonical(v) for v in obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def hash_obj(obj: Any, *, length: int = DIGEST_LENGTH) -> str:
    """Hash any JSON-ish object with key order and float formatting normalised."""
    payload = json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def hash_text_stream(chunks: Iterable[str], *, length: int = DIGEST_LENGTH) -> str:
    """Hash a stream of documents without holding them all in memory."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.encode("utf-8"))
        digest.update(b"\x00")  # unambiguous document separator
    return digest.hexdigest()[:length]


def hash_file(path: Path, *, length: int = DIGEST_LENGTH, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()[:length]
