"""Small helpers shared by every layer: digests, sizes, refs, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


class ShilpError(Exception):
    """Base class for errors this package raises on purpose."""


class RefError(ShilpError):
    """A model reference (`name`, `name:tag`, `hf:org/repo`) could not be parsed."""


class MissingDependency(ShilpError):
    """An optional dependency or external binary is not installed."""


def now_iso() -> str:
    """UTC timestamp in the same shape GitHub uses, so manifests sort as text."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 and return a `sha256:<hex>` digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_to_filename(digest: str) -> str:
    """`sha256:abc...` -> `sha256-abc...`, the flat blob name used on disk."""
    if not DIGEST_RE.match(digest):
        raise RefError(f"not a sha256 digest: {digest!r}")
    return digest.replace(":", "-")


def human_size(num: float) -> str:
    """Bytes to a short human string: 1.4 GB, 812 MB, 3 KB."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def parse_ref(ref: str) -> tuple[str, str]:
    """Split a model ref into (name, tag). `qwen:q4` -> ("qwen", "q4").

    Names may contain slashes (`team/model`), so only the last colon counts.
    A bare name gets the tag `latest`, exactly like Docker and Ollama.
    """
    ref = ref.strip()
    if not ref:
        raise RefError("empty model reference")
    if ref.startswith("hf:"):
        raise RefError(f"{ref!r} is a Hugging Face ref, not a stored model ref")
    name, sep, tag = ref.rpartition(":")
    if not sep:
        return ref, "latest"
    if not name or not tag:
        raise RefError(f"cannot parse model reference {ref!r}")
    return name, tag


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then rename.

    A reader either sees the old file or the new one, never a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
