"""Reconstruct a stored model as a plain directory on disk.

A manifest points at blobs; some tools (llama.cpp's converter, transformers, a
`git lfs` upload) want a directory with normal filenames. This is the bridge,
and it is why every layer records its original `name`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .store import Manifest, Store
from .utils import ShilpError


def materialise(store: Store, manifest: Manifest, dest: Path) -> Path:
    """Write every layer of `manifest` into `dest` under its original name."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ShilpError(f"{dest} is not empty; refusing to overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    for layer in manifest.layers:
        source = store.blob_path(layer.digest)
        if not source.is_file():
            raise ShilpError(f"{manifest.ref} is missing blob {layer.digest}")
        # Layers can be nested (rare, but some repos ship subdirectories).
        target = dest / layer.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return dest


def weights_path(store: Store, manifest: Manifest) -> Path | None:
    """The single GGUF blob a manifest points at, if it has one."""
    from .store import MT_GGUF

    layer = manifest.layer_of(MT_GGUF)
    return store.blob_path(layer.digest) if layer else None
