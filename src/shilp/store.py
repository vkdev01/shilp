"""The model store: content-addressed blobs plus small JSON manifests.

Layout under `$SHILP_HOME`:

    blobs/sha256-<hex>                  file content, named by its own hash
    manifests/<name>/<tag>.json         what a model *is*: layers + params
    recipes/<name>/<tag>.json           how it was made: every step, in order

Why content addressing: two tags that share a weights file store it once, and a
manifest can never silently point at a file whose bytes changed — the name is
the checksum. This is the same idea as OCI images and Ollama's own store, which
is why the API shapes below will look familiar.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .utils import (
    RefError,
    ShilpError,
    atomic_write_json,
    digest_to_filename,
    human_size,
    now_iso,
    parse_ref,
    read_json,
    sha256_bytes,
    sha256_file,
)

SCHEMA_VERSION = 1

# Media types let the loader decide how to open a layer without guessing from
# the extension. They are strings, not enums, so a new format is a new string.
MT_GGUF = "application/vnd.shilp.gguf"
MT_SAFETENSORS = "application/vnd.shilp.safetensors"
MT_CONFIG = "application/vnd.shilp.config+json"
MT_TEMPLATE = "application/vnd.shilp.template"


class StoreError(ShilpError):
    """A model or blob is missing, or a manifest is malformed."""


@dataclass(frozen=True)
class Layer:
    """One file that belongs to a model."""

    digest: str
    size: int
    media_type: str
    #: Filename to restore when exporting the model back to a directory.
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Layer":
        return cls(
            digest=d["digest"],
            size=int(d["size"]),
            media_type=d["mediaType"],
            name=d.get("name", "blob"),
        )


@dataclass
class Manifest:
    """A named, tagged model: its layers, its default sampling, its template."""

    name: str
    tag: str = "latest"
    created: str = field(default_factory=now_iso)
    layers: list[Layer] = field(default_factory=list)
    #: Default sampling and context. Request bodies override these per call.
    params: dict[str, Any] = field(default_factory=dict)
    #: Chat template in llama.cpp's Jinja/`{{ .Prompt }}` dialect, or None to
    #: use whatever the GGUF file already carries.
    template: str | None = None
    system: str | None = None
    #: Free-form facts for `/api/show` and the CLI: architecture, parameter
    #: count, quantization, where it came from.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Digest of the manifest this one was derived from, if any.
    parent: str | None = None
    engine: str | None = None

    # --- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "name": self.name,
            "tag": self.tag,
            "created": self.created,
            "parent": self.parent,
            "engine": self.engine,
            "layers": [layer.to_dict() for layer in self.layers],
            "params": self.params,
            "template": self.template,
            "system": self.system,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        if d.get("schemaVersion") != SCHEMA_VERSION:
            raise StoreError(f"unsupported manifest schema {d.get('schemaVersion')!r}")
        return cls(
            name=d["name"],
            tag=d.get("tag", "latest"),
            created=d.get("created", now_iso()),
            layers=[Layer.from_dict(x) for x in d.get("layers", [])],
            params=d.get("params") or {},
            template=d.get("template"),
            system=d.get("system"),
            metadata=d.get("metadata") or {},
            parent=d.get("parent"),
            engine=d.get("engine"),
        )

    # --- convenience ---------------------------------------------------
    @property
    def ref(self) -> str:
        return f"{self.name}:{self.tag}"

    @property
    def size(self) -> int:
        return sum(layer.size for layer in self.layers)

    @property
    def human_size(self) -> str:
        return human_size(self.size)

    def layer_of(self, media_type: str) -> Layer | None:
        """First layer with the given media type — a model has one weights file."""
        for layer in self.layers:
            if layer.media_type == media_type:
                return layer
        return None


class Store:
    """Reads and writes blobs, manifests and recipes under one root directory."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.blobs = self.home / "blobs"
        self.manifests = self.home / "manifests"
        self.recipes = self.home / "recipes"

    # --- blobs ---------------------------------------------------------
    def blob_path(self, digest: str) -> Path:
        return self.blobs / digest_to_filename(digest)

    def has_blob(self, digest: str) -> bool:
        return self.blob_path(digest).is_file()

    def put_blob(self, src: Path, *, move: bool = False) -> Layer:
        """Copy (or move) a file into the blob store and return its layer."""
        src = Path(src)
        if not src.is_file():
            raise StoreError(f"no such file: {src}")
        digest = sha256_file(src)
        dest = self.blob_path(digest)
        self.blobs.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            # Same bytes already stored: keep one copy, drop the new one.
            if move:
                src.unlink(missing_ok=True)
        elif move:
            shutil.move(str(src), dest)
        else:
            shutil.copyfile(src, dest)
        return Layer(
            digest=digest,
            size=dest.stat().st_size,
            media_type=MT_GGUF,
            name=src.name,
        )

    def put_bytes(self, data: bytes, name: str, media_type: str) -> Layer:
        """Store an in-memory payload (a config, a template) as a blob."""
        digest = sha256_bytes(data)
        dest = self.blob_path(digest)
        if not dest.exists():
            self.blobs.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return Layer(digest=digest, size=len(data), media_type=media_type, name=name)

    def read_blob_text(self, digest: str) -> str:
        path = self.blob_path(digest)
        if not path.is_file():
            raise StoreError(f"missing blob {digest}")
        return path.read_text(encoding="utf-8")

    # --- manifests -----------------------------------------------------
    def manifest_path(self, name: str, tag: str) -> Path:
        # `name` may contain a slash; each segment becomes a directory.
        return self.manifests.joinpath(*name.split("/")) / f"{tag}.json"

    def save_manifest(self, manifest: Manifest) -> None:
        atomic_write_json(self.manifest_path(manifest.name, manifest.tag), manifest.to_dict())

    def load_manifest(self, name: str, tag: str = "latest") -> Manifest:
        path = self.manifest_path(name, tag)
        if not path.is_file():
            raise StoreError(f"model {name}:{tag} not found (try `shilp ls`)")
        return Manifest.from_dict(read_json(path))

    def resolve(self, ref: str) -> Manifest:
        name, tag = parse_ref(ref)
        return self.load_manifest(name, tag)

    def list(self) -> list[Manifest]:
        """Every stored model, newest first."""
        out: list[Manifest] = []
        if not self.manifests.exists():
            return out
        for path in self.manifests.rglob("*.json"):
            try:
                out.append(Manifest.from_dict(read_json(path)))
            except (StoreError, KeyError):
                continue  # a stray or hand-edited file must not break `ls`
        return sorted(out, key=lambda m: m.created, reverse=True)

    def exists(self, ref: str) -> bool:
        try:
            name, tag = parse_ref(ref)
        except RefError:
            return False
        return self.manifest_path(name, tag).is_file()

    def delete(self, ref: str) -> None:
        name, tag = parse_ref(ref)
        self.manifest_path(name, tag).unlink(missing_ok=True)
        self.recipe_path(name, tag).unlink(missing_ok=True)
        self.gc(dry_run=False)

    # --- recipes (provenance) ------------------------------------------
    def recipe_path(self, name: str, tag: str) -> Path:
        return self.recipes.joinpath(*name.split("/")) / f"{tag}.json"

    def save_recipe(self, manifest: Manifest, steps: Iterable[dict[str, Any]]) -> None:
        """Record how a model was produced, so it can be rebuilt or explained."""
        atomic_write_json(
            self.recipe_path(manifest.name, manifest.tag),
            {
                "ref": manifest.ref,
                "created": manifest.created,
                "parent": manifest.parent,
                "steps": list(steps),
            },
        )

    def load_recipe(self, ref: str) -> dict[str, Any] | None:
        name, tag = parse_ref(ref)
        path = self.recipe_path(name, tag)
        return read_json(path) if path.is_file() else None

    # --- garbage collection --------------------------------------------
    def referenced_digests(self) -> set[str]:
        used: set[str] = set()
        for manifest in self.list():
            used.update(layer.digest for layer in manifest.layers)
        return used

    def gc(self, dry_run: bool = True) -> list[str]:
        """Delete blobs no manifest references. Returns the digests removed."""
        used = self.referenced_digests()
        removed: list[str] = []
        if not self.blobs.exists():
            return removed
        for path in self.blobs.iterdir():
            digest = "sha256:" + path.name.removeprefix("sha256-")
            if path.is_file() and digest not in used:
                removed.append(digest)
                if not dry_run:
                    path.unlink(missing_ok=True)
        return removed

    def stats(self) -> dict[str, int]:
        manifests = self.list()
        return {
            "models": len(manifests),
            "blobs": len(list(self.blobs.iterdir())) if self.blobs.exists() else 0,
            "bytes": sum(m.size for m in manifests),
        }
