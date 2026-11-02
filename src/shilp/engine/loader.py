"""Keep models in memory, and decide which one gets evicted.

The policy is deliberately simple, because the constraint is real: a laptop
cannot hold two 7B models. One resident model, LRU eviction, and a reference
count so a model is never closed while a request is still streaming from it.

`acquire` is an async context manager. Loading runs in a worker thread — the
mmap and the KV-cache allocation are blocking — and the lock means two requests
for the same cold model trigger one load, not two.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable

from ..config import Settings
from ..store import MT_GGUF, Manifest, Store
from ..utils import ShilpError
from .base import Engine
from .fake import FakeEngine


class ModelNotReady(ShilpError):
    """The model is not loaded and every slot is busy with other requests."""


@dataclass
class LoadedModel:
    manifest: Manifest
    engine: Engine
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    #: Requests currently streaming. A model with refs > 0 is never evicted.
    refs: int = 0

    def snapshot(self) -> dict:
        return {
            "ref": self.manifest.ref,
            "size": self.manifest.size,
            "refs": self.refs,
            "idle_seconds": round(time.monotonic() - self.last_used, 2),
            "loaded_seconds": round(time.monotonic() - self.loaded_at, 2),
        }


EngineFactory = Callable[[Manifest, Path], Engine]


def _default_factory(settings: Settings) -> EngineFactory:
    def build(manifest: Manifest, gguf_path: Path) -> Engine:
        name = manifest.engine or settings.engine
        if name == "fake":
            return FakeEngine(manifest)
        from .llamacpp import LlamaCppEngine

        return LlamaCppEngine(
            manifest,
            gguf_path,
            num_ctx=int(manifest.params.get("num_ctx", 4096)),
            threads=settings.threads,
        )

    return build


class ModelLoader:
    """Owns the resident engines and their lifecycle."""

    def __init__(
        self,
        store: Store,
        settings: Settings,
        *,
        factory: EngineFactory | None = None,
    ):
        self.store = store
        self.settings = settings
        self._factory = factory or _default_factory(settings)
        self._resident: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()

    # --- introspection -------------------------------------------------
    def loaded(self) -> list[dict]:
        return [rec.snapshot() for rec in self._resident.values()]

    def is_loaded(self, ref: str) -> bool:
        return ref in self._resident

    # --- loading -------------------------------------------------------
    def _build(self, manifest: Manifest) -> LoadedModel:
        """Blocking part: find the weights in the shape this engine needs.

        The two engines want different things on disk — llama.cpp a single GGUF
        blob, transformers a directory of safetensors — so the loader resolves
        that and the engines never touch the store.
        """
        name = manifest.engine or self.settings.engine
        if name == "fake":
            return LoadedModel(manifest=manifest, engine=FakeEngine(manifest))

        if name == "transformers":
            engine: Engine = self._factory(manifest, self._model_dir(manifest))
            return LoadedModel(manifest=manifest, engine=engine)

        layer = manifest.layer_of(MT_GGUF)
        if layer is None:
            raise ModelNotReady(
                f"{manifest.ref} has no GGUF weights. Convert and quantize first, "
                "or pull a GGUF repo: shilp pull hf:org/model-GGUF:q4_k_m"
            )
        engine = self._factory(manifest, self.store.blob_path(layer.digest))
        return LoadedModel(manifest=manifest, engine=engine)

    def _model_dir(self, manifest: Manifest) -> Path:
        """Materialise a manifest's layers into a directory `transformers` can read.

        Cached under $SHILP_HOME/cache so a second load is a no-op; a manifest
        whose layers change gets a new directory name, so the cache cannot go
        stale in place.
        """
        from ..export import materialise

        digest = "-".join(layer.digest.split(":")[1][:8] for layer in manifest.layers[:1])
        dest = self.store.home / "cache" / f"{manifest.name}-{manifest.tag}-{digest}"
        if not (dest / "config.json").is_file():
            if dest.exists():
                shutil.rmtree(dest)
            materialise(self.store, manifest, dest)
        return dest

    async def _evict_one(self) -> bool:
        """Close the least recently used idle model. False if none is idle."""
        candidates = [r for r in self._resident.values() if r.refs == 0]
        if not candidates:
            return False
        victim = min(candidates, key=lambda r: r.last_used)
        del self._resident[victim.manifest.ref]
        await asyncio.to_thread(victim.engine.close)
        return True

    async def _ensure(self, ref: str) -> LoadedModel:
        rec = self._resident.get(ref)
        if rec is not None:
            return rec

        manifest = self.store.resolve(ref)
        while len(self._resident) >= max(1, self.settings.max_resident):
            if not await self._evict_one():
                busy = ", ".join(r.manifest.ref for r in self._resident.values())
                raise ModelNotReady(
                    f"cannot load {ref}: {busy} is still serving requests and "
                    f"SHILP_MAX_RESIDENT={self.settings.max_resident}"
                )
        rec = await asyncio.to_thread(self._build, manifest)
        self._resident[ref] = rec
        return rec

    @asynccontextmanager
    async def acquire(self, ref: str) -> AsyncIterator[Engine]:
        """Yield an engine for `ref`, holding it resident for the duration."""
        async with self._lock:
            rec = await self._ensure(ref)
            rec.refs += 1
            rec.last_used = time.monotonic()
        try:
            yield rec.engine
        finally:
            async with self._lock:
                rec.refs = max(0, rec.refs - 1)
                rec.last_used = time.monotonic()

    # --- teardown ------------------------------------------------------
    async def unload(self, ref: str | None = None) -> list[str]:
        """Close one model, or all of them. Returns the refs unloaded."""
        async with self._lock:
            targets = [ref] if ref else list(self._resident)
            closed: list[str] = []
            for target in targets:
                rec = self._resident.pop(target, None)
                if rec is None:
                    continue
                await asyncio.to_thread(rec.engine.close)
                closed.append(target)
            return closed
