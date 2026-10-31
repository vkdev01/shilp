"""Engines: turn a stored model into a token stream."""

from .base import ChatMessage, Chunk, Engine, EngineError, GenParams
from .fake import FakeEngine

__all__ = ["Engine", "EngineError", "ChatMessage", "Chunk", "GenParams", "FakeEngine", "load_engine"]


def load_engine(manifest, gguf_path, *, num_ctx: int = 4096, threads: int = 0, engine: str | None = None):
    """Build an engine for a manifest.

    Kept as a function so the loader can swap implementations without the server
    knowing which one it got.
    """
    name = engine or manifest.engine or "llamacpp"
    if name == "fake":
        return FakeEngine(manifest)
    if name == "llamacpp":
        from .llamacpp import LlamaCppEngine

        return LlamaCppEngine(manifest, gguf_path, num_ctx=num_ctx, threads=threads)
    raise ValueError(f"unknown engine {name!r} (known: fake, llamacpp)")
