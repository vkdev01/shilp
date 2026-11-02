"""Engines: turn a stored model into a token stream."""

from .base import ChatMessage, Chunk, Engine, EngineError, GenParams
from .fake import FakeEngine

__all__ = ["Engine", "EngineError", "ChatMessage", "Chunk", "GenParams",
           "FakeEngine", "load_engine", "ENGINES"]


#: Every engine the loader can build, and what each one needs on disk.
ENGINES = {
    "llamacpp": "a GGUF file (quantised, mmap'd) — the default",
    "transformers": "a Hugging Face safetensors directory (dense, no quantisation)",
    "fake": "nothing; a deterministic stub for tests and UI work",
}


def load_engine(manifest, weights_path, *, num_ctx: int = 4096, threads: int = 0,
                gpu_layers: int = -1, engine: str | None = None):
    """Build an engine for a manifest.

    Kept as a function so the loader can swap implementations without the server
    knowing which one it got. `weights_path` is a GGUF file for llama.cpp and a
    model directory for transformers — the caller resolves which, because only
    the loader knows how the model is stored.
    """
    name = engine or manifest.engine or "llamacpp"
    if name == "fake":
        return FakeEngine(manifest)
    if name == "llamacpp":
        from .llamacpp import LlamaCppEngine

        return LlamaCppEngine(
            manifest, weights_path, num_ctx=num_ctx, threads=threads, gpu_layers=gpu_layers
        )
    if name == "transformers":
        from .transformers_engine import TransformersEngine

        return TransformersEngine(manifest, weights_path, num_ctx=num_ctx)
    raise ValueError(
        f"unknown engine {name!r} (known: {', '.join(sorted(ENGINES))})"
    )
