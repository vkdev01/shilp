"""The engine contract: everything the server needs from an inference backend.

The server never imports llama.cpp. It talks to this protocol, which is why a
`fake` engine can drive the whole HTTP surface in tests, and why adding a
transformers or vLLM backend is one file rather than a refactor.

Streaming is the primitive. A non-streaming response is just a stream we
concatenate, so there is one code path to get right instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenParams:
    """Sampling and limits. Values come from the manifest, then the request."""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.0
    repeat_penalty: float = 1.1
    num_predict: int = 512
    num_ctx: int = 4096
    seed: int = -1
    stop: tuple[str, ...] = ()

    @classmethod
    def merge(cls, base: dict[str, Any], override: dict[str, Any] | None) -> "GenParams":
        """Manifest defaults overlaid with request values; None means "not set"."""
        merged = {**base, **{k: v for k, v in (override or {}).items() if v is not None}}
        stop = merged.get("stop") or ()
        if isinstance(stop, str):
            stop = (stop,)
        known = {f for f in cls.__dataclass_fields__ if f != "stop"}
        kwargs = {k: merged[k] for k in known if k in merged}
        return cls(stop=tuple(stop), **kwargs)


@dataclass
class Chunk:
    """One piece of a streamed answer.

    `done=True` arrives exactly once, last, and carries the token counts. Both
    API surfaces depend on that: Ollama puts them in its final JSON object and
    OpenAI puts them in a trailing chunk with empty choices.
    """

    text: str = ""
    done: bool = False
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Engine(Protocol):
    """What a backend must provide."""

    #: The `name:tag` this engine was loaded from.
    ref: str

    def chat(self, messages: list[ChatMessage], params: GenParams) -> Iterator[Chunk]:
        """Stream an answer to a conversation."""
        ...

    def generate(self, prompt: str, params: GenParams) -> Iterator[Chunk]:
        """Stream a raw completion, no chat formatting."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings, for `/api/embed` and `/v1/embeddings`."""
        ...

    def info(self) -> dict[str, Any]:
        """Facts for `/api/show` and `/api/ps`: context size, template, counts."""
        ...

    def close(self) -> None:
        """Release memory. Called on eviction and at shutdown."""
        ...


class EngineError(RuntimeError):
    """The backend failed while generating; surfaced to the client as a 500."""


class ContextOverflow(EngineError, ValueError):
    """The prompt does not fit in the model's context window.

    A `ValueError` as well, so the server maps it to a 400: the request is wrong,
    not the server. It is raised before any token is generated, which is what
    lets a streaming response fail with a status code instead of a cut stream.
    """
