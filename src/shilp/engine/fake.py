"""A deterministic engine with no model behind it.

This exists so the HTTP surface can be developed, tested and demonstrated
without downloading 4 GB of weights. It streams word by word, reports plausible
token counts, and its answers are a pure function of the prompt — so an
assertion about the API can be exact rather than "contains something".
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Iterator

from ..store import Manifest
from .base import ChatMessage, Chunk, GenParams

WORDS = (
    "the", "model", "answers", "in", "plain", "text", "because", "that", "is",
    "what", "a", "server", "does", "stream", "tokens", "back",
)


class FakeEngine:
    """Echoes a hashed sentence derived from the prompt, one word at a time."""

    def __init__(self, manifest: Manifest, *, delay: float = 0.0):
        self.ref = manifest.ref
        self.manifest = manifest
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def _answer(self, prompt: str, params: GenParams) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        n = max(3, min(params.num_predict, 12))
        picks = [WORDS[int(digest[i * 2 : i * 2 + 2], 16) % len(WORDS)] for i in range(n)]
        return " ".join(picks) + "."

    def _emit(self, prompt: str, params: GenParams) -> Iterator[Chunk]:
        answer = self._answer(prompt, params)
        self.calls.append({"prompt": prompt, "params": params})
        for word in answer.split(" "):
            if self.delay:
                time.sleep(self.delay)
            yield Chunk(text=word + " ")
        yield Chunk(
            done=True,
            finish_reason="stop",
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=len(answer.split()),
        )

    def chat(self, messages: list[ChatMessage], params: GenParams) -> Iterator[Chunk]:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self._emit(prompt, params)

    def generate(self, prompt: str, params: GenParams) -> Iterator[Chunk]:
        return self._emit(prompt, params)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([byte / 255.0 for byte in digest[:8]])
        return out

    def info(self) -> dict[str, Any]:
        return {
            "engine": "fake",
            "n_ctx": self.manifest.params.get("num_ctx", 4096),
            "n_vocab": 32_000,
            "template": self.manifest.template,
            "params": self.manifest.params,
        }

    def close(self) -> None:
        return None
