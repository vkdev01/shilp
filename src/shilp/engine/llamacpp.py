"""The llama.cpp backend: GGUF on CPU, Metal on Apple Silicon.

`llama-cpp-python` binds the same C++ runtime that `llama-server` uses, so this
is a real inference engine, not a wrapper around a subprocess. Two consequences
worth knowing:

* Calls are blocking. The server runs them in a worker thread (see
  `server/routes_*.py`), never on the event loop.
* One `Llama` instance per model, guarded by a lock, because a GGUF's KV cache
  is mutable state. Requests for the same model therefore run one at a time:
  each streams at full speed, the rest wait their turn.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

from ..store import Manifest
from ..utils import MissingDependency
from .base import ChatMessage, Chunk, ContextOverflow, EngineError, GenParams
from .template import render


class LlamaCppEngine:
    """A loaded GGUF file, wrapped to the `Engine` protocol."""

    def __init__(
        self,
        manifest: Manifest,
        gguf_path: Path,
        *,
        num_ctx: int = 4096,
        threads: int = 0,
        gpu_layers: int = -1,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:  # pragma: no cover - install dependent
            raise MissingDependency(
                "llama-cpp-python is not installed. On Apple Silicon:\n"
                '  CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[engine]"\n'
                "On CPU only:\n"
                '  pip install -e ".[engine]"'
            ) from exc

        if not Path(gguf_path).is_file():
            raise FileNotFoundError(f"weights missing: {gguf_path}")

        self.ref = manifest.ref
        self.manifest = manifest
        self._template = manifest.template
        self._n_ctx = num_ctx
        self._formatter: Any = None
        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=str(gguf_path),
            n_ctx=num_ctx,
            n_threads=threads or None,
            n_gpu_layers=gpu_layers,
            verbose=verbose,
            # A template in the manifest wins over the one baked into the GGUF;
            # that is the whole point of being able to edit it per tag.
            chat_format=None,
        )

    # --- helpers -------------------------------------------------------
    def _sampling(self, params: GenParams) -> dict[str, Any]:
        """Map our parameter names onto llama.cpp's `create_completion` kwargs."""
        return {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "repeat_penalty": params.repeat_penalty,
            "max_tokens": params.num_predict,
            "seed": params.seed if params.seed >= 0 else None,
            "stop": list(params.stop) or None,
        }

    def _count(self, text: str) -> int:
        try:
            return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            return max(1, len(text) // 4)  # rough fallback, never zero

    def _chat_formatter(self) -> Any:
        """The GGUF's own chat template, built the way llama-cpp-python builds it."""
        if self._formatter is None:
            template = (self._llm.metadata or {}).get("tokenizer.chat_template")
            if not template:
                return None
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            def text_of(token_id: int) -> str:
                return self._llm._model.token_get_text(token_id) if token_id != -1 else ""

            self._formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=text_of(self._llm.token_eos()),
                bos_token=text_of(self._llm.token_bos()),
            )
        return self._formatter

    def _prompt_tokens(self, prompt: str | list[dict[str, str]]) -> int:
        """Tokens the model will actually read, chat template included.

        A chat request arrives as a list of messages that llama-cpp-python renders
        internally; counting the list itself (as this used to) always gave 1.
        """
        try:
            if isinstance(prompt, str):
                return len(self._llm.tokenize(prompt.encode("utf-8"), special=True))
            formatter = self._chat_formatter()
            if formatter is not None:
                result = formatter(messages=prompt)
                return len(
                    self._llm.tokenize(
                        result.prompt.encode("utf-8"),
                        add_bos=not result.added_special,
                        special=True,
                    )
                )
        except Exception:
            pass
        if isinstance(prompt, str):
            return self._count(prompt)
        # No template to render: content plus a few framing tokens per message.
        return sum(self._count(m.get("content") or "") + 4 for m in prompt) or 1

    def _stream(self, prompt: Any, params: GenParams, raw_prompt: bool) -> Iterator[Chunk]:
        """Shared streaming loop for both chat and raw completion."""
        prompt_tokens = self._prompt_tokens(prompt)
        if prompt_tokens >= self._n_ctx:
            raise ContextOverflow(
                f"prompt is {prompt_tokens} tokens but the context window is "
                f"{self._n_ctx}; shorten it, or raise num_ctx for this model"
            )
        pieces: list[str] = []
        with self._lock:  # a GGUF has one mutable KV cache
            stream: Any = None
            try:
                if raw_prompt:
                    stream = self._llm.create_completion(
                        prompt=prompt, stream=True, **self._sampling(params)
                    )
                else:
                    stream = self._llm.create_chat_completion(
                        messages=prompt,  # type: ignore[arg-type]
                        stream=True,
                        **self._sampling(params),
                    )
                for event in stream:
                    choice = event["choices"][0]
                    text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                    if text:
                        pieces.append(text)
                        yield Chunk(text=text)
                    if choice.get("finish_reason"):
                        break
            except Exception as exc:  # surface as a 500, not a dropped socket
                raise EngineError(f"llama.cpp failed while generating: {exc}") from exc
            finally:
                # Closing our generator (a cancelled request) lands here: stop
                # llama.cpp now, so the lock is released for the next request
                # instead of after an answer nobody will read.
                if stream is not None and hasattr(stream, "close"):
                    stream.close()

        full = "".join(pieces)
        yield Chunk(
            done=True,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=self._count(full),
        )

    # --- Engine protocol ----------------------------------------------
    def chat(self, messages: list[ChatMessage], params: GenParams) -> Iterator[Chunk]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m for m in messages if m.role != "system"]
        if self._template:
            prompt = render(
                self._template,
                system=system or (self.manifest.system or ""),
                prompt=turns[-1].content if turns else "",
            )
            return self._stream(prompt, params, raw_prompt=True)
        payload = [m.to_dict() for m in messages]
        if not system and self.manifest.system:
            payload.insert(0, {"role": "system", "content": self.manifest.system})
        return self._stream(payload, params, raw_prompt=False)  # type: ignore[arg-type]

    def generate(self, prompt: str, params: GenParams) -> Iterator[Chunk]:
        if self._template:
            prompt = render(
                self._template,
                system=self.manifest.system or "",
                prompt=prompt,
            )
        return self._stream(prompt, params, raw_prompt=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            try:
                out = self._llm.create_embedding(texts)
                return [item["embedding"] for item in out["data"]]
            except Exception as exc:
                raise EngineError(f"embedding failed: {exc}") from exc

    def info(self) -> dict[str, Any]:
        meta = dict(self.manifest.metadata)
        meta.update(
            {
                "engine": "llamacpp",
                "n_ctx": int(self._llm.n_ctx()),
                "n_vocab": int(self._llm.n_vocab()),
                "template": self._template,
                "params": self.manifest.params,
            }
        )
        return meta

    def close(self) -> None:
        # Dropping the reference frees the mmap; llama.cpp has no explicit close.
        self._llm = None  # type: ignore[assignment]
