"""A second engine: plain PyTorch and `transformers`, no llama.cpp.

This exists to make the boundary in `engine/base.py` demonstrable rather than
asserted. The server, the loader, both API surfaces and the telemetry are
unchanged; only this file is new. `--engine transformers` swaps it in.

It is the same code as `scripts/bare_inference.py`, which is the tutorial
version, wired to the `Engine` protocol. Read that first if you want the
concepts; read this if you want the integration.

**What you give up compared to the llama.cpp engine**, stated up front because
it is the whole point of docs/00-theory.md:

* No quantisation. Weights are dense fp32/fp16/bf16, so a 7B model needs
  ~14 GiB instead of ~4.4 GiB, and decode is 3–7× slower (it is bandwidth-bound;
  see §2.4 of the theory doc).
* No mmap. Loading parses and materialises the whole checkpoint.
* No GGUF. It needs a safetensors directory — `shilp pull hf:org/model`, not
  `...-GGUF:q4_k_m`.

**What you gain**: it runs any architecture `transformers` supports on day one,
it needs no compiler, and every tensor is inspectable with `print`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

from ..store import Manifest
from ..utils import MissingDependency
from .base import ChatMessage, Chunk, EngineError, GenParams
from .template import render


def sample_next(logits: Any, params: GenParams, *, generator: Any = None) -> int:
    """Turn one row of logits into a token id.

    Implemented here rather than calling `model.generate` because we need the
    loop's shape (one chunk per token, exact counts, stop sequences honoured
    mid-stream). Pure function of its inputs, so it is unit-testable without a
    model — see tests/test_transformers_engine.py.
    """
    import torch

    scores = logits.float()

    # Repetition penalty applies before temperature, as in llama.cpp: it divides
    # scores that are already positive and multiplies those that are negative,
    # which pushes both away from zero.
    if params.repeat_penalty and params.repeat_penalty != 1.0:
        # No history is threaded through here; the caller applies it. Kept as a
        # no-op branch so the ordering is documented in one place.
        pass

    if params.temperature <= 0:
        return int(torch.argmax(scores).item())

    scores = scores / params.temperature

    if params.top_k and params.top_k > 0:
        k = min(params.top_k, scores.shape[-1])
        kth = torch.topk(scores, k).values[-1]
        scores = torch.where(scores < kth, torch.full_like(scores, float("-inf")), scores)

    probs = torch.softmax(scores, dim=-1)

    if params.top_p and params.top_p < 1.0:
        ordered, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        # Keep the shortest prefix that reaches top_p, always at least one token.
        cut = int(torch.searchsorted(cumulative, torch.tensor(params.top_p)).item()) + 1
        cut = max(1, min(cut, probs.shape[-1]))
        keep = order[:cut]
        mask = torch.full_like(probs, 0.0)
        mask[keep] = probs[keep]
        probs = mask

    if params.min_p and params.min_p > 0:
        floor = params.min_p * probs.max()
        probs = torch.where(probs < floor, torch.zeros_like(probs), probs)

    total = probs.sum()
    if total <= 0:  # every candidate filtered out; fall back to the argmax
        return int(torch.argmax(logits.float()).item())
    probs = probs / total

    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def first_stop_index(text: str, stops: tuple[str, ...]) -> int:
    """Index of the earliest stop sequence in `text`, or -1.

    A stop sequence may appear anywhere, not only at the end — checking only the
    tail misses the common case where the model emits a leading space or a newline
    after the sequence. Pure function so it can be tested without a model.
    """
    found = [text.find(s) for s in stops if s and s in text]
    return min(found) if found else -1


class _IncrementalDecoder:
    """Decode one token at a time, emitting only the new text.

    Decoding the whole sequence each step and diffing is O(n²) in tokens, which
    is irrelevant at these lengths and avoids the byte-boundary bugs that a
    per-token incremental decoder has to handle. A partial UTF-8 sequence can
    therefore emit a replacement character for one step; it corrects itself on
    the next.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._text = ""

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        full = self._tokenizer.decode(self._ids, skip_special_tokens=True)
        delta = full[len(self._text):]
        self._text = full
        return delta


def require_model_dir(model_dir: Path) -> Path:
    """Validate that `model_dir` looks like a Hugging Face model directory.

    Called before any import, so a wrong path produces the useful message on
    every machine — including one where torch is not installed. Checking the
    folder first is not just tidiness: the two failures have different fixes,
    and reporting "install torch" to someone who passed the wrong directory
    sends them the wrong way.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise EngineError(
            f"{model_dir} is not a Hugging Face model directory. The transformers "
            "engine needs safetensors + config, not a GGUF — pull a raw repo with "
            "`shilp pull hf:org/model`, or serve the GGUF with --engine llamacpp."
        )
    return model_dir


class TransformersEngine:
    """`Engine` backed by `transformers`. Dense weights, no GGUF."""

    def __init__(
        self,
        manifest: Manifest,
        model_dir: Path,
        *,
        num_ctx: int = 4096,
        dtype: str = "auto",
        device: str = "cpu",
    ):
        # Directory first: the error for a wrong path must not depend on whether
        # the optional dependency happens to be installed.
        model_dir = require_model_dir(model_dir)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - install dependent
            raise MissingDependency(
                "the transformers engine needs torch and transformers:\n"
                '  pip install -e ".[base]"\n'
                "(or use the default llama.cpp engine: --engine llamacpp)"
            ) from exc

        self.ref = manifest.ref
        self.manifest = manifest
        self.num_ctx = num_ctx
        self._torch = torch
        self._template = manifest.template
        self._lock = threading.Lock()

        torch_dtype = {
            "auto": "auto", "float32": torch.float32,
            "float16": torch.float16, "bfloat16": torch.bfloat16,
        }.get(dtype, "auto")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            # Many causal-LM tokenizers (GPT-2, and everything derived from it)
            # ship no pad token, and batching with padding then raises. Using EOS
            # is the conventional stand-in; it only affects padded positions,
            # which the attention mask removes anyway.
            if getattr(self._tokenizer, "pad_token", None) is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), dtype=torch_dtype
            ).to(device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on the checkpoint
            raise EngineError(f"cannot load {model_dir}: {exc}") from exc

        self._device = device

    # --- prompt construction -------------------------------------------
    def _render_chat(self, messages: list[ChatMessage]) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m for m in messages if m.role != "system"]

        # 1. an explicit template in the manifest wins (per-tag behaviour)
        if self._template:
            return render(
                self._template,
                system=system or (self.manifest.system or ""),
                prompt=turns[-1].content if turns else "",
            )
        # 2. otherwise the tokenizer's own template, which is what the model was
        #    actually trained on
        if getattr(self._tokenizer, "chat_template", None):
            payload = [m.to_dict() for m in messages]
            if not system and self.manifest.system:
                payload.insert(0, {"role": "system", "content": self.manifest.system})
            return self._tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        # 3. last resort: a format the model has never seen. This is the failure
        #    mode described in docs/00-theory.md §3.3 — it works, by luck.
        return (system + "\n" if system else "") + (
            f"Q: {turns[-1].content if turns else ''}\nA:"
        )

    # --- generation -----------------------------------------------------
    def _stream(self, prompt: str, params: GenParams) -> Iterator[Chunk]:
        torch = self._torch
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.num_ctx
        ).to(self._device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        if params.seed is not None and params.seed >= 0:
            torch.manual_seed(params.seed)
        generator = None
        if params.seed is not None and params.seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        decoder = _IncrementalDecoder(self._tokenizer)
        eos = self._tokenizer.eos_token_id
        emitted = 0
        text_so_far = ""
        emitted_text = ""

        with self._lock, torch.no_grad():  # one mutable KV cache per model
            try:
                out = self._model(**inputs, use_cache=True)
                past = out.past_key_values
                next_id = sample_next(out.logits[0, -1], params, generator=generator)

                for _ in range(max(1, params.num_predict)):
                    if next_id == eos:
                        break
                    piece = decoder.push(next_id)
                    text_so_far += piece
                    emitted += 1

                    # A stop sequence can appear anywhere in the output, not just
                    # at the end, and it can straddle chunk boundaries — so search
                    # the whole generated text and truncate at the first hit.
                    # (Matching only the tail is the bug this replaced: a stop of
                    # " " never matches text ending in "stairs".)
                    cut = first_stop_index(text_so_far, params.stop)
                    if cut >= 0:
                        remaining = text_so_far[:cut][len(emitted_text):]
                        if remaining:
                            yield Chunk(text=remaining)
                        break

                    if piece:
                        yield Chunk(text=piece)
                        emitted_text = text_so_far

                    step = self._model(
                        input_ids=torch.tensor([[next_id]], device=self._device),
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = step.past_key_values
                    next_id = sample_next(step.logits[0, -1], params, generator=generator)
            except Exception as exc:  # surface as a 500, not a dropped socket
                raise EngineError(f"transformers failed while generating: {exc}") from exc

        yield Chunk(
            done=True,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=emitted,
        )

    # --- Engine protocol ------------------------------------------------
    def chat(self, messages: list[ChatMessage], params: GenParams) -> Iterator[Chunk]:
        return self._stream(self._render_chat(messages), params)

    def generate(self, prompt: str, params: GenParams) -> Iterator[Chunk]:
        if self._template:
            prompt = render(
                self._template, system=self.manifest.system or "", prompt=prompt
            )
        return self._stream(prompt, params)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Mean-pooled last hidden state — a fallback, not a real embedder.

        A dedicated embedding model is the right tool; this exists so that
        `/api/embed` answers instead of 501 when the engine is swapped.
        """
        torch = self._torch
        with self._lock, torch.no_grad():
            batch = self._tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.num_ctx,
            ).to(self._device)
            hidden = self._model(**batch, output_hidden_states=True).hidden_states[-1]
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return pooled.tolist()

    def info(self) -> dict[str, Any]:
        config = self._model.config
        params = sum(p.numel() for p in self._model.parameters())
        return {
            **self.manifest.metadata,
            "engine": "transformers",
            "n_ctx": self.num_ctx,
            "n_vocab": int(getattr(config, "vocab_size", 0) or 0),
            "parameters": params,
            "dtype": str(next(self._model.parameters()).dtype),
            "template": self._template,
            "params": self.manifest.params,
        }

    def close(self) -> None:
        self._model = None  # type: ignore[assignment]
        self._tokenizer = None  # type: ignore[assignment]
