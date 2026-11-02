#!/usr/bin/env python3
"""The bare minimum: run a language model with no llama.cpp, and serve it.

This is the reference implementation for the theory doc in vkdev01/shilp-docs.
It is deliberately
naive — PyTorch, a tokenizer, a sampling loop, and FastAPI — because its job is
to make the moving parts visible, not to be fast. Every shortcut it takes is
named in the doc, with what it costs.

Two modes, both documented by the argument parser:

  * generate — one prompt, printed to stdout, with token counts on stderr
  * serve    — the same model behind an OpenAI-shaped endpoint on /v1

Why this file exists: `shilp serve` and a static file server are different kinds
of thing, and the gap between them is the whole subject. Read the four functions
below in order — tokenize, forward, sample, stream — and you have the complete
anatomy of any inference server, including the fast ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from typing import Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


# ---------------------------------------------------------------- 1. load
def load(model_id: str, dtype: str = "float32"):
    """Download (once) and materialise the weights.

    `dtype` is the single biggest lever on memory: float32 is 4 bytes per
    parameter, float16 and bfloat16 are 2. For a 1.5B model that is the
    difference between ~6 GB and ~3 GB of RAM.
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.eval()  # disables dropout; without it, sampling is non-deterministic

    return tokenizer, model, time.perf_counter() - started


# ------------------------------------------------------------ 2. tokenize
def build_prompt(tokenizer, messages: list[dict]) -> str:
    """Turn chat messages into the exact string the model was trained on.

    If the tokenizer ships a chat template, use it — it encodes special tokens
    the model depends on. If it does not, you are on your own, and the model
    will behave noticeably worse. This is the single most common reason a local
    model "feels dumb".
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback: a hand-rolled format. Works, but only by luck.
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    user = " ".join(m["content"] for m in messages if m["role"] == "user")
    return (system + "\n" if system else "") + f"Q: {user}\nA:"


# -------------------------------------------------------------- 3. sample
def stream_tokens(
    tokenizer,
    model,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> Iterator[str]:
    """Generate one token at a time and yield decoded text as it arrives.

    `model.generate` handles the KV cache internally: the first call processes
    the whole prompt (prefill), each later call processes one token and reads
    the cache. That asymmetry is why time-to-first-token and tokens-per-second
    are different numbers.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    # A background thread drives generate(); the streamer hands chunks back to
    # this generator. Without this, generate() would block for the whole answer
    # and there would be nothing to stream.
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    thread = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
    thread.start()
    for chunk in streamer:
        if chunk:
            yield chunk
    thread.join()


def generate_once(tokenizer, model, prompt: str, **kwargs) -> tuple[str, dict]:
    """Non-streaming variant, with the timings a server needs to report."""
    inputs = tokenizer(prompt, return_tensors="pt")
    n_prompt = inputs["input_ids"].shape[1]

    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            do_sample=kwargs.get("temperature", 0.0) > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    total = time.perf_counter() - started

    new_ids = output[0][n_prompt:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, {
        "prompt_tokens": n_prompt,
        "completion_tokens": len(new_ids),
        "total_ms": round(total * 1000, 1),
        "tok_per_s": round(len(new_ids) / total, 1) if total else 0.0,
    }


# -------------------------------------------------------------- 4. serve
def serve(model_id: str, host: str, port: int, dtype: str) -> None:
    """A minimal OpenAI-shaped endpoint over the loop above.

    ~40 lines of FastAPI. Everything else in a production server — queueing,
    eviction, metrics, both API dialects — is engineering on top of this, not a
    different mechanism.
    """
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    tokenizer, model, load_s = load(model_id, dtype)
    print(f"loaded {model_id} in {load_s:.1f}s", file=sys.stderr)

    app = FastAPI()

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = ""
        messages: list[Message]
        stream: bool = False
        temperature: float = 0.0
        max_tokens: int = 128

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        prompt = build_prompt(tokenizer, [m.model_dump() for m in req.messages])
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if not req.stream:
            text, stats = generate_once(
                tokenizer, model, prompt,
                max_new_tokens=req.max_tokens, temperature=req.temperature,
            )
            return {
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": stats["prompt_tokens"],
                          "completion_tokens": stats["completion_tokens"],
                          "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]},
            }

        def events():
            for chunk in stream_tokens(
                tokenizer, model, prompt,
                max_new_tokens=req.max_tokens, temperature=req.temperature,
            ):
                payload = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": chunk},
                                 "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------- CLI
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="one prompt, printed to stdout")
    gen.add_argument("--model", default="sshleifer/tiny-gpt2")
    gen.add_argument("--prompt", default="The capital of France is")
    gen.add_argument("--max-tokens", type=int, default=32)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--dtype", default="float32")

    srv = sub.add_parser("serve", help="the same model behind HTTP")
    srv.add_argument("--model", default="sshleifer/tiny-gpt2")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8001)
    srv.add_argument("--dtype", default="float32")

    args = parser.parse_args()

    if args.cmd == "serve":
        serve(args.model, args.host, args.port, args.dtype)
        return 0

    tokenizer, model, load_s = load(args.model, args.dtype)
    params = sum(p.numel() for p in model.parameters())
    weight_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    print(f"load {load_s:.2f}s | {params:,} params | {weight_mb:.1f} MiB weights",
          file=sys.stderr)

    prompt = build_prompt(tokenizer, [{"role": "user", "content": args.prompt}])
    text, stats = generate_once(
        tokenizer, model, prompt,
        max_new_tokens=args.max_tokens, temperature=args.temperature,
    )
    print(text.strip())
    print(f"\n{stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
