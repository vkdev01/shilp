"""One generation path, shared by both API surfaces.

Both `/api/chat` and `/v1/chat/completions` do exactly the same four things:
acquire the model, stream chunks, measure the request, record it. Doing that in
one place is what keeps the two surfaces honest with each other — the numbers
you see in `/api/usage` are the numbers both of them produced.
"""

from __future__ import annotations

from typing import AsyncIterator

from ..engine.base import ChatMessage, Chunk, GenParams
from .state import AppState
from .stream import to_async
from .telemetry import RequestRecord


def _record(state: AppState, rec: RequestRecord) -> None:
    """Persist the request and update the Prometheus counters."""
    try:
        state.telemetry.record(rec)
    except Exception:
        pass  # telemetry must never fail a response
    state.metrics.inc(
        "shilp_requests_total",
        {"surface": rec.surface, "endpoint": rec.endpoint, "model": rec.model},
    )
    state.metrics.inc(
        "shilp_tokens_total",
        {"model": rec.model, "direction": "prompt"},
        rec.prompt_tokens,
    )
    state.metrics.inc(
        "shilp_tokens_total",
        {"model": rec.model, "direction": "completion"},
        rec.completion_tokens,
    )
    state.metrics.observe("shilp_request_duration_ms", rec.total_ms, {"model": rec.model})
    if rec.ttft_ms:
        state.metrics.observe("shilp_time_to_first_token_ms", rec.ttft_ms, {"model": rec.model})
    state.metrics.gauge("shilp_models_resident", len(state.loader.loaded()))


def _track(rec: RequestRecord, chunk: Chunk) -> Chunk:
    if chunk.text:
        rec.first_token()
    if chunk.done:
        rec.finish(
            prompt_tokens=chunk.prompt_tokens or 0,
            completion_tokens=chunk.completion_tokens or 0,
        )
    return chunk


async def stream_chat(
    state: AppState,
    rec: RequestRecord,
    ref: str,
    messages: list[ChatMessage],
    params: GenParams,
) -> AsyncIterator[Chunk]:
    """Stream a chat answer, recording the request whatever happens."""
    try:
        async with state.loader.acquire(ref) as engine:
            async for chunk in to_async(engine.chat(messages, params)):
                yield _track(rec, chunk)
    except Exception:
        rec.status = 500
        raise
    finally:
        _record(state, rec)


async def stream_generate(
    state: AppState,
    rec: RequestRecord,
    ref: str,
    prompt: str,
    params: GenParams,
) -> AsyncIterator[Chunk]:
    """Stream a raw completion, recording the request whatever happens."""
    try:
        async with state.loader.acquire(ref) as engine:
            async for chunk in to_async(engine.generate(prompt, params)):
                yield _track(rec, chunk)
    except Exception:
        rec.status = 500
        raise
    finally:
        _record(state, rec)


async def collect(source: AsyncIterator[Chunk]) -> tuple[str, Chunk]:
    """Drain a stream into (text, final chunk) for non-streaming responses."""
    parts: list[str] = []
    last = Chunk(done=True, finish_reason="stop")
    async for chunk in source:
        if chunk.text:
            parts.append(chunk.text)
        if chunk.done:
            last = chunk
    return "".join(parts), last


async def embed(state: AppState, ref: str, texts: list[str]) -> list[list[float]]:
    """Embeddings are not streamed; the engine call is short and bounded."""
    import asyncio

    async with state.loader.acquire(ref) as engine:
        return await asyncio.to_thread(engine.embed, texts)


def build_params(manifest_defaults: dict, options: dict | None) -> GenParams:
    return GenParams.merge(manifest_defaults, options)
