"""The OpenAI-compatible surface.

`/v1/chat/completions` is the widest-supported shape in the ecosystem, so almost
every client — the `openai` Python and JS SDKs, LangChain, LlamaIndex, Vercel AI
SDK, Continue, Open WebUI — can point at shilp by setting a base URL.

Differences from the real thing, stated rather than hidden: there is no
`n > 1`, no tool-call passthrough, and `logprobs` is unsupported. `usage` is
always present, including in the final streaming chunk.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..engine.base import ChatMessage
from ..utils import parse_ref
from .generate import build_params, collect, embed, stream_chat, stream_generate
from .schemas import (
    OAIChatRequest,
    OAIChatResponse,
    OAIChoice,
    OAICompletionRequest,
    OAIEmbedRequest,
    OAIEmbedResponse,
    OAIMessage,
    OAIModel,
    OAIModelList,
    OAIUsage,
)
from .state import AppState
from .telemetry import RequestRecord

router = APIRouter(prefix="/v1", tags=["openai"])


def _state(request: Request) -> AppState:
    return request.app.state.shilp


def _created(manifest_created: str) -> int:
    """OpenAI timestamps are unix seconds; manifests store ISO-8601."""
    try:
        return int(time.mktime(time.strptime(manifest_created, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return int(time.time())


def _sse(payload: dict[str, Any] | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


@router.get("/models", response_model=OAIModelList)
def list_models(request: Request) -> OAIModelList:
    """Every stored model, in OpenAI's shape."""
    state = _state(request)
    return OAIModelList(
        data=[
            OAIModel(id=m.ref, created=_created(m.created), owned_by="shilp")
            for m in state.store.list()
        ]
    )


@router.get("/models/{model_id:path}", response_model=OAIModel)
def get_model(model_id: str, request: Request) -> OAIModel:
    """One model. `model_id` may contain a slash, so the path converter is greedy."""
    state = _state(request)
    manifest = state.manifest(model_id)
    return OAIModel(id=manifest.ref, created=_created(manifest.created), owned_by="shilp")


@router.post("/chat/completions")
async def chat_completions(body: OAIChatRequest, request: Request):
    """Chat completion, SSE-streamed when `stream: true`."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    manifest = state.manifest(ref)
    params = build_params(manifest.params, body.options())
    messages = [
        ChatMessage(role=m.role, content=m.content or "")
        for m in body.messages
        if m.role != "tool"
    ]
    rec = RequestRecord(
        surface="openai", endpoint="/v1/chat/completions", model=ref, stream=body.stream
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not body.stream:
        text, final = await collect(stream_chat(state, rec, ref, messages, params))
        return OAIChatResponse(
            id=completion_id,
            created=created,
            model=ref,
            choices=[
                OAIChoice(
                    index=0,
                    message=OAIMessage(role="assistant", content=text),
                    finish_reason=final.finish_reason or "stop",
                )
            ],
            usage=OAIUsage(
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                total_tokens=rec.prompt_tokens + rec.completion_tokens,
            ),
        )

    async def events() -> AsyncIterator[str]:
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": ref,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        async for chunk in stream_chat(state, rec, ref, messages, params):
            if chunk.text:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": ref,
                        "choices": [
                            {"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}
                        ],
                    }
                )
            if chunk.done:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": ref,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": chunk.finish_reason or "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": rec.prompt_tokens,
                            "completion_tokens": rec.completion_tokens,
                            "total_tokens": rec.prompt_tokens + rec.completion_tokens,
                        },
                    }
                )
        yield _sse("[DONE]")

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/completions")
async def completions(body: OAICompletionRequest, request: Request):
    """Legacy text completion, kept because some clients still call it."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    manifest = state.manifest(ref)
    stop = body.stop
    params = build_params(
        manifest.params,
        {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "num_predict": body.max_tokens,
            "stop": [stop] if isinstance(stop, str) else stop,
        },
    )
    rec = RequestRecord(
        surface="openai", endpoint="/v1/completions", model=ref, stream=body.stream
    )
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not body.stream:
        text, final = await collect(stream_generate(state, rec, ref, body.prompt, params))
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": ref,
            "choices": [
                {"index": 0, "text": text, "finish_reason": final.finish_reason or "stop"}
            ],
            "usage": {
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "total_tokens": rec.prompt_tokens + rec.completion_tokens,
            },
        }

    async def events() -> AsyncIterator[str]:
        async for chunk in stream_generate(state, rec, ref, body.prompt, params):
            if chunk.text:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": ref,
                        "choices": [{"index": 0, "text": chunk.text, "finish_reason": None}],
                    }
                )
            if chunk.done:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": ref,
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "finish_reason": chunk.finish_reason or "stop",
                            }
                        ],
                    }
                )
        yield _sse("[DONE]")

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/embeddings", response_model=OAIEmbedResponse)
async def embeddings(body: OAIEmbedRequest, request: Request) -> OAIEmbedResponse:
    """Embeddings in OpenAI's shape."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    texts = [body.input] if isinstance(body.input, str) else body.input
    vectors = await embed(state, ref, texts)
    return OAIEmbedResponse(
        data=[
            {"object": "embedding", "index": i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ],
        model=ref,
        usage=OAIUsage(prompt_tokens=sum(len(t.split()) for t in texts)),
    )
