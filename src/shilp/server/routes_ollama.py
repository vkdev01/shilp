"""The Ollama-compatible surface.

Enough of Ollama's API that existing clients — `ollama` CLIs, Continue, Open WebUI,
LangChain's `Ollama` class — talk to shilp without knowing it is not Ollama.

Faithful where it matters: NDJSON streaming, `done: true` as the last line,
durations in nanoseconds, `model` as `name:tag`, and `{"error": …}` as the last
line when a stream fails midway. Approximated where the real thing has no
equivalent: `digest` is the manifest's own hash rather than an Ollama blob id,
and `load_duration` is not measured yet (always 0; see the roadmap).
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..engine.base import ChatMessage
from ..hub import HfSpec, pull as hub_pull
from ..store import Manifest
from ..utils import human_size, sha256_bytes
from .generate import build_params, collect, embed, prime, stream_chat, stream_generate
from .schemas import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaDeleteRequest,
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaModel,
    OllamaModelDetails,
    OllamaPsResponse,
    OllamaPullRequest,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from .state import AppState
from .telemetry import RequestRecord

router = APIRouter(prefix="/api", tags=["ollama"])

NDJSON = "application/x-ndjson"


def _state(request: Request) -> AppState:
    return request.app.state.shilp


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _digest(manifest: Manifest) -> str:
    """A stable id for the model: the hash of its manifest, not of its weights."""
    return sha256_bytes(json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8"))


def _details(manifest: Manifest) -> OllamaModelDetails:
    meta = manifest.metadata
    return OllamaModelDetails(
        format=str(meta.get("format", "gguf")),
        family=str(meta.get("architecture", "unknown")),
        families=[str(meta.get("architecture", "unknown"))],
        parameter_size=str(meta.get("parameters", "unknown")),
        quantization_level=str(meta.get("quantization", "unknown")),
    )


def _modelfile(manifest: Manifest) -> str:
    """Render the manifest back as a Shilpfile, so `/api/show` round-trips."""
    lines = [f"FROM {manifest.metadata.get('source', manifest.ref)}"]
    for key, value in manifest.params.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            lines.append(f'PARAMETER {key} "{item}"')
    if manifest.system:
        lines.append(f'SYSTEM """{manifest.system}"""')
    if manifest.template:
        lines.append(f'TEMPLATE """{manifest.template}"""')
    return "\n".join(lines) + "\n"


def _ms_to_ns(value: float) -> int:
    return int(value * 1_000_000)


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


# --- model management --------------------------------------------------
@router.get("/tags", response_model=OllamaTagsResponse)
def tags(request: Request) -> OllamaTagsResponse:
    """List stored models."""
    state = _state(request)
    return OllamaTagsResponse(
        models=[
            OllamaModel(
                name=m.ref,
                model=m.ref,
                modified_at=m.created,
                size=m.size,
                digest=_digest(m),
                details=_details(m),
            )
            for m in state.store.list()
        ]
    )


@router.post("/show", response_model=OllamaShowResponse)
def show(body: OllamaShowRequest, request: Request) -> OllamaShowResponse:
    """Everything known about one model: its Shilpfile, template and metadata."""
    state = _state(request)
    manifest = state.manifest(state.resolve_model(body.model))
    return OllamaShowResponse(
        modelfile=_modelfile(manifest),
        parameters="\n".join(f"{k} {v}" for k, v in manifest.params.items()),
        template=manifest.template or "",
        details=_details(manifest),
        model_info={
            **manifest.metadata,
            "size_bytes": manifest.size,
            "size_human": human_size(manifest.size),
            "layers": len(manifest.layers),
        },
        capabilities=["completion", "chat", "embeddings"],
    )


@router.get("/ps", response_model=OllamaPsResponse)
def ps(request: Request) -> OllamaPsResponse:
    """Models currently in memory."""
    return OllamaPsResponse(models=_state(request).loader.loaded())


@router.post("/delete")
def delete(body: OllamaDeleteRequest, request: Request) -> dict[str, Any]:
    """Remove a model and garbage-collect its blobs."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    state.store.delete(ref)
    return {"status": "deleted", "model": ref}


@router.post("/pull")
async def pull(body: OllamaPullRequest, request: Request):
    """Download a model from the Hugging Face Hub and register it.

    Progress is coarse — three phases — because the Hub client reports bytes per
    file and shilp imports into the store afterwards. Coarse and true beats a
    fake percentage.
    """
    state = _state(request)
    spec = HfSpec.parse(body.model if body.model.startswith("hf:") else f"hf:{body.model}")
    hub = state.hub

    async def events() -> AsyncIterator[str]:
        yield _line({"status": "pulling manifest", "model": spec.repo_id})
        try:
            manifest = hub_pull(state.store, spec, hub=hub, token=state.settings.hf_token)
        except Exception as exc:
            yield _line({"status": "error", "error": str(exc)})
            return
        yield _line({"status": "importing layers", "model": manifest.ref})
        yield _line({"status": "success", "model": manifest.ref, "digest": _digest(manifest)})

    if not body.stream:
        manifest = hub_pull(state.store, spec, hub=hub, token=state.settings.hf_token)
        return {"status": "success", "model": manifest.ref}
    return StreamingResponse(events(), media_type=NDJSON)


# --- generation --------------------------------------------------------
@router.post("/chat")
async def chat(body: OllamaChatRequest, request: Request):
    """Conversational completion, streamed as NDJSON unless `stream: false`."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    manifest = state.manifest(ref)
    params = build_params(manifest.params, body.options)
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    rec = RequestRecord(
        surface="ollama", endpoint="/api/chat", model=ref, stream=body.stream
    )

    if not body.stream:
        text, final = await collect(stream_chat(state, rec, ref, messages, params))
        return OllamaChatResponse(
            model=ref,
            created_at=_now(),
            message=OllamaMessage(role="assistant", content=text),
            done=True,
            done_reason=final.finish_reason or "stop",
            total_duration=_ms_to_ns(rec.total_ms),
            prompt_eval_count=rec.prompt_tokens,
            eval_count=rec.completion_tokens,
            eval_duration=_ms_to_ns(max(0.0, rec.total_ms - rec.ttft_ms)),
        )

    # Errors before the first token become a status code, not a cut stream.
    stream = await prime(stream_chat(state, rec, ref, messages, params))

    async def lines() -> AsyncIterator[str]:
        try:
            async for chunk in stream:
                if chunk.text:
                    yield _line(
                        {
                            "model": ref,
                            "created_at": _now(),
                            "message": {"role": "assistant", "content": chunk.text},
                            "done": False,
                        }
                    )
                if chunk.done:
                    yield _line(
                        {
                            "model": ref,
                            "created_at": _now(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": chunk.finish_reason or "stop",
                            "total_duration": _ms_to_ns(rec.total_ms),
                            "prompt_eval_count": rec.prompt_tokens,
                            "eval_count": rec.completion_tokens,
                            "eval_duration": _ms_to_ns(max(0.0, rec.total_ms - rec.ttft_ms)),
                        }
                    )
        except Exception as exc:
            yield _line({"error": str(exc)})  # what Ollama itself sends mid-stream

    return StreamingResponse(lines(), media_type=NDJSON)


@router.post("/generate")
async def generate(body: OllamaGenerateRequest, request: Request):
    """Raw completion against a prompt, with no chat formatting."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    manifest = state.manifest(ref)
    params = build_params(manifest.params, body.options)
    prompt = body.prompt
    if body.system:
        prompt = f"{body.system}\n\n{prompt}"
    rec = RequestRecord(
        surface="ollama", endpoint="/api/generate", model=ref, stream=body.stream
    )

    if not body.stream:
        text, final = await collect(stream_generate(state, rec, ref, prompt, params))
        return OllamaGenerateResponse(
            model=ref,
            created_at=_now(),
            response=text,
            done=True,
            done_reason=final.finish_reason or "stop",
            total_duration=_ms_to_ns(rec.total_ms),
            prompt_eval_count=rec.prompt_tokens,
            eval_count=rec.completion_tokens,
        )

    stream = await prime(stream_generate(state, rec, ref, prompt, params))

    async def lines() -> AsyncIterator[str]:
        try:
            async for chunk in stream:
                if chunk.text:
                    yield _line(
                        {
                            "model": ref,
                            "created_at": _now(),
                            "response": chunk.text,
                            "done": False,
                        }
                    )
                if chunk.done:
                    yield _line(
                        {
                            "model": ref,
                            "created_at": _now(),
                            "response": "",
                            "done": True,
                            "done_reason": chunk.finish_reason or "stop",
                            "total_duration": _ms_to_ns(rec.total_ms),
                            "prompt_eval_count": rec.prompt_tokens,
                            "eval_count": rec.completion_tokens,
                        }
                    )
        except Exception as exc:
            yield _line({"error": str(exc)})

    return StreamingResponse(lines(), media_type=NDJSON)


@router.post("/embed", response_model=OllamaEmbedResponse)
async def embeddings(body: OllamaEmbedRequest, request: Request) -> OllamaEmbedResponse:
    """Embeddings for one string or a batch."""
    state = _state(request)
    ref = state.resolve_model(body.model)
    texts = [body.input] if isinstance(body.input, str) else body.input
    vectors = await embed(state, ref, texts)
    return OllamaEmbedResponse(model=ref, embeddings=vectors)


@router.get("/usage")
def usage(request: Request) -> JSONResponse:
    """Aggregate latency and throughput for everything served so far."""
    return JSONResponse(_state(request).telemetry.summary())
