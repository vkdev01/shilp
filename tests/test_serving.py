"""Serving behaviour that only shows up under real use.

Each test here pins a bug that was found by measuring a running server rather
than by reading the code: durations that counted from process start, a GPU that
was never used, prompt counts stuck at 1, abandoned answers that kept the model
busy, and errors that cut a stream instead of returning a status code.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from shilp.config import Settings
from shilp.engine.base import ChatMessage, Chunk, ContextOverflow, EngineError, GenParams
from shilp.engine.fake import FakeEngine
from shilp.store import Manifest, Store


@pytest.fixture
def seeded(store: Store, tmp_path: Path) -> Store:
    weights = tmp_path / "tiny-Q4_K_M.gguf"
    weights.write_bytes(b"GGUF" * 32)
    store.save_manifest(
        Manifest(
            name="tiny",
            tag="q4",
            layers=[store.put_blob(weights)],
            params={"num_predict": 6},
            metadata={"source": "hf:acme/tiny-GGUF", "quantization": "Q4_K_M"},
        )
    )
    return store


def _ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sse(text: str) -> list[str]:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


# --- 1. durations are per request ----------------------------------------
def test_each_request_record_starts_its_own_clock():
    from shilp.server.telemetry import RequestRecord

    first = RequestRecord(surface="ollama", endpoint="/api/chat", model="m", stream=True)
    time.sleep(0.05)
    second = RequestRecord(surface="ollama", endpoint="/api/chat", model="m", stream=True)
    assert second.started - first.started >= 0.04
    second.first_token()
    assert second.ttft_ms < 1000


def test_recorded_latency_is_the_request_not_the_process_uptime(client, seeded):
    time.sleep(0.3)  # uptime the old default would have folded into every number
    client.post("/api/generate", json={"model": "tiny:q4", "prompt": "x", "stream": False})
    usage = client.get("/api/usage").json()
    assert 0 < usage["avg_ttft_ms"] < 250


# --- 2. the GPU is used unless told otherwise -----------------------------
def test_gpu_layers_default_to_all_and_read_the_environment(monkeypatch):
    monkeypatch.delenv("SHILP_GPU_LAYERS", raising=False)
    assert Settings.from_env().gpu_layers == -1
    monkeypatch.setenv("SHILP_GPU_LAYERS", "0")
    assert Settings.from_env().gpu_layers == 0


def test_loader_passes_gpu_layers_to_llamacpp(monkeypatch, settings, store, tmp_path):
    import dataclasses

    import shilp.engine.llamacpp as llamacpp
    from shilp.engine.loader import _default_factory

    seen: dict = {}

    class Spy:
        def __init__(self, manifest, path, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(llamacpp, "LlamaCppEngine", Spy)
    build = _default_factory(dataclasses.replace(settings, engine="llamacpp", gpu_layers=12))
    build(Manifest(name="m", tag="t", layers=[]), tmp_path / "m.gguf")
    assert seen["gpu_layers"] == 12


# --- 3. prompt tokens count the rendered prompt ---------------------------
class _StubLlama:
    """Just enough of `llama_cpp.Llama` for token counting: one token per byte."""

    def __init__(self, template: str | None):
        self.metadata = {"tokenizer.chat_template": template} if template else {}

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return list(data) + ([1] if add_bos else [])

    def token_eos(self) -> int:
        return -1  # "no such token": the formatter gets an empty string

    def token_bos(self) -> int:
        return -1


def _bare_engine(llm) -> object:
    from shilp.engine.llamacpp import LlamaCppEngine

    engine = object.__new__(LlamaCppEngine)
    engine._llm = llm
    engine._formatter = None
    engine._n_ctx = 4096
    engine._lock = threading.Lock()
    return engine


MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is the capital of France?"},
]


def test_chat_prompt_tokens_without_a_template_count_the_content():
    engine = _bare_engine(_StubLlama(template=None))
    content = sum(len(m["content"]) for m in MESSAGES)
    assert engine._prompt_tokens(MESSAGES) >= content  # never the old constant 1


def test_chat_prompt_tokens_render_the_gguf_template():
    pytest.importorskip("llama_cpp")
    template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</s>{% endfor %}<assistant>"
    engine = _bare_engine(_StubLlama(template=template))
    rendered = "<system>Be brief.</s><user>What is the capital of France?</s><assistant>"
    assert engine._prompt_tokens(MESSAGES) == len(rendered.encode())


# --- 4. a prompt that does not fit is a 400, before anything streams ------
def test_engine_rejects_a_prompt_longer_than_the_context():
    engine = _bare_engine(_StubLlama(template=None))
    engine._n_ctx = 16
    with pytest.raises(ContextOverflow, match="context window is 16"):
        next(engine._stream("x" * 64, GenParams(), raw_prompt=True))
    assert not engine._lock.locked()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/chat", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/api/generate", {"prompt": "hi"}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/completions", {"prompt": "hi"}),
    ],
)
def test_streaming_failure_before_first_token_is_a_status_code(
    client, seeded, monkeypatch, path, body
):
    def overflow(self, *args, **kwargs) -> Iterator[Chunk]:
        raise ContextOverflow("prompt is 9000 tokens but the context window is 4096")
        yield  # pragma: no cover

    monkeypatch.setattr(FakeEngine, "_emit", overflow)
    res = client.post(path, json={"model": "tiny:q4", "stream": True, **body})
    assert res.status_code == 400
    assert "context window" in res.json()["error"]
    # Telemetry records what the client saw, not a blanket 500.
    usage_db = client.app.state.shilp.telemetry._conn
    assert usage_db.execute("SELECT status FROM requests").fetchone()[0] == 400
    assert 'status="400"' in client.get("/metrics").text


def test_ollama_stream_ends_with_an_error_line_when_it_fails_midway(client, seeded, monkeypatch):
    def flaky(self, prompt, params) -> Iterator[Chunk]:
        yield Chunk(text="partial ")
        raise EngineError("llama.cpp failed while generating: boom")

    monkeypatch.setattr(FakeEngine, "_emit", flaky)
    res = client.post(
        "/api/chat",
        json={"model": "tiny:q4", "messages": [{"role": "user", "content": "hi"}]},
    )
    lines = _ndjson(res.text)
    assert lines[0]["message"]["content"] == "partial "
    assert "boom" in lines[-1]["error"]


def test_openai_stream_ends_with_an_error_frame_and_no_done(client, seeded, monkeypatch):
    def flaky(self, prompt, params) -> Iterator[Chunk]:
        yield Chunk(text="partial ")
        raise EngineError("llama.cpp failed while generating: boom")

    monkeypatch.setattr(FakeEngine, "_emit", flaky)
    res = client.post(
        "/v1/chat/completions",
        json={"model": "tiny:q4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    frames = _sse(res.text)
    assert "[DONE]" not in frames
    assert "boom" in json.loads(frames[-1])["error"]["message"]


# --- 5. an abandoned answer stops, and is recorded as such ---------------
def test_closing_the_stream_stops_the_worker_and_closes_the_source():
    from shilp.server.stream import to_async

    produced: list[int] = []
    closed = threading.Event()

    def slow() -> Iterator[Chunk]:
        try:
            for i in range(200):
                time.sleep(0.01)
                produced.append(i)
                yield Chunk(text=str(i))
        finally:
            closed.set()  # what releases the engine lock in the real engine

    async def read_two_then_leave() -> None:
        stream = to_async(slow())
        await stream.__anext__()
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(read_two_then_leave())
    assert closed.wait(1.0), "the source generator was never closed"
    assert len(produced) < 20, f"kept generating after the reader left: {len(produced)}"


def test_abandoned_request_is_recorded_with_status_499(settings, seeded):
    from shilp.server.app import create_app
    from shilp.server.generate import STATUS_CLIENT_CLOSED, stream_chat
    from shilp.server.telemetry import RequestRecord

    state = create_app(settings=settings, engine="fake").state.shilp
    rec = RequestRecord(surface="ollama", endpoint="/api/chat", model="tiny:q4", stream=True)

    async def leave_after_first_chunk() -> None:
        stream = stream_chat(
            state, rec, "tiny:q4", [ChatMessage(role="user", content="hi")], GenParams()
        )
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(leave_after_first_chunk())
    assert rec.status == STATUS_CLIENT_CLOSED
    row = state.telemetry._conn.execute("SELECT status FROM requests").fetchone()
    assert row[0] == STATUS_CLIENT_CLOSED
