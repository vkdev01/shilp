"""The HTTP surface, end to end, against the fake engine.

These are the tests that matter most: they are what proves both API surfaces are
the same engine, that streaming really streams, and that telemetry records what
it claims to.
"""

from __future__ import annotations

import json

import pytest

from shilp.store import Manifest, Store


@pytest.fixture
def seeded(store: Store, tmp_path):
    """One runnable model in the store: 'tiny:q4'."""
    weights = tmp_path / "tiny-Q4_K_M.gguf"
    weights.write_bytes(b"GGUF" * 32)
    store.save_manifest(
        Manifest(
            name="tiny",
            tag="q4",
            layers=[store.put_blob(weights)],
            params={"temperature": 0.2, "num_predict": 6},
            metadata={"source": "hf:acme/tiny-GGUF", "quantization": "Q4_K_M"},
        )
    )
    return store


def test_health_and_status_page(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "shilp" in client.get("/").text
    assert client.get("/metrics").text  # may be empty, but must answer


def test_tags_lists_the_model(client, seeded):
    body = client.get("/api/tags").json()
    assert body["models"][0]["name"] == "tiny:q4"
    assert body["models"][0]["details"]["quantization_level"] == "Q4_K_M"


def test_show_round_trips_a_modelfile(client, seeded):
    body = client.post("/api/show", json={"model": "tiny:q4"}).json()
    assert body["modelfile"].startswith("FROM hf:acme/tiny-GGUF")
    assert "PARAMETER temperature" in body["modelfile"]
    assert body["details"]["format"] == "gguf"


def test_chat_streams_ndjson_and_ends_with_done(client, seeded):
    with client.stream(
        "POST",
        "/api/chat",
        json={"model": "tiny:q4", "messages": [{"role": "user", "content": "hello"}]},
    ) as response:
        assert response.status_code == 200
        lines = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert lines[0]["done"] is False
    assert lines[0]["message"]["role"] == "assistant"
    assert lines[-1]["done"] is True
    assert lines[-1]["eval_count"] > 0
    assert lines[-1]["total_duration"] > 0


def test_chat_non_stream_returns_one_object(client, seeded):
    body = client.post(
        "/api/chat",
        json={
            "model": "tiny:q4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    ).json()
    assert body["done"] is True
    assert body["message"]["content"]
    assert body["prompt_eval_count"] > 0


def test_generate_non_stream(client, seeded):
    body = client.post(
        "/api/generate",
        json={"model": "tiny:q4", "prompt": "write something", "stream": False},
    ).json()
    assert body["done"] is True
    assert body["response"]


def test_unknown_model_is_404_with_a_readable_message(client, seeded):
    response = client.post(
        "/api/chat", json={"model": "ghost:1", "messages": [{"role": "user", "content": "x"}]}
    )
    assert response.status_code == 404
    assert "ghost:1" in response.json()["error"]


def test_model_can_be_omitted_when_only_one_exists(client, seeded):
    body = client.post(
        "/api/generate", json={"model": "", "prompt": "hi", "stream": False}
    ).json()
    assert body["model"] == "tiny:q4"


def test_openai_chat_completion(client, seeded):
    body = client.post(
        "/v1/chat/completions",
        json={"model": "tiny:q4", "messages": [{"role": "user", "content": "hello"}]},
    ).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_openai_stream_uses_sse_and_done_sentinel(client, seeded):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "tiny:q4", "messages": [{"role": "user", "content": "hello"}], "stream": True},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            line.removeprefix("data: ").strip()
            for line in response.iter_lines()
            if line.strip()
        ]

    assert payloads[-1] == "[DONE]"
    deltas = [json.loads(p) for p in payloads[:-1]]
    assert deltas[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(d["choices"][0]["delta"].get("content") for d in deltas)
    assert deltas[-1]["choices"][0]["finish_reason"] == "stop"


def test_openai_models_listing(client, seeded):
    body = client.get("/v1/models").json()
    assert body["data"][0]["id"] == "tiny:q4"


def test_embeddings_on_both_surfaces(client, seeded):
    ollama = client.post("/api/embed", json={"model": "tiny:q4", "input": "a"}).json()
    assert len(ollama["embeddings"][0]) == 8

    openai = client.post("/v1/embeddings", json={"model": "tiny:q4", "input": ["a", "b"]}).json()
    assert len(openai["data"]) == 2


def test_telemetry_records_the_request(client, seeded):
    client.post(
        "/api/generate", json={"model": "tiny:q4", "prompt": "measure me", "stream": False}
    )
    summary = client.get("/api/usage").json()
    assert summary["requests"] == 1
    assert summary["completion_tokens"] > 0
    assert summary["by_model"][0]["model"] == "tiny:q4"


def test_metrics_expose_counters(client, seeded):
    client.post("/api/generate", json={"model": "tiny:q4", "prompt": "x", "stream": False})
    text = client.get("/metrics").text
    assert "shilp_requests_total" in text
    assert "shilp_tokens_total" in text


def test_ps_reports_the_resident_model(client, seeded):
    client.post("/api/generate", json={"model": "tiny:q4", "prompt": "x", "stream": False})
    body = client.get("/api/ps").json()
    assert body["models"][0]["ref"] == "tiny:q4"


def test_pull_endpoint_uses_the_injected_hub(settings, fake_hub):
    from fastapi.testclient import TestClient

    from shilp.server.app import create_app

    with TestClient(create_app(settings=settings, engine="fake", hub=fake_hub)) as client:
        body = client.post("/api/pull", json={"model": "hf:acme/tiny-GGUF:q4_k_m", "stream": False}).json()
    assert body["status"] == "success"
    assert body["model"] == "tiny-gguf:q4_k_m"
