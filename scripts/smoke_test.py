"""Boot a real server and exercise every endpoint over HTTP.

The unit tests use Starlette's TestClient, which never opens a socket. This
script does: uvicorn in a thread, real HTTP on a real port, real streaming. It is
the check that catches the failures in-process tests cannot see — wrong media
type on a stream, a response that buffers until the end, a shutdown that hangs.

    python scripts/smoke_test.py            # fake engine, no model needed
    SHILP_HOME=~/.shilp python scripts/smoke_test.py --model qwen2.5-1.5b:q4_k_m
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from shilp.config import Settings, reset_settings_cache  # noqa: E402
from shilp.server.app import create_app  # noqa: E402
from shilp.store import Manifest, Store  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def seed_if_empty(settings: Settings) -> str:
    """Guarantee at least one model, so the script runs on a clean machine."""
    store = Store(settings.home)
    models = store.list()
    if models:
        return models[0].ref
    weights = settings.home / "tmp" / "smoke.gguf"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"GGUF" + b"\x00" * 1024)
    manifest = Manifest(
        name="smoke",
        tag="q4",
        layers=[store.put_blob(weights, move=True)],
        params={"num_predict": 6},
        metadata={
            "source": "hf:acme/smoke-GGUF",
            "quantization": "Q4_K_M",
            "architecture": "smoke",
        },
    )
    store.save_manifest(manifest)
    return manifest.ref


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model to test against (default: first in the store)")
    parser.add_argument("--engine", default="fake", help="fake | llamacpp")
    args = parser.parse_args()

    reset_settings_cache()
    settings = Settings.from_env()
    model = args.model or seed_if_empty(settings)

    app = create_app(settings=settings, engine=args.engine)
    app.state.shilp.default_model = model
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    check("server started", server.started, base)

    with httpx.Client(base_url=base, timeout=60) as client:
        health = client.get("/health").json()
        check("GET /health", health["status"] == "ok", f"engine={health['engine']}")
        check("GET /", "shilp" in client.get("/").text)

        tags = client.get("/api/tags").json()["models"]
        check("GET /api/tags", any(m["name"] == model for m in tags), f"{len(tags)} model(s)")

        show = client.post("/api/show", json={"model": model}).json()
        check("POST /api/show", show["modelfile"].startswith("FROM"), show["modelfile"][:40])

        # Ollama chat, streaming: assert the first line arrives before the last.
        started = time.perf_counter()
        first_line_at = None
        streamed = 0
        final: dict = {}
        with client.stream(
            "POST",
            "/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": "hello there"}]},
        ) as response:
            check(
                "POST /api/chat content-type",
                response.headers["content-type"].startswith("application/x-ndjson"),
                response.headers["content-type"],
            )
            for line in response.iter_lines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if first_line_at is None:
                    first_line_at = time.perf_counter()
                streamed += 1
                if payload.get("done"):
                    final = payload
        check("POST /api/chat streamed", streamed > 1, f"{streamed} NDJSON lines")
        check("POST /api/chat final chunk", final.get("done") is True, f"eval_count={final.get('eval_count')}")
        if first_line_at is not None:
            check(
                "streaming is incremental",
                first_line_at - started < 5.0,
                f"first line after {(first_line_at - started) * 1000:.0f} ms",
            )

        chat = client.post(
            "/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        ).json()
        check("POST /api/chat (non-stream)", bool(chat["message"]["content"]))

        gen = client.post(
            "/api/generate", json={"model": model, "prompt": "hi", "stream": False}
        ).json()
        check("POST /api/generate", bool(gen["response"]))

        sse: list[str] = []
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            check(
                "POST /v1/chat/completions content-type",
                response.headers["content-type"].startswith("text/event-stream"),
            )
            sse = [line for line in response.iter_lines() if line.strip()]
        check("OpenAI SSE terminates with [DONE]", sse[-1].strip() == "data: [DONE]", sse[-1][:30])

        completion = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        ).json()
        check("POST /v1/chat/completions (non-stream)", completion["object"] == "chat.completion")
        check("GET /v1/models", client.get("/v1/models").json()["data"][0]["id"] == model)

        embeds = client.post("/api/embed", json={"model": model, "input": "hi"}).json()
        check("POST /api/embed", len(embeds["embeddings"][0]) > 0)

        check("GET /api/ps", isinstance(client.get("/api/ps").json()["models"], list))
        usage = client.get("/api/usage").json()
        check("GET /api/usage", usage["requests"] >= 5, f"{usage['requests']} requests recorded")
        check("GET /metrics", "shilp_requests_total" in client.get("/metrics").text)

    server.should_exit = True
    thread.join(timeout=10)
    check("server shut down", not thread.is_alive())

    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        marker = "ok  " if status == PASS else "FAIL"
        print(f"  {marker} {name.ljust(width)}  {detail}")
    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
