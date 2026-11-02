"""The web UI's server side: `/api/stats`, window aggregation, and static serving."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shilp.server.app import create_app
from shilp.server.telemetry import RequestRecord, Telemetry, _pct
from shilp.store import Manifest, Store


@pytest.fixture
def seeded(store: Store, tmp_path):
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


@pytest.fixture
def built_ui(tmp_path: Path) -> Path:
    """The shape `next build` exports with `trailingSlash: true`."""
    root = tmp_path / "webui"
    (root / "chat").mkdir(parents=True)
    (root / "_next" / "static").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>overview</title>")
    (root / "chat" / "index.html").write_text("<!doctype html><title>chat</title>")
    (root / "_next" / "static" / "app.js").write_text("console.log(1)")
    return root


# --- /api/stats ---------------------------------------------------------
def test_stats_before_any_request(client):
    body = client.get("/api/stats").json()
    assert body["server"]["engine"] == "fake"
    assert body["server"]["uptime_s"] >= 0
    assert body["in_flight"] == 0
    assert body["window"]["requests"] == 0
    assert body["window"]["ttft_ms"]["p95"] == 0
    assert len(body["window"]["series"]) == 60
    assert body["recent"] == []


def test_stats_reflect_served_requests(client, seeded):
    for surface in ("ollama", "openai"):
        url = "/api/chat" if surface == "ollama" else "/v1/chat/completions"
        res = client.post(
            url,
            json={"model": "tiny:q4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code == 200
    failed = RequestRecord(surface="ollama", endpoint="/api/chat", model="broken:x", stream=True)
    failed.status = 500
    client.app.state.shilp.telemetry.record(failed)

    body = client.get("/api/stats?window=600&recent=10").json()
    window = body["window"]
    assert window["seconds"] == 600
    assert window["requests"] == 3
    assert window["errors"] == 1
    assert window["error_rate"] == round(1 / 3, 4)
    assert window["by_status"] == {"200": 2, "500": 1}
    assert {s["surface"] for s in window["by_surface"]} == {"ollama", "openai"}
    assert window["completion_tokens"] > 0
    assert window["latency_ms"]["p50"] > 0
    assert sum(b["requests"] for b in window["series"]) == 3
    assert body["recent"][0]["model"] == "broken:x"
    assert body["recent"][0]["status"] == 500
    assert [r["ref"] for r in body["resident"]] == ["tiny:q4"]
    assert body["lifetime"]["requests"] == 3


@pytest.mark.parametrize("query", ["window=10", "window=9999999", "recent=-1", "recent=501"])
def test_stats_rejects_out_of_range_queries(client, query):
    assert client.get(f"/api/stats?{query}").status_code == 422


def test_client_hangups_are_not_errors(tmp_path):
    telemetry = Telemetry(tmp_path / "t.db")
    rec = RequestRecord(surface="ollama", endpoint="/api/chat", model="m", stream=True)
    rec.status = 499
    telemetry.record(rec)
    window = telemetry.window(60)
    assert window["requests"] == 1
    assert window["cancelled"] == 1
    assert window["errors"] == 0
    telemetry.close()


def test_percentiles_use_nearest_rank():
    values = list(range(1, 101))
    assert _pct(values, 50) == 50
    assert _pct(values, 95) == 95
    assert _pct(values, 99) == 99
    assert _pct([7.0], 99) == 7.0
    assert _pct([], 50) == 0.0


# --- static UI ----------------------------------------------------------
def test_unbuilt_ui_explains_how_to_build_it(settings, tmp_path):
    app = create_app(settings=settings, engine="fake", ui_dir=tmp_path / "missing")
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "make ui" in res.text
        assert client.get("/api/stats").status_code == 200


def test_built_ui_is_served_and_api_routes_still_win(settings, built_ui, seeded):
    app = create_app(settings=settings, engine="fake", ui_dir=built_ui)
    with TestClient(app) as client:
        assert "overview" in client.get("/").text
        assert "chat" in client.get("/chat/").text
        assert client.get("/chat", follow_redirects=False).status_code in (301, 307)
        assert client.get("/_next/static/app.js").status_code == 200
        assert client.get("/api/tags").json()["models"][0]["name"] == "tiny:q4"
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/docs").status_code == 200


def test_unknown_api_paths_stay_json_with_the_ui_mounted(settings, built_ui):
    (built_ui / "404.html").write_text("<!doctype html><title>404</title>")
    app = create_app(settings=settings, engine="fake", ui_dir=built_ui)
    with TestClient(app) as client:
        for path in ("/api/nope", "/v1/nope"):
            res = client.get(path)
            assert res.status_code == 404
            assert res.json() == {"detail": "Not Found"}
        page = client.get("/no-such-page")
        assert page.status_code == 404
        assert "text/html" in page.headers["content-type"]


def test_built_ui_does_not_escape_its_directory(settings, built_ui):
    (built_ui.parent / "secret.txt").write_text("nope")
    app = create_app(settings=settings, engine="fake", ui_dir=built_ui)
    with TestClient(app) as client:
        assert client.get("/../secret.txt").status_code == 404
        assert client.get("/%2e%2e/secret.txt").status_code == 404
