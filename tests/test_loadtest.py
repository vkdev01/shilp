"""The load-test script, driven in-process against the fake engine.

It is a measuring instrument, so the thing worth pinning is that it parses both
dialects, counts what it claims to count, and fails the SLO gate when it should.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

from shilp.store import Manifest, Store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "loadtest.py"
spec = importlib.util.spec_from_file_location("loadtest", SCRIPT)
loadtest = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["loadtest"] = loadtest  # dataclasses resolve types through sys.modules
spec.loader.exec_module(loadtest)


@pytest.fixture
def app(settings, store: Store, tmp_path: Path):
    from shilp.server.app import create_app

    weights = tmp_path / "tiny-Q4_K_M.gguf"
    weights.write_bytes(b"GGUF" * 32)
    store.save_manifest(
        Manifest(name="tiny", tag="q4", layers=[store.put_blob(weights)], params={"num_predict": 8})
    )
    return create_app(settings=settings, engine="fake")


def _run(app, **overrides):
    cfg = loadtest.Config(url="http://test", warmup=1, **overrides)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await loadtest.run(cfg, client)

    results, wall = asyncio.run(go())
    return cfg, results, loadtest.summarise(results, wall)


@pytest.mark.parametrize("surface", ["openai", "ollama"])
def test_closed_loop_counts_every_request_on_both_surfaces(app, surface):
    cfg, results, summary = _run(app, surface=surface, concurrency=3, requests=9)
    assert cfg.model == "tiny:q4"  # resolved from /v1/models
    assert summary["requests"] == 9 and summary["succeeded"] == 9
    assert summary["statuses"] == {"200": 9}
    assert all(r.completion_tokens > 0 and r.prompt_tokens > 0 for r in results)
    assert summary["ttft_ms"]["p50"] is not None
    assert summary["latency_ms"]["p99"] >= summary["latency_ms"]["p50"]


def test_open_loop_issues_the_requested_number(app):
    _, results, summary = _run(app, rate=50.0, requests=10)
    assert summary["requests"] == 10
    dues = sorted(r.due for r in results)
    assert all(b > a for a, b in zip(dues, dues[1:])), "arrivals must be distinct, increasing times"
    # Latency is charged from when a request was due, never from a later send time.
    assert all(r.sent >= r.due - 1e-3 for r in results)


def test_errors_are_counted_not_raised(app, monkeypatch):
    from shilp.engine.base import ContextOverflow
    from shilp.engine.fake import FakeEngine

    def overflow(self, *args, **kwargs):
        raise ContextOverflow("prompt is 9000 tokens but the context window is 4096")
        yield  # pragma: no cover

    monkeypatch.setattr(FakeEngine, "_emit", overflow)
    _, _, summary = _run(app, concurrency=2, requests=4)
    assert summary["succeeded"] == 0
    assert summary["statuses"] == {"400": 4}
    assert summary["error_rate"] == 1.0


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert loadtest.percentile(values, 50) == 50.0
    assert loadtest.percentile(values, 95) == 95.0
    assert loadtest.percentile(values, 99) == 99.0
    assert loadtest.percentile([], 50) is None


def test_slo_gate_reports_each_breach():
    summary = {"requests": 50, "ttft_ms": {"p95": 1800.0}, "error_rate": 0.02}
    assert loadtest.check_slos(summary, ttft_p95_ms=2000, error_rate=0.05) == []
    failures = loadtest.check_slos(summary, ttft_p95_ms=1000, error_rate=0.01)
    assert len(failures) == 2


def test_mix_is_validated_and_normalised():
    assert loadtest.parse_mix("short=3,long=1") == {"short": 0.75, "long": 0.25}
    with pytest.raises(ValueError):
        loadtest.parse_mix("huge=1")


def test_open_loop_does_not_overshoot_its_duration(app):
    import time as _time

    started = _time.monotonic()
    # A mean gap of 100 s: the next arrival is almost surely past the 0.5 s window.
    _run(app, rate=0.01, duration=0.5)
    assert _time.monotonic() - started < 5


def test_a_malformed_stream_is_one_failed_request_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, text="data: {not json}\n\n")

    cfg = loadtest.Config(url="http://test", warmup=0, concurrency=2, requests=4)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            return await loadtest.run(cfg, client)

    results, wall = asyncio.run(go())
    summary = loadtest.summarise(results, wall)
    assert summary["requests"] == 4 and summary["succeeded"] == 0
    assert all("JSONDecodeError" in r.error for r in results)


def test_an_slo_gate_with_nothing_measured_fails():
    empty = loadtest.summarise([], 0.0)
    assert loadtest.check_slos(empty, ttft_p95_ms=1000, error_rate=None) == ["no requests were measured"]
    assert loadtest.check_slos(empty, ttft_p95_ms=None, error_rate=None) == []


@pytest.mark.parametrize(
    "argv",
    [["--concurrency", "0"], ["--rate", "-1"], ["--requests", "-5"], ["--duration", "-1"]],
)
def test_nonsense_load_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        loadtest.main(argv)
    assert exc.value.code == 2
