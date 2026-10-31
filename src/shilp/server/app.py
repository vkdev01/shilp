"""Assemble the app: state, two routers, health, metrics and a status page.

`create_app` takes its collaborators as arguments. That one decision is what
makes the whole HTTP surface testable: pass a temp `SHILP_HOME` and the `fake`
engine, and every route runs in-process with no weights and no network.
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .. import __version__
from ..config import Settings, get_settings
from ..engine.loader import ModelLoader, ModelNotReady
from ..hub import Hub, HubError
from ..modify.toolchain import ToolError
from ..store import Store, StoreError
from ..utils import MissingDependency, RefError
from .metrics import Metrics
from .routes_ollama import router as ollama_router
from .routes_openai import router as openai_router
from .state import AppState
from .telemetry import Telemetry


def _status_page(state: AppState) -> str:
    models = state.store.list()
    rows = "\n".join(
        f"<tr><td><code>{m.ref}</code></td><td>{m.metadata.get('quantization', '—')}</td>"
        f"<td>{m.human_size}</td><td>{m.metadata.get('source', '—')}</td></tr>"
        for m in models
    ) or "<tr><td colspan=4>no models yet — run <code>shilp pull hf:org/model-GGUF:q4_k_m</code></td></tr>"
    resident = ", ".join(r["ref"] for r in state.loader.loaded()) or "none"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>shilp {__version__}</title>
<style>
 body{{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:2.5rem;max-width:52rem;
      margin-inline:auto;color:#18181b;background:#fafafa}}
 h1{{font-size:1.4rem;margin:0 0 .25rem}} .muted{{color:#71717a}}
 code{{background:#f4f4f5;padding:.1rem .3rem;border-radius:4px;font-size:.9em}}
 table{{width:100%;border-collapse:collapse;margin-top:1.5rem}}
 th,td{{text-align:left;padding:.5rem .4rem;border-bottom:1px solid #e4e4e7}}
 th{{font-weight:600;color:#52525b;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}}
 .apis{{margin-top:1.75rem}} .apis li{{margin:.3rem 0}}
</style></head><body>
<h1>shilp</h1>
<p class="muted">A model server you can read. Currently serving: <strong>{resident}</strong></p>
<table><thead><tr><th>model</th><th>quant</th><th>size</th><th>source</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="apis"><h3>Same engine, two surfaces</h3><ul>
 <li>Ollama: <code>POST /api/chat</code>, <code>/api/generate</code>, <code>/api/tags</code>, <code>/api/pull</code></li>
 <li>OpenAI: <code>POST /v1/chat/completions</code>, <code>GET /v1/models</code></li>
 <li>Ops: <a href="/health">/health</a>, <a href="/metrics">/metrics</a>, <a href="/api/usage">/api/usage</a>, <a href="/api/ps">/api/ps</a></li>
</ul></div>
<p class="muted" style="margin-top:2rem">Point any OpenAI client at <code>http://127.0.0.1:{state.settings.port}/v1</code>.</p>
</body></html>"""


def create_app(
    settings: Settings | None = None,
    *,
    engine: str | None = None,
    hub: Hub | None = None,
    store: Store | None = None,
    loader: ModelLoader | None = None,
) -> FastAPI:
    """Build the ASGI app.

    `engine="fake"` swaps in the deterministic engine — used by the test suite and
    by `make demo`, so the API can be shown without downloading a model.
    """
    base = settings or get_settings()
    if engine:
        base = dataclasses.replace(base, engine=engine)
    model_store = store or Store(base.home)
    telemetry = Telemetry(base.db_path)
    metrics = Metrics()
    model_loader = loader or ModelLoader(model_store, base)

    state = AppState(
        settings=base,
        store=model_store,
        loader=model_loader,
        telemetry=telemetry,
        metrics=metrics,
        hub=hub,
        default_model=None,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await model_loader.unload()
        telemetry.close()

    app = FastAPI(
        title="shilp",
        version=__version__,
        summary="Pull a model from Hugging Face, reshape it, serve it like Ollama.",
        lifespan=lifespan,
    )
    app.state.shilp = state
    app.include_router(ollama_router)
    app.include_router(openai_router)

    # --- error translation --------------------------------------------
    # One place decides what a failure looks like over HTTP, so both surfaces
    # behave the same way and the message names the next action.
    @app.exception_handler(StoreError)
    async def _not_found(request: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "model_not_found"}, status_code=404)

    @app.exception_handler(RefError)
    async def _bad_ref(request: Request, exc: RefError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "bad_request"}, status_code=400)

    @app.exception_handler(ValueError)
    async def _bad_value(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "bad_request"}, status_code=400)

    @app.exception_handler(ModelNotReady)
    async def _busy(request: Request, exc: ModelNotReady) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "model_unavailable"}, status_code=503)

    @app.exception_handler(MissingDependency)
    async def _missing(request: Request, exc: MissingDependency) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "missing_dependency"}, status_code=500)

    @app.exception_handler(ToolError)
    async def _tool(request: Request, exc: ToolError) -> JSONResponse:
        return JSONResponse(
            {"error": str(exc), "type": "tool_failed", "stderr": exc.stderr}, status_code=500
        )

    @app.exception_handler(HubError)
    async def _hub(request: Request, exc: HubError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "hub_error"}, status_code=502)

    # --- ops -----------------------------------------------------------
    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        """Liveness plus the facts you want when something looks slow."""
        stats = model_store.stats()
        return {
            "status": "ok",
            "version": __version__,
            "engine": base.engine,
            "home": str(base.home),
            "models": stats["models"],
            "store_bytes": stats["bytes"],
            "resident": model_loader.loaded(),
        }

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics_endpoint() -> str:
        """Prometheus text exposition."""
        return metrics.render()

    @app.get("/", response_class=HTMLResponse, tags=["ops"])
    def index() -> str:
        """A status page, because a bare 404 at `/` reads as broken."""
        return _status_page(state)

    return app


# No module-level app on purpose: building one at import time would create the
# store and the telemetry database as a side effect of `import shilp`. Run it
# with a factory instead:
#     uvicorn shilp.server.app:create_app --factory
