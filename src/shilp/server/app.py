"""Assemble the app: state, two API routers, health, metrics and the browser UI.

`create_app` takes its collaborators as arguments. That one decision is what
makes the whole HTTP surface testable: pass a temp `SHILP_HOME` and the `fake`
engine, and every route runs in-process with no weights and no network.
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

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
from .ui import STATIC_DIR
from .ui import router as ui_router


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
    app.include_router(ui_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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

    return app


# No module-level app on purpose: building one at import time would create the
# store and the telemetry database as a side effect of `import shilp`. Run it
# with a factory instead:
#     uvicorn shilp.server.app:create_app --factory
