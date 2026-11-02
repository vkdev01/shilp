"""The browser UI and the one JSON endpoint it adds.

The UI is a Next.js app in `web/`, exported to static files and copied into
`webui/` beside this module by `make ui`. The server only hands out files: the
pages talk to `/api/stats` and the public Ollama/OpenAI endpoints, exactly as
any other client would, which keeps the UI an honest test of the API.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from .. import __version__
from .state import AppState

UI_DIR = Path(__file__).resolve().parent / "webui"

router = APIRouter(tags=["ops"])


@router.get("/api/stats")
def stats(
    request: Request,
    window: int = Query(3600, ge=60, le=7 * 86400, description="Seconds to aggregate"),
    recent: int = Query(50, ge=0, le=500, description="Newest requests to return"),
) -> dict[str, Any]:
    """Everything the dashboard draws: server facts, residency, and request telemetry."""
    state: AppState = request.app.state.shilp
    settings = state.settings
    resident = state.loader.loaded()
    return {
        "server": {
            "version": __version__,
            "engine": settings.engine,
            "host": settings.host,
            "port": settings.port,
            "max_resident": settings.max_resident,
            "gpu_layers": settings.gpu_layers,
            "threads": settings.threads or os.cpu_count(),
            "default_model": state.default_model,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "started_at": state.started_at,
            "uptime_s": round(time.time() - state.started_at, 1),
        },
        "store": state.store.stats(),
        "resident": resident,
        "in_flight": sum(r["refs"] for r in resident),
        "lifetime": state.telemetry.summary(),
        "window": state.telemetry.window(window),
        "recent": state.telemetry.recent(recent),
    }


_NOT_BUILT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>shilp</title>
<style>body{font:15px/1.6 system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem}
code{background:#f3efe9;padding:.1rem .35rem;border-radius:4px}</style></head>
<body><h1>shilp {version}</h1>
<p>The server is up, but the web UI has not been built.</p>
<p>Build it with <code>make ui</code> (needs Node 20+), then reload.</p>
<p>The API works without it: <a href="/docs">/docs</a> · <a href="/health">/health</a>
· <a href="/api/stats">/api/stats</a> · <a href="/metrics">/metrics</a></p>
</body></html>
"""


class _UIFiles(StaticFiles):
    """The export, except that an unknown API path stays a JSON 404 for API clients."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        if path.split("/", 1)[0] in ("api", "v1"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return await super().get_response(path, scope)


def mount_ui(app: FastAPI, ui_dir: Path | None = None) -> None:
    """Serve the exported UI at `/`, or a page saying how to build it.

    Mounted last, so every API route is matched before a file lookup happens.
    """
    directory = Path(ui_dir) if ui_dir else UI_DIR
    if (directory / "index.html").is_file():
        app.mount("/", _UIFiles(directory=str(directory), html=True), name="ui")
        return

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def not_built() -> str:
        return _NOT_BUILT.replace("{version}", __version__)
