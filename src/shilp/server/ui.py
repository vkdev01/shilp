"""Browser pages: a status page at `/` and a chat client at `/chat`.

Markup lives in `templates/`, styling and behaviour in `static/`; this module
only gathers what each page needs. The chat page has no private endpoints — it
calls `/api/chat` or `/v1/chat/completions` exactly as any other client would,
which is what makes it useful for testing the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import __version__
from .state import AppState

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["version"] = __version__

router = APIRouter(tags=["ui"])


def _state(request: Request) -> AppState:
    return request.app.state.shilp


def _models(state: AppState) -> list[dict[str, Any]]:
    return [
        {
            "ref": m.ref,
            "quantization": m.metadata.get("quantization", "—"),
            "size": m.human_size,
            "source": m.metadata.get("source", "—"),
        }
        for m in state.store.list()
    ]


@router.get("/", response_class=HTMLResponse)
def status_page(request: Request) -> HTMLResponse:
    """What is stored, what is resident, and where the APIs are."""
    state = _state(request)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "page": "status",
            "models": _models(state),
            "resident": [r["ref"] for r in state.loader.loaded()],
            "engine": state.settings.engine,
            "port": state.settings.port,
        },
    )


@router.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    """A chat client for both API surfaces."""
    state = _state(request)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"page": "chat", "models": _models(state), "engine": state.settings.engine},
    )
