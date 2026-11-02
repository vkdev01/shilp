"""The shared runtime the routes depend on.

FastAPI's `app.state` holds one of these. Routes reach it through `get_state`,
so nothing is a module-level global and tests can build an app with a different
store, loader or hub.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..engine.loader import ModelLoader
from ..hub import Hub
from ..store import Manifest, Store
from .metrics import Metrics
from .telemetry import Telemetry


@dataclass
class AppState:
    """Everything a request may need, constructed once at startup."""

    settings: Settings
    store: Store
    loader: ModelLoader
    telemetry: Telemetry
    metrics: Metrics
    #: Set for `/api/pull`. Optional so tests can inject a fake hub.
    hub: Hub | None = None
    #: Default model when a request omits one.
    default_model: str | None = None
    started_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_model(self, requested: str | None) -> str:
        """Pick the model for a request: explicit, then default, then the only one.

        Guessing here is worth it: a client that supports only one model (most
        of them) should not have to know the tag.
        """
        if requested:
            return requested
        if self.default_model:
            return self.default_model
        models = self.store.list()
        if len(models) == 1:
            return models[0].ref
        if not models:
            raise ValueError("no models in the store; run `shilp pull hf:org/model-GGUF:q4_k_m`")
        raise ValueError(
            "several models are available; name one in the request "
            f"({', '.join(m.ref for m in models[:4])})"
        )

    def manifest(self, ref: str) -> Manifest:
        return self.store.resolve(ref)
