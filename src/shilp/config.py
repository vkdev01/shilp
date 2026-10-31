"""Runtime settings, read once from the environment.

Precedence is deliberately boring: flags > environment > defaults. There is no
config file format to learn, because the only things that vary per machine are
paths and ports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_HOME = Path.home() / ".shilp"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class Settings:
    """Everything the process needs to find its files and its engine."""

    home: Path
    #: Checkout of ggml-org/llama.cpp. Used for `convert_hf_to_gguf.py` and
    #: `llama-quantize`; optional, and only needed for the reshape step.
    llamacpp_dir: Path | None = None
    #: Token for gated repos (Llama, Gemma) on the Hub.
    hf_token: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    #: How many models may stay in memory at once. One is the honest default on
    #: a laptop: a 7B Q4 model is ~4.5 GB of resident RAM.
    max_resident: int = 1
    #: Engine used when a manifest does not pin one.
    engine: str = "llamacpp"
    #: Threads handed to llama.cpp. 0 means "let the engine decide".
    threads: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            home=_env_path("SHILP_HOME") or DEFAULT_HOME,
            llamacpp_dir=_env_path("SHILP_LLAMACPP_DIR"),
            hf_token=os.environ.get("HF_TOKEN") or None,
            host=os.environ.get("SHILP_HOST", "127.0.0.1"),
            port=int(os.environ.get("SHILP_PORT", "8000")),
            max_resident=int(os.environ.get("SHILP_MAX_RESIDENT", "1")),
            engine=os.environ.get("SHILP_ENGINE", "llamacpp"),
            threads=int(os.environ.get("SHILP_THREADS", "0")),
        )

    # --- derived paths -------------------------------------------------
    @property
    def blobs_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def manifests_dir(self) -> Path:
        return self.home / "manifests"

    @property
    def recipes_dir(self) -> Path:
        return self.home / "recipes"

    @property
    def db_path(self) -> Path:
        return self.home / "logs" / "shilp.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    """Tests mutate the environment; this drops the memoised Settings."""
    get_settings.cache_clear()
