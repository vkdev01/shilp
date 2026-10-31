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


def load_dotenv(path: Path) -> int:
    """Read `KEY=VALUE` lines from `path` into the environment.

    Real environment variables always win (`setdefault`), so a shell export or a
    host that injects secrets is never overridden by a stray file. Returns how
    many keys were set, which is what `shilp doctor` reports.
    """
    if not path.is_file():
        return 0
    applied = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        applied += 1
    return applied


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
        # `.env` in the working directory, unless SHILP_ENV_FILE points elsewhere.
        load_dotenv(_env_path("SHILP_ENV_FILE") or Path.cwd() / ".env")
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
