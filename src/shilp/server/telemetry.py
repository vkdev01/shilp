"""One row per request, in SQLite.

Same idea as the telemetry table in my other projects: a request is either
measured or it did not happen. The columns answer the questions you actually ask
about a local server — how fast, how much, which model, and where the time went.

`ttft_ms` is time to first token, which is the number users feel. `tok_per_s` is
completion tokens over generation time, excluding load and prompt processing.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    surface       TEXT    NOT NULL,   -- 'ollama' | 'openai'
    endpoint      TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    stream        INTEGER NOT NULL,
    status        INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    ttft_ms       REAL    NOT NULL DEFAULT 0,
    total_ms      REAL    NOT NULL DEFAULT 0,
    tok_per_s     REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


@dataclass
class RequestRecord:
    """The mutable facts of one in-flight request."""

    surface: str
    endpoint: str
    model: str
    stream: bool
    started: float = time.monotonic()
    first_token_at: float | None = None
    finished: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status: int = 200

    def first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()

    def finish(self, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.finished = time.monotonic()
        self.prompt_tokens = prompt_tokens or self.prompt_tokens
        self.completion_tokens = completion_tokens or self.completion_tokens

    # --- derived -------------------------------------------------------
    @property
    def total_ms(self) -> float:
        end = self.finished or time.monotonic()
        return (end - self.started) * 1000.0

    @property
    def ttft_ms(self) -> float:
        if self.first_token_at is None:
            return 0.0
        return (self.first_token_at - self.started) * 1000.0

    @property
    def tok_per_s(self) -> float:
        if not self.first_token_at or not self.finished or self.completion_tokens == 0:
            return 0.0
        generation = max(1e-6, self.finished - self.first_token_at)
        return self.completion_tokens / generation


class Telemetry:
    """A tiny write-behind store: one connection, one lock, no ORM."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def record(self, rec: RequestRecord) -> None:
        rec.finish()  # idempotent: keeps whatever counts were already set
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests (ts, surface, endpoint, model, stream, status,"
                " prompt_tokens, completion_tokens, ttft_ms, total_ms, tok_per_s)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    rec.surface,
                    rec.endpoint,
                    rec.model,
                    int(rec.stream),
                    rec.status,
                    rec.prompt_tokens,
                    rec.completion_tokens,
                    rec.ttft_ms,
                    rec.total_ms,
                    rec.tok_per_s,
                ),
            )
            self._conn.commit()

    def summary(self, *, model: str | None = None) -> dict[str, Any]:
        """Aggregates for `GET /api/usage` and `shilp usage`."""
        where, args = ("WHERE model = ?", (model,)) if model else ("", ())
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                f" COALESCE(SUM(completion_tokens),0), COALESCE(AVG(ttft_ms),0),"
                f" COALESCE(AVG(tok_per_s),0), COALESCE(MAX(tok_per_s),0)"
                f" FROM requests {where}",
                args,
            ).fetchone()
            by_model = self._conn.execute(
                "SELECT model, COUNT(*), SUM(completion_tokens), AVG(tok_per_s)"
                " FROM requests GROUP BY model ORDER BY COUNT(*) DESC"
            ).fetchall()
        return {
            "requests": row[0],
            "prompt_tokens": row[1],
            "completion_tokens": row[2],
            "avg_ttft_ms": round(row[3], 1),
            "avg_tok_per_s": round(row[4], 1),
            "best_tok_per_s": round(row[5], 1),
            "by_model": [
                {
                    "model": m,
                    "requests": c,
                    "completion_tokens": t or 0,
                    "avg_tok_per_s": round(s or 0, 1),
                }
                for m, c, t, s in by_model
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
