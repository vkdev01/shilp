"""One row per request, in SQLite.

Same idea as the telemetry table in my other projects: a request is either
measured or it did not happen. The columns answer the questions you actually ask
about a local server — how fast, how much, which model, and where the time went.

`ttft_ms` is time to first token, which is the number users feel. `tok_per_s` is
completion tokens over generation time, excluding load and prompt processing.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
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
    # A factory, not `= time.monotonic()`: a plain default is evaluated once, at
    # import, which made every duration "time since the server started".
    started: float = field(default_factory=time.monotonic)
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

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """The newest requests, newest first, for the dashboard's request log."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM requests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(r) for r in rows]

    def window(self, seconds: float, *, buckets: int = 60) -> dict[str, Any]:
        """Percentiles, breakdowns and a time series for the last `seconds`.

        Percentiles are computed in Python over the window's rows: SQLite has no
        percentile function, and a local server's hour is a few thousand rows.
        """
        now = time.time()
        since = now - seconds
        with self._lock:
            rows = [
                _row(r)
                for r in self._conn.execute(
                    f"SELECT {_COLUMNS} FROM requests WHERE ts >= ? ORDER BY ts LIMIT ?",
                    (since, WINDOW_ROW_LIMIT),
                ).fetchall()
            ]

        ok = [r for r in rows if r["status"] < 400]
        errors = [r for r in rows if _is_error(r)]
        ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
        latency = [r["total_ms"] for r in ok]
        tps = [r["tok_per_s"] for r in ok if r["tok_per_s"] > 0]
        completion = sum(r["completion_tokens"] for r in rows)

        width = seconds / buckets
        series = [
            {"ts": since + i * width, "requests": 0, "errors": 0, "tokens": 0,
             "_ttft": [], "_latency": [], "_tps": []}
            for i in range(buckets)
        ]
        for r in rows:
            b = series[min(buckets - 1, int((r["ts"] - since) / width))]
            b["requests"] += 1
            b["tokens"] += r["completion_tokens"]
            if _is_error(r):
                b["errors"] += 1
            elif r["status"] < 400:
                if r["ttft_ms"] > 0:
                    b["_ttft"].append(r["ttft_ms"])
                b["_latency"].append(r["total_ms"])
                if r["tok_per_s"] > 0:
                    b["_tps"].append(r["tok_per_s"])
        for b in series:
            b["ttft_p50_ms"] = _pct(b.pop("_ttft"), 50)
            lat = b.pop("_latency")
            b["latency_p50_ms"] = _pct(lat, 50)
            b["latency_p95_ms"] = _pct(lat, 95)
            b["tok_per_s"] = _mean(b.pop("_tps"))

        return {
            "seconds": seconds,
            "bucket_seconds": width,
            "requests": len(rows),
            "errors": len(errors),
            "cancelled": sum(1 for r in rows if r["status"] == 499),
            "error_rate": round(len(errors) / len(rows), 4) if rows else 0.0,
            "rps": round(len(rows) / seconds, 4),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": completion,
            "output_tok_per_s": round(completion / seconds, 3),
            "ttft_ms": _dist(ttft),
            "latency_ms": _dist(latency),
            "tok_per_s": _dist(tps),
            "by_model": _group(rows, "model"),
            "by_surface": _group(rows, "surface"),
            "by_endpoint": _group(rows, "endpoint"),
            "by_status": _count(rows, "status"),
            "series": series,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


#: Caps the rows a dashboard window reads, so a long window cannot stall the server.
WINDOW_ROW_LIMIT = 20_000

_COLUMNS = (
    "id, ts, surface, endpoint, model, stream, status, prompt_tokens,"
    " completion_tokens, ttft_ms, total_ms, tok_per_s"
)
_KEYS = [c.strip() for c in _COLUMNS.split(",")]


def _row(r: tuple) -> dict[str, Any]:
    out = dict(zip(_KEYS, r))
    out["stream"] = bool(out["stream"])
    for k in ("ttft_ms", "total_ms", "tok_per_s"):
        out[k] = round(out[k], 2)
    return out


def _is_error(r: dict[str, Any]) -> bool:
    """A failure the server owns; 499 is the client hanging up, not an error."""
    return r["status"] >= 400 and r["status"] != 499


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return round(ordered[rank], 2)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _dist(values: list[float]) -> dict[str, float]:
    return {
        "avg": _mean(values),
        "p50": _pct(values, 50),
        "p90": _pct(values, 90),
        "p95": _pct(values, 95),
        "p99": _pct(values, 99),
        "max": round(max(values), 2) if values else 0.0,
    }


def _group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    out = []
    for name, items in groups.items():
        ok = [r for r in items if r["status"] < 400]
        out.append(
            {
                key: name,
                "requests": len(items),
                "errors": sum(1 for r in items if _is_error(r)),
                "completion_tokens": sum(r["completion_tokens"] for r in items),
                "ttft_p50_ms": _pct([r["ttft_ms"] for r in ok if r["ttft_ms"] > 0], 50),
                "latency_p95_ms": _pct([r["total_ms"] for r in ok], 95),
                "tok_per_s": _mean([r["tok_per_s"] for r in ok if r["tok_per_s"] > 0]),
            }
        )
    return sorted(out, key=lambda g: g["requests"], reverse=True)


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r[key])] = counts.get(str(r[key]), 0) + 1
    return counts
