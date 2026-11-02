#!/usr/bin/env python3
"""Load-test a running shilp server — or anything that speaks its two dialects.

Two ways to apply load, because they answer different questions:

  * **closed loop** (`--concurrency N`): N simulated users, each sending the next
    request when the previous one finishes. Answers "how fast is it with N
    users?" — but a slow server slows the load down with it, which hides queueing.
  * **open loop** (`--rate R`): requests arrive at R per second (Poisson), whether
    or not the server keeps up. Answers "what happens at this traffic level?".
    Latency is measured from when a request was *due*, so time spent waiting for
    a free connection counts. Not doing that is "coordinated omission", and it
    is the most common way load tests report latencies nobody actually sees.

Every request streams, so time to first token (TTFT) and time per output token
(TPOT) are measured separately from end-to-end latency. The prompt mix defaults
to short/medium/long chat turns that share a system prompt, as real apps do.

    python scripts/loadtest.py --concurrency 4 --requests 40
    python scripts/loadtest.py --rate 0.5 --duration 60 --slo-ttft-p95 1500
    python scripts/loadtest.py --surface ollama --json > result.json

Exit code: 0 if every SLO given passed, 1 if any failed, 2 on a usage error —
so the same command can gate a deploy in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

SYSTEM_PROMPT = (
    "You are a concise assistant for an internal engineering team. Answer in plain "
    "language, prefer short lists, and say so when you are not sure."
)

# A long, stable block that repeats across requests — like retrieved documents or
# a policy appendix. Repetition is deliberate: it is what the prefix cache sees.
REFERENCE = " ".join(
    f"Section {i}: service {i} owns queue q{i % 11} and pages team t{i % 7} on failure."
    for i in range(90)
)

QUESTIONS = [
    "What is a KV cache, in two sentences?",
    "Give three ways to reduce time to first token.",
    "Why does decode speed depend on memory bandwidth?",
    "When is batching worth the extra latency?",
    "Explain Little's law with one example.",
    "What should a health check for a model server report?",
]

DEFAULT_MIX = {"short": 0.6, "medium": 0.3, "long": 0.1}


def build_messages(kind: str, rng: random.Random) -> list[dict[str, str]]:
    """One chat request of the given size class."""
    question = rng.choice(QUESTIONS)
    if kind == "short":
        return [{"role": "user", "content": question}]
    if kind == "medium":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nReference:\n{REFERENCE}"},
        {"role": "user", "content": f"Using the reference, which team does service {rng.randint(0, 89)} page?"},
    ]


def parse_mix(text: str) -> dict[str, float]:
    mix: dict[str, float] = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        if key.strip() not in DEFAULT_MIX:
            raise ValueError(f"unknown prompt class {key!r} (use short, medium, long)")
        mix[key.strip()] = float(value)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("the mix must have a positive weight")
    return {k: v / total for k, v in mix.items()}


@dataclass
class Result:
    """What happened to one request. Times are seconds on a monotonic clock."""

    kind: str
    due: float
    sent: float = 0.0
    first_token: float | None = None
    done: float = 0.0
    status: int = 0
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    frames: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.due

    @property
    def latency(self) -> float:
        return self.done - self.due

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first — the decode rate one user sees."""
        tokens = self.completion_tokens or self.frames
        if self.first_token is None or tokens < 2:
            return None
        return (self.done - self.first_token) / (tokens - 1)


@dataclass
class Config:
    url: str = "http://127.0.0.1:8000"
    model: str = ""
    surface: str = "openai"
    concurrency: int = 0
    rate: float = 0.0
    requests: int = 0
    duration: float = 0.0
    warmup: int = 2
    max_tokens: int = 128
    timeout: float = 120.0
    seed: int = 7
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))


async def resolve_model(client: httpx.AsyncClient, cfg: Config) -> str:
    if cfg.model:
        return cfg.model
    res = await client.get("/v1/models")
    res.raise_for_status()
    models = res.json().get("data") or []
    if not models:
        raise SystemExit("the server has no models; pull one first")
    return models[0]["id"]


async def one_request(client: httpx.AsyncClient, cfg: Config, kind: str, due: float,
                      rng: random.Random) -> Result:
    result = Result(kind=kind, due=due)
    messages = build_messages(kind, rng)
    if cfg.surface == "openai":
        path = "/v1/chat/completions"
        body = {"model": cfg.model, "messages": messages, "stream": True,
                "max_tokens": cfg.max_tokens, "temperature": 0}
    else:
        path = "/api/chat"
        body = {"model": cfg.model, "messages": messages, "stream": True,
                "options": {"num_predict": cfg.max_tokens, "temperature": 0}}

    result.sent = time.monotonic()
    try:
        async with client.stream("POST", path, json=body, timeout=cfg.timeout) as res:
            result.status = res.status_code
            if res.status_code != 200:
                await res.aread()
                result.error = res.text[:200]
            else:
                async for line in res.aiter_lines():
                    _consume(cfg.surface, line, result)
    except Exception as exc:  # one bad response is a failed request, not a failed test
        result.error = f"{type(exc).__name__}: {exc}"[:200]
    result.done = time.monotonic()
    if result.status == 200 and result.error is None and result.first_token is None:
        result.error = "stream ended without any content"
    return result


def _consume(surface: str, line: str, result: Result) -> None:
    """Fold one streamed line into the result, for either dialect."""
    if surface == "openai":
        if not line.startswith("data: ") or line == "data: [DONE]":
            return
        obj = json.loads(line[6:])
        if "error" in obj:
            err = obj["error"]
            result.error = (err.get("message") if isinstance(err, dict) else str(err))[:200]
            return
        choice = (obj.get("choices") or [{}])[0]
        if (choice.get("delta") or {}).get("content"):
            result.frames += 1
            if result.first_token is None:
                result.first_token = time.monotonic()
        usage = obj.get("usage")
        if usage:
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
        return
    if not line.strip():
        return
    obj = json.loads(line)
    if "error" in obj:
        result.error = str(obj["error"])[:200]
        return
    if (obj.get("message") or {}).get("content"):
        result.frames += 1
        if result.first_token is None:
            result.first_token = time.monotonic()
    if obj.get("done"):
        result.prompt_tokens = obj.get("prompt_eval_count", 0)
        result.completion_tokens = obj.get("eval_count", 0)


def _pick(mix: dict[str, float], rng: random.Random) -> str:
    return rng.choices(list(mix), weights=list(mix.values()))[0]


async def run(cfg: Config, client: httpx.AsyncClient | None = None) -> tuple[list[Result], float]:
    """Apply the load and return (measured results, wall seconds of the measured phase)."""
    own = client is None
    if own:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        client = httpx.AsyncClient(base_url=cfg.url, timeout=cfg.timeout, limits=limits)
    rng = random.Random(cfg.seed)
    try:
        cfg.model = await resolve_model(client, cfg)
        for _ in range(cfg.warmup):  # page in weights, fill the prefix cache
            await one_request(client, cfg, "medium", time.monotonic(), rng)

        results: list[Result] = []
        start = time.monotonic()

        def more(issued: int) -> bool:
            if cfg.requests and issued >= cfg.requests:
                return False
            if cfg.duration and time.monotonic() - start >= cfg.duration:
                return False
            return True

        if cfg.rate > 0:  # open loop
            tasks: list[asyncio.Task] = []
            due = start
            while more(len(tasks)):
                due += rng.expovariate(cfg.rate)
                # Check the window against the arrival time *before* sleeping, or a
                # long gap would stretch the test past --duration with nothing sent.
                if cfg.duration and due - start >= cfg.duration:
                    break
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(
                    one_request(client, cfg, _pick(cfg.mix, rng), due, rng)))
            results = list(await asyncio.gather(*tasks))
        else:  # closed loop
            issued = 0

            async def user() -> None:
                nonlocal issued
                while more(issued):
                    issued += 1
                    results.append(await one_request(
                        client, cfg, _pick(cfg.mix, rng), time.monotonic(), rng))

            await asyncio.gather(*(user() for _ in range(cfg.concurrency)))
        return results, time.monotonic() - start
    finally:
        if own:
            await client.aclose()


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * p // 100))  # ceil without math
    return ordered[int(rank) - 1]


def summarise(results: list[Result], wall: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    statuses: dict[str, int] = {}
    for r in results:
        key = str(r.status) if r.status else "network"
        if r.status == 200 and r.error:
            key = "200-error"
        statuses[key] = statuses.get(key, 0) + 1

    def dist(values: list[float], scale: float = 1000.0) -> dict[str, float | None]:
        return {
            f"p{p}": (None if (v := percentile(values, p)) is None else round(v * scale, 1))
            for p in (50, 90, 95, 99)
        }

    out_tokens = sum(r.completion_tokens or r.frames for r in ok)
    tpots = [t for r in ok if (t := r.tpot) is not None]
    return {
        "requests": len(results),
        "succeeded": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else 0.0,
        "statuses": statuses,
        "wall_s": round(wall, 2),
        "throughput_rps": round(len(ok) / wall, 3) if wall else 0.0,
        "output_tok_per_s": round(out_tokens / wall, 1) if wall else 0.0,
        "ttft_ms": dist([r.ttft for r in ok if r.ttft is not None]),
        "latency_ms": dist([r.latency for r in ok]),
        "tpot_ms": dist(tpots),
        "per_stream_tok_per_s_p50": (
            round(1 / percentile(tpots, 50), 1) if tpots and percentile(tpots, 50) else None
        ),
        "by_kind": {
            kind: {
                "n": len(rs),
                "ttft_p50_ms": (
                    None if (v := percentile([r.ttft for r in rs if r.ttft is not None], 50)) is None
                    else round(v * 1000, 1)
                ),
                "prompt_tokens_p50": percentile([r.prompt_tokens for r in rs], 50),
            }
            for kind in ("short", "medium", "long")
            if (rs := [r for r in ok if r.kind == kind])
        },
        "first_errors": [r.error for r in results if r.error][:3],
    }


def check_slos(summary: dict[str, Any], ttft_p95_ms: float | None,
               error_rate: float | None) -> list[str]:
    """Human-readable SLO failures; empty means every SLO given passed."""
    failures = []
    if (ttft_p95_ms is not None or error_rate is not None) and not summary.get("requests"):
        return ["no requests were measured"]
    if ttft_p95_ms is not None:
        got = summary["ttft_ms"]["p95"]
        if got is None or got > ttft_p95_ms:
            failures.append(f"TTFT p95 {got} ms > {ttft_p95_ms} ms")
    if error_rate is not None and summary["error_rate"] > error_rate:
        failures.append(f"error rate {summary['error_rate']:.2%} > {error_rate:.2%}")
    return failures


def render(summary: dict[str, Any], cfg: Config, failures: list[str]) -> str:
    mode = f"open loop, {cfg.rate} req/s" if cfg.rate else f"closed loop, {cfg.concurrency} users"
    t, e, d = summary["ttft_ms"], summary["latency_ms"], summary["tpot_ms"]
    lines = [
        f"shilp load test — {cfg.surface} surface, model {cfg.model}, {mode}",
        f"  requests      {summary['succeeded']}/{summary['requests']} ok   statuses {summary['statuses']}",
        f"  throughput    {summary['throughput_rps']} req/s, {summary['output_tok_per_s']} output tok/s over {summary['wall_s']} s",
        f"  TTFT  ms      p50 {t['p50']}  p90 {t['p90']}  p95 {t['p95']}  p99 {t['p99']}",
        f"  e2e   ms      p50 {e['p50']}  p90 {e['p90']}  p95 {e['p95']}  p99 {e['p99']}",
        f"  TPOT  ms      p50 {d['p50']}  p95 {d['p95']}   (≈ {summary['per_stream_tok_per_s_p50']} tok/s per stream)",
    ]
    for kind, row in summary["by_kind"].items():
        lines.append(f"  {kind:<7}       n={row['n']}  TTFT p50 {row['ttft_p50_ms']} ms  prompt ≈ {row['prompt_tokens_p50']} tokens")
    if summary["first_errors"]:
        lines.append(f"  errors        {summary['first_errors']}")
    lines.append("  SLO           " + ("FAIL: " + "; ".join(failures) if failures else "pass"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="", help="default: the first model the server lists")
    parser.add_argument("--surface", choices=("openai", "ollama"), default="openai")
    load = parser.add_mutually_exclusive_group()
    load.add_argument("--concurrency", type=int, help="closed loop: simulated users")
    load.add_argument("--rate", type=float, help="open loop: arrivals per second (Poisson)")
    parser.add_argument("--requests", type=int, default=0, help="stop after this many")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds")
    parser.add_argument("--warmup", type=int, default=2, help="unmeasured requests first")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--mix", default="short=0.6,medium=0.3,long=0.1")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slo-ttft-p95", type=float, help="fail if p95 TTFT exceeds this (ms)")
    parser.add_argument("--slo-error-rate", type=float, help="fail if the error rate exceeds this (0–1)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.rate is not None and args.rate <= 0:
        parser.error("--rate must be positive")
    if args.requests < 0 or args.duration < 0 or args.warmup < 0:
        parser.error("--requests, --duration and --warmup cannot be negative")
    if not args.requests and not args.duration:
        args.requests = 20
    try:
        cfg = Config(
            url=args.url, model=args.model, surface=args.surface,
            concurrency=args.concurrency or (0 if args.rate else 1), rate=args.rate or 0.0,
            requests=args.requests, duration=args.duration, warmup=args.warmup,
            max_tokens=args.max_tokens, timeout=args.timeout, seed=args.seed,
            mix=parse_mix(args.mix),
        )
    except ValueError as exc:
        parser.error(str(exc))

    results, wall = asyncio.run(run(cfg))
    summary = summarise(results, wall)
    failures = check_slos(summary, args.slo_ttft_p95, args.slo_error_rate)
    if args.json:
        print(json.dumps({"config": {k: v for k, v in asdict(cfg).items()}, "summary": summary,
                          "slo_failures": failures}, indent=2))
    else:
        print(render(summary, cfg, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
