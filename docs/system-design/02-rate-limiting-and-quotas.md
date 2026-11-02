# 02 — Rate limiting and quotas

**Status:** ⬜ not implemented · **Blocks:** more than one user sharing a server · **Depends on:** [01 Authentication](01-authentication.md) · **Roadmap:** P1.3 (request limits), P2.2 (rate limits), P2.5 (quotas) · **Effort:** 2–3 days (in-memory + Redis), +1 day for quotas once metering exists

---

## 1. The problem

One shilp process serves about **0.81 requests per second** (measured, one stream at a time, ~1.2 s per answer). A single client that sends its next request as soon as the last one finishes therefore uses **100% of the machine**, and everyone else waits behind it. There is no malice required: a retry loop, a batch script or an agent stuck in a tool-call loop does exactly this.

Requests are also not equal. A 20-token question with a 50-token answer and a 30,000-token document with `max_tokens: 4096` differ in cost by roughly 100–1000×. Counting requests alone lets one client consume the box with a handful of huge calls.

Today the only per-request limit is the context window (a prompt over `num_ctx` is rejected with 400). A client may set `max_tokens` up to the whole context, and there is no limit on how often anyone may call.

---

## 2. Concepts

### Three different controls

| Control | Protects | Time scale | Example |
|---|---|---|---|
| **Rate limit** | the server and other users, from bursts | seconds to a minute | 30 requests/min, 20,000 tokens/min per key |
| **Concurrency limit** | the server, from one client holding many slots | instantaneous | at most 2 requests in flight per key |
| **Quota** | the budget, from runaway spend | day or month | 5 M tokens per month per tenant |

A queue ([03](03-admission-control.md)) protects the server from *total* load; rate limits protect it from *one* client. You need both.

### The token bucket

The most common rate-limit algorithm, and the one this design uses:

- A bucket holds up to **C** tokens (the *burst*).
- It refills at **r** tokens per second, never above C.
- A request takes its cost from the bucket; if there is not enough, it is rejected, and the client is told how long until there will be.

Worked example — a key limited to **10 requests per minute with a burst of 5** (C = 5, r = 1/6 per second). The client fires 8 requests at once at t = 0, then one more at t = 7 s:

| Time | Request | Bucket before | Result | Bucket after |
|---|---|---|---|---|
| 0 s | 1–5 | 5 | allowed | 0 |
| 0 s | 6, 7, 8 | 0 | **429**, `Retry-After: 6` | 0 |
| 7 s | 9 | 0 + 7 × (1/6) = 1.17 | allowed | 0.17 |

The burst absorbs a client that opens a page and fires several calls at once; the refill rate caps the sustained rate at 10/min.

Other algorithms, for comparison:

| Algorithm | Behaviour | Why not the default here |
|---|---|---|
| Fixed window ("100 per calendar minute") | simple counter per minute | allows 2× the limit across a minute boundary |
| Sliding window | exact count over the last 60 s | more state; no burst control |
| Leaky bucket (as a queue) | smooths to a constant rate by delaying | adds latency instead of rejecting; the queue in [03](03-admission-control.md) already does that job |

### Counting tokens, not just requests

LLM APIs limit **requests per minute (RPM)** *and* **tokens per minute (TPM)**. The difficulty: output length is not known when the request arrives. The standard answer is **reserve and refund**:

1. At admission, reserve `estimated_prompt_tokens + max_tokens` from the TPM bucket.
2. When the request ends, compute the actual `prompt_tokens + completion_tokens` and put the difference back.

Example — a tenant with **60,000 TPM**. A request has a ~1,500-token prompt and `max_tokens: 512`:

| Step | Bucket |
|---|---|
| Before | 60,000 |
| Reserve 1,500 + 512 = 2,012 | 57,988 |
| Answer was 300 tokens; actual cost 1,800; refund 212 | 58,200 |

Without the refund, a client that always asks for `max_tokens: 4096` with a 50-token prompt and gets 50-token answers would be charged 4,146 tokens a call — and would hit the limit after 14 calls instead of the ~600 its real usage (100 tokens a call) allows. Without the *reservation*, a client could start dozens of long generations at once before any of them was counted.

The prompt estimate at admission does not need to be exact: `len(text) / 4` plus a few tokens per message is close for English and is corrected at settlement with the real count shilp already computes (the rendered prompt, chat template included).

### Sizing limits from capacity

Limits are a share of what the box can do. **Measured** capacity for the reference setup: ~49 requests/min and ~5,200 output tokens/min on one serial process.

- **No single key should exceed ~50% of capacity**, so one client can never lock everyone out: ~25 RPM, ~2,500 output tokens/min.
- **The sum of all keys' limits may exceed capacity** (overbooking), because not everyone is active at once. Ten engineers at 15 RPM each is 150 RPM of limits against 49 of capacity — 3× overbooked, fine for a team; the queue and SLO alerts tell you when it stops being fine.
- **On a serial server, cap in-flight requests per key at 1–2.** A second concurrent request from the same key cannot run any faster; it only takes someone else's turn.

### Telling the client

```
HTTP/1.1 429 Too Many Requests
Retry-After: 6
x-ratelimit-limit-requests: 10
x-ratelimit-remaining-requests: 0
x-ratelimit-reset-requests: 6s
x-ratelimit-limit-tokens: 60000
x-ratelimit-remaining-tokens: 57988
x-ratelimit-reset-tokens: 2s

{"error": {"message": "Rate limit reached for requests: 10 per minute. Retry in 6s.",
           "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
```

These are the header names OpenAI's API uses, so the official SDKs' retry logic (which honours `Retry-After` and backs off on 429) works unchanged. Quota exhaustion uses the same status with `code: "insufficient_quota"` and a `Retry-After` pointing at the period reset — clients should not hammer it.

### Across replicas

One process can keep buckets in memory. With several replicas, counters must be shared or the effective limit becomes N × the configured one. Two approaches:

| Approach | Accuracy | Complexity |
|---|---|---|
| **Shared store (Redis)** — every replica updates the same bucket atomically | exact | one more dependency; ~0.1–0.5 ms per call on a local network — negligible next to a ~1,200 ms request |
| **Split the budget** — each of N replicas enforces limit / N locally | approximate; good when the load balancer spreads a key's traffic evenly | none |

---

## 3. Design options

| Option | Fits |
|---|---|
| **A. At the proxy/gateway** (nginx `limit_req`, Envoy local/global rate limit, a cloud gateway) | request-count limits per IP or per key header; cannot do token-based limits because it cannot see token counts |
| **B. In shilp, in memory** | single process; RPM + TPM + in-flight with reserve/refund |
| **C. In shilp, Redis-backed** | replicas; the same interface as B |

**Recommendation:** B behind an interface, with C as a drop-in for replicas. Keep A as a coarse outer limit (per IP) against floods that never authenticate.

---

## 4. The design for shilp

### 4.1 Where it runs

```
authenticate ─► request limits (P1.3) ─► rate limiter: reserve ─► admission queue ─► generate
                 body size, messages,       RPM, TPM, in-flight        ([03])            │
                 max_tokens ceiling                                                       ▼
                                           rate limiter: settle ◄──── telemetry record (finally)
```

Settlement hooks into the existing `finally` in `server/generate.py`, where `_record()` already has the real `prompt_tokens` and `completion_tokens` on the `RequestRecord` — so refunds are exact, and abandoned (499) requests are charged only for what they generated.

### 4.2 Request limits first (P1.3)

Before any bucket, cap what one request may ask for:

| Limit | Setting | Example default | Response |
|---|---|---|---|
| Body size | `SHILP_MAX_BODY_BYTES` | 1 MiB | 413 |
| Messages per request | `SHILP_MAX_MESSAGES` | 256 | 400 |
| Output tokens | `SHILP_MAX_OUTPUT_TOKENS` | 1,024 | clamp `max_tokens` / `num_predict` (or 400 if the client asked for more — a policy choice) |
| Generation deadline | `SHILP_REQUEST_TIMEOUT` | 120 s | stop generation through the existing cancellation flag; error frame or `finish_reason: "length"` — **checked between tokens**, so it cannot interrupt a prefill already running; a hard deadline needs the engine's abort callback or, with an upstream engine ([04](04-batching-and-concurrency.md)), closing the upstream connection. The prompt-size cap is what bounds prefill time |

These bound the **worst case** of a single request, which is what makes token-based rate limits meaningful.

### 4.3 The limiter interface

New module `src/shilp/server/ratelimit.py`:

```python
@dataclass(frozen=True)
class Limits:
    rpm: int | None = None          # requests per minute
    tpm: int | None = None          # tokens per minute
    burst: float = 1.0              # bucket capacity as minutes of rate (1.0 = one minute's worth)
    max_in_flight: int | None = None

@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: float = 0.0        # seconds
    headers: dict[str, str] = field(default_factory=dict)

class RateLimiter(Protocol):
    async def reserve(self, key: str, limits: Limits, tokens: int, request_id: str) -> Decision: ...
    async def settle(self, key: str, limits: Limits, reserved: int, actual: int, request_id: str) -> None: ...
```

`MemoryLimiter` keeps one bucket per (key, kind) in a dict, guarded by an `asyncio.Lock`:

```python
def refill(bucket: Bucket, rate_per_s: float, capacity: float, now: float) -> None:
    bucket.tokens = min(capacity, bucket.tokens + (now - bucket.updated) * rate_per_s)
    bucket.updated = now
```

`reserve()` refills both buckets, checks `tokens >= 1` (requests) and `tokens >= cost` (tokens) and the in-flight count, and either takes from all three or none. `settle()` returns `reserved - actual` to the token bucket (never above capacity) and decrements in-flight.

### 4.4 Redis implementation

**One Lua script per admission decision**, covering *every* limit at once — the request bucket, the token bucket and the in-flight lease set — so a request rejected by one limit never consumes another. The script refills both buckets, checks all three conditions, and only if all pass takes from both buckets and adds the lease. It uses Redis's own clock to avoid skew between hosts. The core of the bucket arithmetic, for one bucket:

```lua
-- KEYS[1] = bucket hash, ARGV = rate_per_s, capacity, cost
local t = redis.call('TIME'); local now = t[1] + t[2] / 1e6
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[2])
local ts = tonumber(b[2]) or now
tokens = math.min(tonumber(ARGV[2]), tokens + (now - ts) * tonumber(ARGV[1]))
local allowed = tokens >= tonumber(ARGV[3])
if allowed then tokens = tokens - tonumber(ARGV[3]) end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], 3600)
return {allowed and 1 or 0, tostring(tokens)}
```

The full script applies this to `rl:{key_id}:req` and `rl:{key_id}:tok` (passed together in `KEYS`), computes both decisions **without writing**, checks the in-flight lease set `inflight:{key_id}`, and writes all three only if all three allow. Leases are a sorted set scored by expiry time: the script removes expired members (`ZREMRANGEBYSCORE`), checks `ZCARD` against the cap, and `ZADD`s the event id with `now + deadline`; settlement `ZREM`s it. If a replica crashes mid-request, its leases expire instead of leaking forever — a plain counter would.

**If Redis is unavailable**: fall back to a `MemoryLimiter` enforcing `limit / replicas` and raise an alert. Failing *open* keeps the service up; failing *closed* would turn a Redis outage into a full outage.

### 4.5 Where limits come from

- **P2.2 (with per-tenant keys):** limits live on the tenant or key record in the key store ([01 §4.5](01-authentication.md)), e.g. `rpm=30, tpm=20000, max_in_flight=2`.
- **Before that:** one set of defaults from settings — `SHILP_RATE_RPM`, `SHILP_RATE_TPM`, `SHILP_MAX_IN_FLIGHT_PER_KEY` — applied to the single key, which mainly protects against one runaway script.

### 4.6 Quotas (P2.5)

Quotas are long-period totals, so they come from metering ([06](06-metering-and-billing.md)) rather than buckets:

1. At admission, compare the tenant's usage this period plus this request's reservation against the quota. Usage this period = **closed days** from the `usage_daily` aggregate + **today so far**, which the aggregate does not yet contain. Today's figure comes from a live counter — incremented at settlement (in Redis with replicas, in memory otherwise) — or from recomputing the open day on each ~10-second refresh. Outstanding reservations are added too, so many concurrent requests cannot all slip under the line.
2. Over the limit → 429 `insufficient_quota`, `Retry-After` = seconds until the period resets.
3. Soft thresholds (80%, 100%) emit events for alerts or e-mails.

A 10-second cache means a tenant can overshoot by about 10 seconds of their own traffic — a deliberate trade for not querying the usage store on every request.

---

## 5. Step-by-step implementation

| Step | Change | Files |
|---|---|---|
| 1 | Request limits: body size middleware, message count and `max_tokens` ceiling in the routes, deadline via the stop flag | `server/app.py`, `routes_*.py`, `server/stream.py`, `config.py` |
| 2 | `Limits`, `Decision`, `RateLimiter`, `MemoryLimiter` with an injectable clock | `server/ratelimit.py` (new) |
| 3 | Cheap prompt estimate (`len(text)//4 + 4 per message`) | `server/ratelimit.py` |
| 4 | Call `reserve()` in each generating route after authentication; on reject return 429 with headers | `routes_openai.py`, `routes_ollama.py` |
| 5 | Call `settle()` from the `finally` in `stream_chat` / `stream_generate`, using `RequestRecord` tokens | `server/generate.py` |
| 6 | Rate-limit headers on successful responses too (remaining, reset) | routes |
| 7 | Metrics: `shilp_ratelimited_total{key,kind}` | `server/generate.py` / `metrics.py` |
| 8 | `RedisLimiter` behind an optional `redis` extra; fallback to memory | `server/ratelimit.py`, `pyproject.toml` |
| 9 | Quotas from metering aggregates | after [06](06-metering-and-billing.md) |

---

## 6. Testing and verification

- **Bucket arithmetic with a fake clock**: the burst table in §2 reproduced exactly.
- **Reserve and refund**: the 60,000-TPM example leaves 58,200 tokens.
- **Abandoned request**: a 499 is charged for its generated tokens only.
- **429 shape**: status, `Retry-After`, `x-ratelimit-*` headers, and the OpenAI error body; the official SDK raises `RateLimitError`.
- **In-flight cap**: with `max_in_flight=1`, a second concurrent request from the same key gets 429 while a request from another key is admitted.
- **Redis**: the same test suite against a Redis instance (or `fakeredis`), plus lease expiry after a simulated crash.
- **Load test**: `scripts/loadtest.py --rate` above a key's limit shows the expected 429 share and stable latency for admitted requests.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Estimate far below real prompt size (non-English text, code) | settlement corrects the bucket; a request is never *stopped* mid-way for exceeding its estimate |
| Many replicas, uneven traffic, split budgets | use Redis; or accept approximate limits |
| Clock skew between replicas | Redis `TIME` as the single clock |
| A client ignores `Retry-After` and hammers | the outer proxy limit per IP; temporary key suspension for repeated violations ([07](07-multi-tenancy-and-abuse.md)) |
| Retry storms after an outage | jittered `Retry-After`; clients use exponential backoff |
| Limits set above capacity | the queue and SLO alerts still protect latency; review limits against measured capacity |

---

## 8. Status

| Item | Status |
|---|---|
| Context-window check (400) | ✅ |
| Body size, message count, `max_tokens` ceiling, deadline | ⬜ P1.3 |
| RPM / TPM / in-flight per key, in memory | ⬜ P2.2 (a single-key version can ship with P1) |
| Redis-backed limiter for replicas | ⬜ P2.2 / P3.3 |
| 429 with `Retry-After` and `x-ratelimit-*` headers | ⬜ P2.2 |
| Quotas and budget alerts | ⬜ P2.5 (needs [06](06-metering-and-billing.md)) |
