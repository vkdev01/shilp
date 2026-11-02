# 06 — Metering and billing

**Status:** 🟡 one usage row per request that reaches generation; no key, no aggregation, no prices · **Blocks:** quotas, charging, cost per team · **Depends on:** [01 Authentication](01-authentication.md) for the "who" · **Roadmap:** P2.3 (metering per tenant), P3.3 (shared across replicas), P4.1 (billing) · **Effort:** 3–4 days for metering, about a week more for billing

---

## 1. The problem

shilp already writes one row per request into `$SHILP_HOME/logs/shilp.db` (`server/telemetry.py`):

| Recorded today | Missing for metering |
|---|---|
| timestamp, surface, endpoint, model | **who**: key id, tenant id |
| status (200, 400, 499, 503, 500) — the client-visible status, except that a failure *after* streaming has started is recorded as 500 while the client got 200 plus an error frame, and 499 is shilp's own marker for a client that left | **event id** (server-owned, for deduplication) and **request id** (for support) |
| prompt tokens (exact: rendered prompt incl. template) | **cached** prompt tokens (cheaper to serve, often priced lower) |
| completion tokens (re-tokenized from text: close, not exact) | exact generated-token count, finish reason, model **digest** (which exact version ran) |
| time to first token, total time, tok/s | rows for requests rejected **before** generation (404, 422, future 401/429) |
| — | aggregation per tenant × model × day, a price book, invoices, exports |

And the table is per process: with replicas, there are N partial ledgers.

That is enough for capacity planning and a per-model cost estimate. It is not enough to answer *"how much did team Acme use in October, and what do they owe?"*

---

## 2. Concepts

### Metering is not logging

| | Logs / metrics | Meter |
|---|---|---|
| Purpose | debugging, dashboards | money, quotas, contracts |
| May be sampled or dropped | yes | **no** |
| Aggregation | pre-aggregated, lossy (counters, summaries) | raw events kept, aggregates rebuilt from them |
| Correctness bar | approximately right | exactly right, auditable, reproducible |

Prometheus counters (`shilp_tokens_total`) are useful for *reconciling* the meter — they should agree within a small tolerance — but a counter that resets on restart cannot be a ledger.

### The usage event

One record per request — opened when the request is admitted, finalised when it ends:

| Field | Why |
|---|---|
| `event_id` | **server-generated**, immutable: the deduplication key — a record delivered twice is counted once |
| `request_id` | the client's `X-Request-ID` (or a generated one), for support and tracing — **never** used for deduplication, because a client could reuse one to make later usage disappear |
| `tenant_id`, `key_id` | who pays, and which key (for rotation and abuse) |
| `model`, `model_digest` | the price depends on the model; the digest proves which version ran |
| `prompt_tokens`, `cached_prompt_tokens`, `completion_tokens` | the three billable quantities |
| `http_status`, `outcome`, `finish_reason` | what the client was sent, what actually happened (`completed`, `client_closed`, `failed_midstream`, `rejected`, `crashed`), and why the answer ended — billing policy (below) keys off `outcome` |
| `started_at`, `ended_at`, `ttft_ms` | billing period, SLO credits, disputes |
| `replica` | debugging and reconciliation across replicas |

No prompt or answer text — ever. The meter needs counts, not content ([07](07-multi-tenancy-and-abuse.md)).

### Exactly-once from at-least-once

Networks and processes fail, so delivering an event exactly once is not achievable directly. It is achieved by **at-least-once delivery plus deduplication**: every step may retry, and every store ignores an `event_id` it has already seen (`INSERT … ON CONFLICT (event_id) DO NOTHING`).

Delivery only starts once the event exists, so the first write must be durable too. An event written only when the request ends is lost if the process is killed mid-request (an out-of-memory kill, `SIGKILL`). The fix is to **open** the event at admission — a durable `pending` row with the estimated tokens — and **finalise** it at the end. On start-up, any `pending` rows left by a previous process are closed with `outcome = crashed` and billed by policy (usually not at all), so nothing silently disappears. The same rule makes re-running an aggregation safe.

### Aggregation by recomputation

Rather than incrementing totals as events arrive (fragile under retries), recompute each **closed** period from the raw events:

```sql
DELETE FROM usage_daily WHERE day = :day;
INSERT INTO usage_daily (day, tenant_id, model, requests, prompt_tokens,
                         cached_prompt_tokens, completion_tokens, errors)
SELECT date(ts, 'unixepoch'), tenant_id, model, COUNT(*),
       SUM(prompt_tokens), SUM(COALESCE(cached_prompt_tokens, 0)), SUM(completion_tokens),
       SUM(status >= 500)
FROM requests                      -- shilp's existing per-request table, extended in §4.2
WHERE date(ts, 'unixepoch') = :day
GROUP BY 1, 2, 3;
```

Running it twice gives the same answer. Late events (a replica that was offline) are absorbed by re-running the affected day.

### Billing policy for unusual outcomes

Decide and write down, because each one shows up in practice:

| Outcome (`outcome` / `http_status`) | Suggested policy |
|---|---|
| `completed` / 200 | bill prompt + completion tokens |
| `client_closed` / recorded 499 | bill prompt + tokens actually generated (work was done) |
| `rejected` / 4xx before generation (401, 404, 422, 429, 400 context overflow) | free |
| `failed_midstream` / client saw 200 + an error frame | bill tokens generated, or waive — pick one and be consistent |
| 5xx before the first token | free; count against the SLO |
| `crashed` (process died mid-request) | free; count against the SLO |

### Why tokens are priced the way they are

The physics in [04](04-batching-and-concurrency.md) shows up directly in prices:

- **Output costs more than input.** **Measured** on one stream: prefill ran at ~2,000 tok/s and decode at ~98.7 tok/s, so an output token took ~20× the device time of an input token. Batching narrows the gap at scale, which is why typical price ratios are smaller (often 3–5×), but the direction comes from this.
- **Cached input is discounted.** A prefix-cache hit skips prefill: 823 ms → 43 ms to first token on a 1.7k-token prefix (measured).
- **Long context costs more** at the extreme, because attention grows quadratically past tens of thousands of tokens.

### Cost: what one token costs you

```
cost per 1M output tokens = machine $/hour ÷ (output tokens per second × 3600 ÷ 1,000,000)
```

**Computed** with the measured serial throughput of ~87 output tok/s (one M3 Pro, Gemma-3-1B):

| Machine cost | $ per 1M output tokens (serial) |
|---|---|
| $0.05 / hour | $0.16 |
| $0.10 / hour | $0.32 |
| $0.50 / hour | $1.60 |
| $1.00 / hour | $3.19 |

Batching divides these by its throughput gain ([04](04-batching-and-concurrency.md) estimates several ×). These figures are the floor for pricing — and, for an internal service, the basis for **showback** (reporting cost per team) or **chargeback** (actually billing teams).

---

## 3. Design options

| Option | Fits |
|---|---|
| **SQLite events + aggregates, per process** | one replica; zero new dependencies |
| **Postgres as the shared ledger** | replicas; SQL aggregation; easy exports |
| **Event log (Kafka, Kinesis, files in object storage) → warehouse** | high volume; analytics alongside billing |
| **A billing platform** (Stripe Billing usage records, Lago, Orb, OpenMeter) | invoices, payments, tax, dunning — do not build these yourself |

**Recommendation:** SQLite events now (it is what exists), Postgres when there are replicas, and an external billing platform fed from the daily aggregates when money changes hands.

---

## 4. The design for shilp

### 4.1 Data flow

```
request admitted ──► INSERT pending row (event_id, request_id, key, tenant, estimate)   ← durable before work starts
       │                                                                                    rejections are written as final rows
request ends (finally in server/generate.py) ──► UPDATE row: outcome, tokens, timings  (local SQLite, event_id unique)
start-up ──► pending rows from a dead process → outcome = crashed
                                                     │  shipper: batches, retries, at-least-once
                                                     ▼
                                       shared usage_events (Postgres, ON CONFLICT DO NOTHING)
                                                     │  hourly job: recompute closed periods
                                                     ▼
                                       usage_daily (tenant × model × day)
                                     ┌───────────────┼────────────────────┐
                                     ▼               ▼                    ▼
                              quotas ([02])   price book → invoices   shilp usage / GET /v1/usage
```

### 4.2 Schema changes

Extend the existing `requests` table — it already is a usage-event log — rather than adding a second one. Its existing `ts` column is the event time. SQLite has no `ADD COLUMN IF NOT EXISTS`, so migrations are **versioned**: read `PRAGMA user_version`, apply each numbered migration above it inside one transaction, then set the new version — running start-up twice is then harmless. Migration 1:

```sql
ALTER TABLE requests ADD COLUMN event_id TEXT;
ALTER TABLE requests ADD COLUMN request_id TEXT;
ALTER TABLE requests ADD COLUMN outcome TEXT;            -- completed, client_closed, failed_midstream, rejected, crashed, pending
ALTER TABLE requests ADD COLUMN tenant_id TEXT;
ALTER TABLE requests ADD COLUMN key_id TEXT;
ALTER TABLE requests ADD COLUMN model_digest TEXT;
ALTER TABLE requests ADD COLUMN cached_prompt_tokens INTEGER;
ALTER TABLE requests ADD COLUMN finish_reason TEXT;
ALTER TABLE requests ADD COLUMN replica TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_requests_event_id ON requests(event_id);
CREATE INDEX IF NOT EXISTS idx_requests_request_id ON requests(request_id);
PRAGMA user_version = 1;
CREATE INDEX IF NOT EXISTS idx_requests_tenant_ts ON requests(tenant_id, ts);
```

Plus `usage_daily` and a versioned `price_book`:

```sql
CREATE TABLE price_book (
  model TEXT, effective_from TEXT, currency TEXT,
  input_per_m REAL, cached_input_per_m REAL, output_per_m REAL,
  PRIMARY KEY (model, effective_from)
);
```

Prices are looked up by the model and the date of usage, so changing a price never rewrites history.

### 4.3 Where the fields come from

| Field | Source in shilp |
|---|---|
| `event_id` | generated by shilp at admission (a UUID); never taken from the client |
| `request_id` | `X-Request-ID` header or generated (P1.5) |
| `tenant_id`, `key_id` | the `Principal` from authentication ([01](01-authentication.md)) |
| `model_digest` | the manifest digest already returned by `/api/tags` |
| `prompt_tokens` | today's rendered-prompt count |
| `completion_tokens` | today: re-tokenized text; with upstream engines, the engine's own count ([04](04-batching-and-concurrency.md)) |
| `cached_prompt_tokens` | from engines that report it (llama-server timings, vLLM `cached_tokens`); `NULL` where unknown — never guessed |
| `finish_reason` | `stop` or `length` once P1.6 passes it through |

### 4.4 Worked invoice

A tenant in one month on a model priced (example prices) at **$0.10 per 1M input tokens, $0.025 per 1M cached input tokens, $0.40 per 1M output tokens**:

| Quantity | Tokens | Rate | Amount |
|---|---|---|---|
| Prompt tokens, uncached | 40,000,000 | $0.10 / 1M | $4.00 |
| Prompt tokens, cached | 80,000,000 | $0.025 / 1M | $2.00 |
| Completion tokens | 30,000,000 | $0.40 / 1M | $12.00 |
| **Total** | 150,000,000 | | **$18.00** |

The same aggregates, multiplied by your *cost* per token instead of a price, give the cost of serving that tenant — which tells you whether the price is above cost.

### 4.5 Interfaces

```bash
shilp usage --tenant acme --since 2026-10-01 --until 2026-11-01 --group-by day,model
shilp usage --since 2026-10-01 --format csv > october.csv        # for finance or a billing platform
```

`GET /v1/usage?start=…&end=…` returns the **caller's own** tenant usage (scope `usage:read`); admins can query any tenant. Today's `GET /api/usage` (an all-traffic summary) becomes admin-only once keys exist.

### 4.6 Reconciliation

A daily check compares the meter with independent counters:

- Σ `completion_tokens` in the events vs the increase of `shilp_tokens_total{direction="completion"}` over the same window, per replica — alert if they differ by more than 1%.
- Events with a `request_id` but no matching access-log line, and vice versa.

---

## 5. Step-by-step implementation

| Step | Change | Files |
|---|---|---|
| 1 | Request id: accept `X-Request-ID` or generate; return it; put it on the record | routes, `server/telemetry.py` (P1.5) |
| 2 | Create the record on arrival so rejections are recorded | `server/generate.py`, routes, `server/app.py` handlers |
| 3 | Additive migration: new columns and indexes | `server/telemetry.py` |
| 4 | Tenant, key, digest, finish reason on the record | `server/auth.py` ([01](01-authentication.md)), `server/generate.py` |
| 5 | `usage_daily` recomputation job (`shilp usage rollup`, or on a timer in the server) | `server/telemetry.py`, `cli.py` |
| 6 | `shilp usage` filters, grouping, CSV/JSON export; `GET /v1/usage` scoped to the caller | `cli.py`, `routes_openai.py` |
| 7 | Quotas read the aggregates ([02 §4.6](02-rate-limiting-and-quotas.md)) | `server/ratelimit.py` |
| 8 (P3.3) | Shipper to Postgres with retries and `ON CONFLICT DO NOTHING` | `server/usage_shipper.py` (new) |
| 9 (P4.1) | Price book, invoice export, or a feed into a billing platform | `billing/` (new) |

---

## 6. Testing and verification

- Every request — including 401, 404, 422, 429 — produces exactly one event with an event id and a request id.
- Inserting the same event twice leaves one row; re-running a day's aggregation gives identical totals.
- Two requests sent with the **same** `X-Request-ID` produce two billable events.
- Killing the process mid-request leaves a `pending` row that the next start-up closes as `crashed`.
- Running the migrations twice changes nothing (`PRAGMA user_version`).
- A 499 records the tokens generated before the disconnect; a 400 context overflow records zero completion tokens.
- The worked invoice in §4.4 is reproduced from fixture events and a fixture price book.
- Load test: the load generator's own token totals and the meter's agree within 1%.
- A price change dated in the middle of a month bills each part at its own rate.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Event lost on crash | durable `pending` row at admission, finalised in `finally`; `crashed` recovery at start-up; ship from the local table, not from memory |
| A client reuses a request id | deduplication uses the server's `event_id`, never the client's id |
| Duplicate events after retries | unique `request_id`; idempotent aggregation |
| Replica offline during the rollup | recompute closed periods again when it reports; invoices close a few days after period end |
| Clock skew between replicas | timestamps from each replica are fine for daily buckets; use the shared store's time for cut-off decisions |
| Price changes | versioned price book keyed by effective date |
| Disputes | raw events kept for the retention period; digest proves the model version |

---

## 8. Status

| Item | Status |
|---|---|
| One row per request that reaches generation, with true status and tokens | ✅ |
| Prometheus token and request counters (for reconciliation) | ✅ |
| No prompt text stored | ✅ |
| Request id, server event id, rows for rejections before generation | ⬜ P1.5 |
| Durable pending row at admission; crash recovery; `http_status` vs `outcome` | ⬜ P2.3 |
| Tenant / key on events | ⬜ P2.3 (needs [01](01-authentication.md)) |
| Cached-token count; exact completion count | ⬜ P1.6 / with upstream engines (P3.1) |
| Daily aggregates, CLI filters, CSV export, `GET /v1/usage` | ⬜ P2.3 |
| Shared ledger across replicas | ⬜ P3.3 |
| Price book, invoices, billing integration | ⬜ P4.1 |
