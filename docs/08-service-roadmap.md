# 08 — Service roadmap: from local server to inference service

**What is implemented, what is partial, and what is missing for shilp to run as a real inference service — with the order to build it in and how to know each step is done.**

This document tracks *service readiness*: everything between "a model answers on localhost" and "other people's products depend on this endpoint". The [project roadmap](04-roadmap.md) covers the rest of shilp (evaluation, tool calling, sharded GGUFs, surgery, fine-tuning). Concepts are explained in [06-service-fundamentals.md](06-service-fundamentals.md); operations in [07-operate.md](07-operate.md). The detailed designs behind the Phase 1–4 items — authentication, rate limiting, admission control, batching, replicas, metering and tenancy, each mapped onto this codebase — are in the **[system design set](system-design/README.md)**, which uses the same item ids.

| Mark | Meaning |
|---|---|
| ✅ | implemented and covered by tests |
| 🟡 | partly there — the gap is stated |
| ⬜ | not implemented |

---

## 1. Readiness levels

| Level | Who may call it | What it requires | shilp |
|---|---|---|---|
| **L0 — Local** | you, on this machine | a correct engine and API, honest errors, basic telemetry | ✅ **here** |
| **L1 — Team** | a trusted team, through a proxy | one key, admin endpoints closed, request limits, bounded queue, request ids | ⬜ Phase 1 |
| **L2 — Multi-tenant** | several teams or customers | per-tenant keys, rate limits and quotas, metering per tenant, fair queueing | ⬜ Phase 2 |
| **L3 — Scaled** | real traffic | batching, replicas with a router, shared metering, containers, autoscaling | ⬜ Phase 3 |
| **L4 — Provider** | paying customers | billing, SLO automation, automated rollout and evaluation gates | ⬜ Phase 4 |

Each level depends on the one before it. In particular, **nothing above L0 is safe without authentication** — rate limits, quotas, fair queueing and metering all need to know who is calling ([06-service-fundamentals.md §7](06-service-fundamentals.md)).

---

## 2. Capability matrix

### Inference engine

| Capability | Status | Today | Next |
|---|---|---|---|
| GGUF inference via llama.cpp | ✅ | `engine/llamacpp.py` | — |
| GPU offload (Metal/CUDA) | ✅ | all layers by default, `SHILP_GPU_LAYERS` (`config.py`) | — |
| Alternative engines | ✅ | `transformers`, `fake` behind one `Engine` protocol (`engine/__init__.py`) | P3: `llama-server`, vLLM |
| Prompt-prefix reuse | 🟡 | previous request per model only (llama-cpp-python) | P3: multi-slot via `llama-server` |
| Continuous batching | ⬜ | one sequence at a time per model | P3.1 |
| KV-cache knobs (cache type, flash attention, sliding window) | ⬜ | not exposed | P3.6 |
| Multiple resident models | ✅ | `SHILP_MAX_RESIDENT`, LRU, reference counts (`engine/loader.py`) | — |
| Unload / reload a model without restart | ⬜ | loaded engines are cached by name until eviction or restart | P1.9 |

### API contract and streaming

| Capability | Status | Today | Next |
|---|---|---|---|
| OpenAI and Ollama dialects | ✅ | `server/routes_openai.py`, `server/routes_ollama.py` | — |
| Streaming (SSE, NDJSON) | ✅ | one generation path (`server/generate.py`) | — |
| Status codes for failures before the first token, even when streaming | ✅ | `prime()` in `server/generate.py` | — |
| Error frame for failures after the first token | ✅ | both dialects | — |
| Context-window check | ✅ | 400 before any work (`ContextOverflow`, `engine/base.py`) | — |
| Token usage | 🟡 | prompt tokens exact (rendered prompt, `engine/llamacpp.py`); completion tokens by re-tokenizing the output, close but not guaranteed exact | P1.6: use the engine's generated-token count; P2.3: cached-token count |
| Finish reason `length` vs `stop` | 🟡 | always `stop` | P1.6 |
| Error body shape per dialect | 🟡 | Ollama's flat shape on both surfaces | P1.7 |
| Request ids | ⬜ | OpenAI `id` generated but not logged or returned as a header | P1.5 |
| Ollama requests without `Content-Type` | ⬜ | 422 | P1.10 |
| Structured output / JSON mode | ⬜ | — | P2.9 |
| CORS | ⬜ | same-origin only | P2.7 |

### Capacity, admission and limits

| Capability | Status | Today | Next |
|---|---|---|---|
| Non-blocking server under load | ✅ | worker thread per generation (`server/stream.py`) | — |
| Cancellation of abandoned answers | 🟡 | while generating: stop flag + generator close, recorded as 499. Not while waiting for the model lock or during prefill | P1.4: cancellable admission queue; P3.8: abort during prefill |
| Bounded queue with 429 | ⬜ | requests wait on the engine lock without limit | P1.4 |
| Body size, message count, `max_tokens` ceiling, deadline | ⬜ | only the context window is enforced | P1.3 |
| Rate limits (RPM, TPM, in-flight) | ⬜ | — | P2.2 |
| Quotas and budgets | ⬜ | — | P2.5 |
| Fair queueing between tenants | ⬜ | — | P2.4 |

### Identity, tenancy and abuse

| Capability | Status | Today | Next |
|---|---|---|---|
| Authentication | ⬜ | none; binds to `127.0.0.1` by default | P1.1, then P2.1 |
| Admin endpoint separation (`/api/pull`, `/api/delete`) | ⬜ | open on the same port as inference | P1.2 |
| Tenants and scoped keys | ⬜ | — | P2.1 |
| Moderation hook | ⬜ | — | P2.6 |
| Prefix-cache isolation per tenant | ⬜ | one shared cache per model | P2.8 |

### Metering, observability and billing

| Capability | Status | Today | Next |
|---|---|---|---|
| Per-request usage records | 🟡 | SQLite row per request *that reaches generation*, with true status, tokens, TTFT, total time, tok/s (`server/telemetry.py`); 404 and 422 rejections are not recorded | P1.5: record from request arrival |
| Prometheus metrics | ✅ | counters by status, token counts, mean durations (`server/metrics.py`) | P1.11: histograms |
| Percentiles | 🟡 | from SQLite with SQL ([05-deploy.md §11](05-deploy.md)); `/metrics` has means only | P1.11 |
| Tenant on usage records | ⬜ | — | P2.3 |
| Shared metering across replicas | ⬜ | one SQLite file per process | P3.3 |
| Tracing | ⬜ | — | P3.7 |
| Billing (price book, aggregation, invoices) | ⬜ | — | P4.1 |

### Model lifecycle and deployment

| Capability | Status | Today | Next |
|---|---|---|---|
| Content-addressed weights, manifest digest, recipes | ✅ | `store.py`, `GET /api/tags` digest | — |
| Aliases (`assistant:prod`) | ✅ | a Shilpfile `FROM` an existing model shares its blobs | P1.9: reload on change |
| Pin an upstream revision | 🟡 | honoured for safetensors repos; **ignored for GGUF pulls** | P1.8 |
| Load testing with an SLO gate | ✅ | `scripts/loadtest.py`, `make loadtest` | — |
| Smoke test over real HTTP | ✅ | `scripts/smoke_test.py` | — |
| Health / readiness | 🟡 | `/health` is unconditional liveness and lists resident models; there is no readiness signal, and models load lazily on the first request | P3.5: `--preload`, `/ready` |
| Graceful shutdown | 🟡 | uvicorn's default on `SIGTERM` | P3.5: drain timeout |
| Containers and orchestration | ⬜ | — | P3.5 |
| Evaluation gate for model changes | ⬜ | — | P4.4 (project roadmap item 1) |

---

## 3. Phase 0 — done

What it took to make L0 honest, each found by measuring a running server:

| Change | Evidence |
|---|---|
| Per-request timing (TTFT and totals had counted from process start) | Ollama `total_duration` ~1,093 s → 0.05 s |
| GPU offload by default | TTFT 178 → 27 ms; 3.1k-token prefill 9.2 → 1.6 s |
| Rendered-prompt token counting | chat `prompt_tokens` 1 → actual |
| Cancellation releases the model while generating | next request after an abandoned one: 6.9 s → 0.2 s |
| Early failures become status codes even when streaming; error frames after | over-long prompt: cut connection → 400 |
| True status codes in telemetry and `/metrics` | 400/499/503/500 instead of a blanket 500 |
| Load-test script with open-loop arrivals and an SLO gate | measured SLO capacity ≈ 30% of throughput ([07-operate.md §1](07-operate.md)) |

---

## 4. Phase 1 — safe for a team (L1)

Goal: shilp can sit behind a proxy and be used by a trusted team without anyone being able to break it by accident.

**P1.1 Single-key authentication.** A FastAPI dependency checks `Authorization: Bearer <key>` against `SHILP_API_KEY` with `hmac.compare_digest`; inference and model-listing routes require it; `/health` stays open; `/metrics` open or keyed by setting. When the variable is unset the server refuses to bind to a non-loopback address.
*Done when:* every inference route returns 401 without the key (tested); starting with `--host 0.0.0.0` and no key fails with a clear message.

**P1.2 Admin separation.** `/api/pull` and `/api/delete` disabled over HTTP unless `SHILP_ENABLE_ADMIN=1`, and then only with an admin key. The CLI is unaffected.
*Done when:* both return **404** by default — the routes are not registered at all — and **403** when admin is enabled but the key lacks the admin scope (tested); the deploy doc's proxy allowlist becomes defence in depth rather than the only defence.

**P1.3 Request limits.** `SHILP_MAX_BODY_BYTES`, a maximum message count, `SHILP_MAX_OUTPUT_TOKENS` (a ceiling on `max_tokens` / `num_predict`), and `SHILP_REQUEST_TIMEOUT` (a deadline that cancels the generation through the existing stop flag — between tokens, so it cannot cut a prefill short; the prompt cap bounds that).
*Done when:* each limit returns a 400 or 413 naming the limit, or ends the stream with `finish_reason: "length"` / a timeout error frame (tested).

**P1.4 Admission control.** An explicit, FIFO, *cancellable* wait queue per model in the loader — replacing the wait on the engine's thread lock, so a client that leaves while queued is removed at once instead of later paying for its prefill — with a bound on waiting requests (`SHILP_MAX_WAITING`); beyond it, `429` with `Retry-After` ≈ waiting × recent mean service time. A `shilp_requests_waiting` gauge in `/metrics`. Because routes already await the first chunk before sending headers, the 429 works for streaming requests too.
*Done when:* an open-loop load test at 2× capacity shows bounded p95 TTFT for accepted requests and a 429 share that matches the overload (measured); unit tests for the bound, the header, arrival order, and a queued request being dropped when its client disconnects.

**P1.5 Request ids, and records from arrival.** Accept `X-Request-ID` (or generate one), return it on every response, store it on the telemetry row, include it in error bodies and logs. Create the usage record when the request arrives, so rejections before generation (unknown model 404, validation 422) are recorded too.
*Done when:* a request id round-trips and appears in the usage row, and a 404 produces a row (tested).

**P1.6 Finish reasons and exact output counts.** Pass llama.cpp's `length` through instead of always reporting `stop`, on both surfaces and in telemetry; count completion tokens from the engine's generated tokens rather than by re-tokenizing the text.
*Done when:* a request with `max_tokens: 5` reports `length` (tested against the fake engine and verified against llama.cpp).

**P1.7 OpenAI error shape.** `/v1/*` errors as `{"error": {"message", "type", "code"}}`; `/api/*` keeps Ollama's flat shape.
*Done when:* the official `openai` Python client surfaces the message text for a 400 and a 404 (tested).

**P1.8 Revision pinning for GGUF pulls.** Honour `--revision` for `hf:org/repo-GGUF:quant` and record the resolved commit in the recipe.
*Done when:* a pinned pull downloads the pinned commit's file (tested with the fake hub).

**P1.9 Unload and reload.** `POST /api/unload` (and Ollama's `keep_alive: 0`), plus reloading a resident model when its manifest digest changes — so promoting an alias does not need a restart.
*Done when:* repointing `assistant:prod` changes the served model on the next request (tested).

**P1.10 Ollama content-type tolerance.** Parse a JSON body regardless of `Content-Type` on `/api/*`, as Ollama does.

**P1.11 Histograms.** Bucketed histograms for TTFT, total time and time per output token in `/metrics`, so p95 is available to Prometheus without SQL.

**Phase 1 exit criteria:** the [07-operate.md](07-operate.md) checklist items marked *(roadmap)* except evaluation are satisfied; an open-loop load test at the measured SLO-safe rate passes `--slo-ttft-p95` and `--slo-error-rate` through the proxy; a security review finds no unauthenticated route other than `/health`.

---

## 5. Phase 2 — multi-tenant (L2)

Goal: several teams or customers share one deployment without seeing or starving each other, and every token is attributable.

**P2.1 Tenants and keys.** A key store (SQLite at first) of tenants and keys: keys generated with a public prefix, stored as SHA-256 hashes, shown once; scopes for models and endpoints; revocation and last-used time. CLI: `shilp keys create|list|revoke`.

**P2.2 Rate limits.** Per-key token buckets for requests per minute and tokens per minute, plus an in-flight cap; reserve `prompt_tokens + max_tokens` at admission and refund the unused part at the end. `429` with `Retry-After` and `x-ratelimit-*` headers.

**P2.3 Metering per tenant.** Tenant id, key id, request id and cached prompt tokens on every usage row; abandoned requests billed for tokens actually generated; `shilp usage --tenant`; JSONL/CSV export.

**P2.4 Fair queueing.** The admission queue becomes per-tenant queues served by weighted round robin, so one tenant's burst cannot fill everyone's queue.

**P2.5 Quotas and budgets.** Daily and monthly token quotas per tenant, with alerts at thresholds and a hard stop at the limit.

**P2.6 Moderation hook.** An optional, pluggable check on input and output (a classifier endpoint or a local model), configured per tenant.

**P2.7 CORS.** An explicit origin allowlist, per tenant if needed.

**P2.8 Cache isolation.** Option to keep prefix caches per tenant (or route tenants to separate replicas), closing the timing side channel ([06-service-fundamentals.md §8](06-service-fundamentals.md)).

**P2.9 Structured output.** `response_format` (OpenAI) and `format` (Ollama) mapped onto llama.cpp's grammar-constrained decoding.

**Phase 2 exit criteria:** a two-tenant load test where tenant A floods shows tenant B's p95 TTFT within its SLO; usage per tenant reconciles with the load generator's own counts to within 1%.

---

## 6. Phase 3 — scaled (L3)

Goal: capacity grows with hardware, not with hope.

**P3.1 `llama-server` engine.** An `Engine` that starts or attaches to `llama-server --parallel N` on the same GGUF blobs and streams from it — continuous batching and multi-slot prefix caching, on Metal as well as CUDA. The most important item in this phase: the worked example in [05-deploy.md §4.7](05-deploy.md) shows a workload that needs three serial machines fitting on one batching box.
*Done when:* the closed-loop curve in [07-operate.md §2](07-operate.md) shows throughput rising with users instead of staying flat.

**P3.2 vLLM engine** for NVIDIA GPUs: paged attention, higher concurrency per GPU.

**P3.3 Shared metering.** Usage events to a shared store (Postgres, or an append-only log consumed into one), idempotent on request id, so replicas report one set of numbers.

**P3.4 Routing.** Reference configurations for a router in front of replicas: least outstanding requests, with prefix/tenant affinity under a load bound. External (Envoy, nginx, a small custom router) rather than inside shilp.

**P3.5 Containers and orchestration.** CPU and CUDA images with weights on a volume; compose and Kubernetes examples with readiness after warmup, a `preStop` drain, a termination grace period longer than the longest stream, and autoscaling on requests in flight.

**P3.6 KV-cache knobs.** `kv_cache_type`, `flash_attn` and sliding-window settings as Shilpfile parameters, measured with the evaluation harness before becoming defaults.

**P3.7 Tracing.** OpenTelemetry spans for queue wait, prefill, decode and streaming, carrying the request id.

**P3.8 Cancellation during prefill** via llama.cpp's abort callback once the binding exposes it.

**Phase 3 exit criteria:** with N replicas behind the router, SLO capacity scales close to N× in an open-loop test; a rolling deploy under load produces no failed requests.

---

## 7. Phase 4 — provider (L4)

**P4.1 Billing.** A versioned price book (input, cached input and output prices per model), hourly aggregation per tenant from the metering store, invoices or exports, and reconciliation against infrastructure cost.

**P4.2 SLO tooling.** Recording rules and multi-window burn-rate alerts shipped as configuration ([07-operate.md §1](07-operate.md)).

**P4.3 Rollout automation.** Weighted canaries, shadow traffic and automatic rollback when a canary burns error budget faster than the baseline.

**P4.4 Evaluation gate.** The [project roadmap's](04-roadmap.md) evaluation harness run in the deploy pipeline alongside the load test; a model or configuration change that regresses either does not ship.

**P4.5 Efficiency.** Response caching for identical `temperature: 0` requests; speculative decoding for latency-sensitive models.

---

## 8. Order of work and dependencies

```
P1.1 auth ──► P1.2 admin separation
   │
   ├──► P1.3 limits ──► P1.4 admission (429) ──► P2.4 fair queueing
   │                                                 ▲
   └──► P2.1 tenants & keys ──► P2.2 rate limits ────┤
                 │                                   │
                 └──► P2.3 metering per tenant ──► P3.3 shared metering ──► P4.1 billing

P1.5 request ids ──► P3.7 tracing
P1.9 unload/reload ──► P4.3 rollout automation
P3.1 llama-server engine ──► P3.4 routing ──► P3.5 containers/autoscaling
```

If only a few items get built, build them in this order: **P1.1 auth, P1.2 admin separation, P1.3 limits, P1.4 admission control, P3.1 the `llama-server` engine.** The first four make it safe to share; the fifth makes it worth sharing.

---

## 9. Out of scope

- A hosted control plane, web console or signup flow.
- Training or fine-tuning as a service (the project roadmap covers a local fine-tuning pipeline).
- Multi-region active-active deployment.
- Content policy itself — shilp can call a moderation hook; deciding what is allowed is the operator's job.
