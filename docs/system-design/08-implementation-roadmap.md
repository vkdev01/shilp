# 08 — Implementation roadmap

**The build plan for the seven designs: what exists, what to build, in which order, how long it should take, which files change, and how to know each step is done.**

Item ids match [08-service-roadmap.md](../08-service-roadmap.md). Effort is an estimate for one engineer who knows this codebase, including tests and documentation, excluding infrastructure provisioning. Legend: ✅ implemented · 🟡 partial · ⬜ missing.

---

## 1. Where each topic stands

| # | Topic | In place today | To build | Design |
|---|---|---|---|---|
| 1 | **Authentication** | loopback bind by default | single key, admin separation, refuse public bind without auth; then tenants and hashed scoped keys | [01](01-authentication.md) |
| 2 | **Rate limiting & quotas** | context-window check | request caps; RPM/TPM/in-flight per key with reserve-and-refund; Redis for replicas; quotas | [02](02-rate-limiting-and-quotas.md) |
| 3 | **Queue & admission** | implicit unbounded wait on the engine lock; 503 for a different busy model; cancellation while generating | bounded FIFO queue per model, 429 + `Retry-After`, timeout, removal of clients who leave while waiting, metrics; fair queueing | [03](03-admission-control.md) |
| 4 | **Batching** | concurrency-safe server; an engine protocol and cancellation bridge a forwarding engine can reuse | `llama-server` engine with N slots; vLLM engine | [04](04-batching-and-concurrency.md) |
| 5 | **Replicas & LB** | stateless API; liveness; content-addressed blobs | real preload + `/ready`; draining; models/state directory split; shared state; reference configs; images | [05](05-replicas-and-load-balancing.md) |
| 6 | **Metering & billing** | one row per request that reaches generation, true status, tokens; Prometheus counters | request ids; rows for rejections; tenant/key/digest/cached tokens; daily aggregates; exports; shared ledger; price book and invoices | [06](06-metering-and-billing.md) |
| 7 | **Tenancy & abuse** | no prompt text stored | tenant policy; caps; model allowlists; moderation hook; cache isolation; opt-in prompt log; audit log; anomaly alerts | [07](07-multi-tenancy-and-abuse.md) |

---

## 2. Phase 1 — safe for a team (L1) · ~7–10 days

Goal: a trusted team can use one shilp process behind a proxy, and nobody can break it by accident.

| Id | Item | Topic | Effort | Main files |
|---|---|---|---|---|
| P1.1 | Single-key auth (`SHILP_API_KEY`), scopes, refuse public bind without auth | 1 | 1 d | `server/auth.py` (new), `server/app.py`, `config.py`, `cli.py` |
| P1.2 | Admin router off by default (`SHILP_ENABLE_ADMIN`, `SHILP_ADMIN_KEY`) | 1, 7 | 0.5 d | `server/routes_admin.py` (new), `server/routes_ollama.py` |
| P1.3 | Request caps: body size, messages, `max_tokens` ceiling, deadline | 2, 7 | 1 d | `server/app.py`, `routes_*.py`, `server/stream.py`, `config.py` |
| P1.4 | Bounded FIFO admission queue per model, 429/503, cancellable waits, queue metrics | 3 | 2–4 d | `engine/admission.py` (new), `server/generate.py`, `server/state.py`, `server/app.py` |
| P1.5 | Request ids; usage record created on arrival (rejections recorded) | 6 | 1 d | routes, `server/telemetry.py`, `server/generate.py` |
| P1.6 | `finish_reason` passthrough; exact completion counts where the engine reports them | 6 | 0.5 d | `engine/llamacpp.py`, routes |
| P1.7–P1.11 | OpenAI error shape, GGUF revision pinning, unload/reload, Ollama content-type, histograms | — | 1–2 d | see [08-service-roadmap.md](../08-service-roadmap.md) |

**Exit criteria**

- Every route except `/health` returns 401 without a key; admin routes are 404 unless enabled.
- An open-loop load test at **2× capacity** through the proxy: accepted requests keep p95 time-to-first-token within the queue budget; the rest get 429 with `Retry-After` in milliseconds.
- A client that disconnects while queued never reaches the engine (fake-engine call log).
- Every request, including rejections, has exactly one usage row with a request id.

---

## 3. Phase 2 — multi-tenant (L2) · ~3–4 weeks

Goal: several teams or customers share a deployment without seeing or starving each other, and every token is attributable.

| Id | Item | Topic | Effort | Main files |
|---|---|---|---|---|
| P2.1 | Tenants and keys: hashed, prefixed, scoped, revocable; `shilp keys` CLI; tenant policy | 1, 7 | 2–3 d | `keys.py` (new), `server/auth.py`, `cli.py` |
| P2.2 | Rate limits per key: RPM, TPM (reserve/refund), in-flight; memory + Redis; 429 headers | 2 | 2–3 d | `server/ratelimit.py` (new), routes, `server/generate.py` |
| P2.3 | Metering per tenant: tenant/key/digest/cached tokens, daily aggregates, `shilp usage` filters and CSV, `GET /v1/usage` | 6 | 3–4 d | `server/telemetry.py`, `cli.py`, `routes_openai.py` |
| P2.4 | Fair queueing by tenant (deficit round robin) | 3, 7 | 1–2 d | `engine/admission.py` |
| P2.5 | Quotas and budget alerts from aggregates | 2 | 1 d | `server/ratelimit.py` |
| P2.6 | Moderation hook (none / HTTP / model-served), hold-back window, `content_filter` | 7 | 2–3 d | `server/moderation.py` (new), `server/generate.py` |
| P2.8 | Cache isolation option per tenant | 7 | 1 d | `engine/loader.py`, routing docs |
| — | Opt-in prompt log with retention and purge; audit log; anomaly alerts | 7 | 2–3 d | `server/promptlog.py`, `server/audit.py` (new) |

**Exit criteria**

- Two-tenant load test: tenant A floods, tenant B's p95 time-to-first-token stays within its SLO.
- Per-tenant usage reconciles with the load generator's own counts to within 1%.
- A revoked key stops working within the cache TTL (30 s).
- No prompt text exists at rest unless a tenant opted in.

---

## 4. Phase 3 — scaled (L3) · ~3–5 weeks

Goal: capacity grows with hardware.

| Id | Item | Topic | Effort | Main files |
|---|---|---|---|---|
| P3.1 | `UpstreamEngine` + `llama-server` launcher, N slots, exact usage from the upstream | 4 | 1–2 weeks | `engine/upstream.py` (new), `engine/__init__.py`, `engine/loader.py` |
| P3.2 | vLLM launcher (CUDA) | 4 | 2–3 d | `engine/upstream.py` |
| P3.3 | Shared metering: shipper to Postgres, idempotent | 5, 6 | 2–3 d | `server/usage_shipper.py` (new) |
| P3.4 | Routing reference configs: least outstanding, model pools, bounded affinity | 5 | 1–2 d | `deploy/` (new), docs |
| P3.5 | Preload + `/ready`, draining, `SHILP_MODELS_DIR` / `SHILP_STATE_DIR`, images, Kubernetes example | 5 | 3–5 d | `cli.py`, `server/app.py`, `config.py`, `store.py`, `Dockerfile`s |

**Exit criteria**

- Closed-loop curve with the batching engine: aggregate throughput **rises** with users (≥ 2.5× single-user at 4 users), per-stream speed ≥ 20 tok/s.
- N replicas behind the router: SLO-safe throughput scales close to N× in an open-loop test.
- A rolling deploy under load (`maxSurge: 1, maxUnavailable: 0`) produces no failed requests.

---

## 5. Phase 4 — provider (L4) · ~1–2 weeks for billing

| Id | Item | Topic | Effort |
|---|---|---|---|
| P4.1 | Versioned price book, invoice export or billing-platform feed, cost reconciliation | 6 | ~1 week |
| P4.2–P4.5 | SLO tooling, rollout automation, evaluation gate, response cache | ops | see [08-service-roadmap.md](../08-service-roadmap.md) |

---

## 6. Dependencies and parallel tracks

```
Track A — protect the box (one engineer, ~2 weeks)
  P1.1 auth ─► P1.2 admin ─► P1.3 caps ─► P1.4 admission queue ─► P1.5 request ids

Track B — share it (needs Track A)
  P2.1 tenants & keys ─┬─► P2.2 rate limits ─► P2.5 quotas
                       ├─► P2.3 metering ─────► P3.3 shared metering ─► P4.1 billing
                       ├─► P2.4 fair queueing
                       └─► P2.6 moderation, P2.8 cache isolation

Track C — scale it (independent; can start any time)
  P3.1 llama-server engine ─► P3.2 vLLM
  P3.5 readiness, drain, store split ─► P3.4 routing configs
```

Track C does not need identity, so a second engineer can start the batching engine on day one. The only coupling is that P1.4's queue reads its slot count from the engine, which P3.1 changes from 1 to N.

**Total for the seven topics:** roughly **8–11 engineer-weeks** (Phase 1 ~2, Phase 2 ~3–4, Phase 3 ~3–5, billing ~1), or about half that elapsed with two engineers on Tracks A/B and C.

---

## 7. If only a few things get built

| Goal | Build | Time |
|---|---|---|
| Expose to a team safely | P1.1, P1.2, P1.3, P1.4 | ~1 week |
| …and know who used what | + P1.5, P2.1, P2.3 | +1 week |
| Serve several times more users per machine | P3.1 | 1–2 weeks |
| Charge for it | P2.1, P2.2, P2.3, P4.1 | ~3 weeks |

---

## 8. New settings, all in one place

| Variable | Item | Default |
|---|---|---|
| `SHILP_API_KEY`, `SHILP_ADMIN_KEY`, `SHILP_AUTH`, `SHILP_ENABLE_ADMIN`, `SHILP_TRUSTED_IDENTITY_HEADER` | P1.1, P1.2 | unset, unset, auto, `0`, unset |
| `SHILP_MAX_BODY_BYTES`, `SHILP_MAX_MESSAGES`, `SHILP_MAX_OUTPUT_TOKENS`, `SHILP_REQUEST_TIMEOUT` | P1.3 | 1 MiB, 256, 1024, 120 s |
| `SHILP_MAX_WAITING`, `SHILP_MAX_QUEUE_WAIT`, `SHILP_SLOTS` | P1.4 | 4, 10 s, from engine |
| `SHILP_RATE_RPM`, `SHILP_RATE_TPM`, `SHILP_MAX_IN_FLIGHT_PER_KEY`, `SHILP_REDIS_URL` | P2.2 | unset (off) |
| `SHILP_MODERATION`, `SHILP_MODERATION_HOLDBACK` | P2.6 | `off`, 64 |
| `SHILP_LLAMA_SERVER`, `SHILP_UPSTREAM_START_TIMEOUT` | P3.1 | on `PATH`, 120 s |
| `SHILP_MODELS_DIR`, `SHILP_STATE_DIR`, `SHILP_DRAIN_TIMEOUT` | P3.5 | `$SHILP_HOME`, `$SHILP_HOME`, 120 s |

Every new setting defaults to today's behaviour, so each item can ship on its own without changing what existing users see.

---

## 9. Decisions to make before starting

| Decision | Options | Affects |
|---|---|---|
| `max_tokens` above the ceiling | clamp silently, or reject with 400 | P1.3, client compatibility |
| Rate-limit store with replicas | Redis, or split budgets per replica | P2.2, infrastructure |
| Usage ledger | Postgres, or an event log into a warehouse | P3.3 |
| Billing | build invoices, or feed a billing platform | P4.1 |
| Moderation | external API, or a guard model served by shilp | P2.6, data residency |
| Batching backend first | `llama-server` (Mac + CUDA) or vLLM (CUDA only) | P3.1/P3.2, target hardware |
| Error body for 401/403/429 on `/api/*` | Ollama flat shape (consistent) or OpenAI shape | P1.1, P1.7 |

---

## 10. How to track progress

Update the **Status** table at the end of each topic document and the capability matrix in [08-service-roadmap.md](../08-service-roadmap.md) in the same commit as the code. A status changes to ✅ only when the item's tests pass and, where an exit criterion names a load test, the `scripts/loadtest.py` run that proves it is recorded in the commit message or the docs.
