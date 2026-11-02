# 07 — Multi-tenancy and abuse controls

**Status:** 🟡 no prompt text is stored anywhere; every other control is missing · **Blocks:** sharing one deployment between teams or customers · **Depends on:** [01](01-authentication.md) (tenants), [02](02-rate-limiting-and-quotas.md) (limits), [03](03-admission-control.md) (fair queueing), [06](06-metering-and-billing.md) (per-tenant usage) · **Roadmap:** P1.2, P1.3, P2.1, P2.4–P2.8 · **Effort:** 1–2 weeks across those items

---

## 1. The problem

Sharing infrastructure is what makes a service cheap; it is also what lets one customer hurt another. On a shared shilp deployment today:

| Risk | What happens now |
|---|---|
| **Noisy neighbour** | one tenant with 4 requests in flight makes another tenant's single request wait ~4 × 1.23 ≈ **4.9 s** (serial engine, measured service time) |
| **Cost attack** | a request with a 30,000-token prompt and `max_tokens: 4096` holds the machine for ~30,000 / 2,000 + 4,096 / 98.7 ≈ 15 + 41 ≈ **56 s** (computed from measured prefill and decode rates); nothing caps it |
| **Destructive admin calls** | `/api/pull` and `/api/delete` are reachable by anyone who can reach the port |
| **Data exposure via timing** | the shared prefix cache answers ~19× faster for a prefix someone sent recently, revealing that it was sent |
| **No identity** | none of the above can be attributed, limited or billed to anyone |

The phrase "all above this layer" is half right. *Some* of these controls belong in the application that calls shilp — user accounts, conversation history, defences against prompt injection. But the ones that protect the **shared resource** — identity, limits, fairness, isolation, metering, and a hook for content checks — can only be enforced where the resource is, which is shilp.

---

## 2. Concepts

### Tenant

The unit that owns usage, limits and bills: a customer, a team, or an application. Keys belong to tenants ([01](01-authentication.md)); usage is aggregated per tenant ([06](06-metering-and-billing.md)).

### Isolation levels

| Level | Shared | Cost | Use when |
|---|---|---|---|
| **Shared pool** | replicas, models, caches | lowest | many small tenants; requires per-tenant limits and fair queueing |
| **Dedicated replicas** | hardware pool, control plane | medium | a tenant with steady load, strict latency, or secret prompts |
| **Dedicated deployment** | nothing | highest | regulatory or contractual isolation |

Most services run a shared pool with a few dedicated pools for large or sensitive tenants.

### Three kinds of isolation

1. **Performance isolation** — one tenant cannot degrade another's latency: per-tenant rate limits and in-flight caps ([02](02-rate-limiting-and-quotas.md)), fair queueing ([03 §4.7](03-admission-control.md)), request caps.
2. **Data isolation** — one tenant cannot see another's data: no shared text storage, per-tenant access to usage, no cross-tenant inference from timing.
3. **Policy isolation** — tenants can have different rules: which models, how much context, which content policy, whether prompts may be logged.

### Where text lives in an inference server

| Place | Lifetime | Risk |
|---|---|---|
| Request body in memory | the request | low |
| KV / prefix cache | until evicted (in shilp: until the next request with a different prefix) | the timing side channel; crash dumps |
| Logs | as configured | **highest** — usually the real leak |
| Swap, core dumps | until overwritten | host-level exposure |

shilp's telemetry stores counts and timings, **never text**, and uvicorn's access log records method, path and status, not bodies. That default is the most important data-isolation property it has and should stay the default.

### The prefix-cache side channel

A prefix-cache hit is ~19× faster to first token (823 ms → 43 ms, measured). An attacker who can guess part of another tenant's prompt (a template, a leaked system prompt) can send it and time the response to learn whether it was recently used. It reveals *that* something was sent, not new content — a weak channel, but real. Mitigations: per-tenant caches, per-tenant replicas, or resetting the cache when the tenant changes.

### Content controls

- **Input moderation** — classify the prompt before generating; reject with a clear code. Adds the classifier's latency (typically tens of milliseconds for a small classifier) to time-to-first-token.
- **Output moderation** — harder with streaming, because tokens reach the client before the full answer exists. Options:

| Approach | Latency cost | Protection |
|---|---|---|
| Moderate the full answer before sending (no streaming) | the whole generation | complete |
| **Hold-back window**: release tokens only after the next *k* have been checked | *k* tokens; at ~98 tok/s, 64 tokens ≈ **0.65 s** | most cases; a stream can still be cut mid-way |
| Check periodically, cut the stream on a hit | none until a hit | the client has already seen some text |
| Post-hoc review only | none | detection, not prevention |

When a stream is cut for policy reasons, end it with `finish_reason: "content_filter"` (the value OpenAI uses) so clients can tell it from an error.

### Abuse patterns

| Pattern | Control |
|---|---|
| Leaked or shared keys | key prefixes for secret scanners, per-key usage anomalies, revocation |
| Flooding | per-key RPM/TPM, in-flight caps, IP limits at the proxy |
| Cost attacks (huge prompts, huge `max_tokens`) | request caps, token-based limits, quotas |
| Slow-loris clients holding streams | generation deadline; cancellation on disconnect |
| Admin misuse | admin endpoints off by default, admin-only keys, an audit log |
| Harmful content | moderation hook with per-tenant policy |
| Prompt injection | an *application* concern — the server cannot know intent — but do not give the model tools or data by default |

---

## 3. Responsibility split

| Concern | shilp (this layer) | Calling application | Operator |
|---|---|---|---|
| Identity of the caller | keys, tenants, scopes | end-user accounts | who gets a key |
| Resource fairness | limits, fair queueing, caps | — | limit values |
| Usage and cost | metering per tenant | per-end-user accounting if needed | prices |
| Conversation storage | **none** (stateless) | yes, under its own retention rules | retention policy |
| Prompt logging | off; opt-in per tenant with retention | — | whether to enable |
| Content policy | a moderation **hook** | UX for refusals | the policy itself |
| Prompt injection, tool permissions | — | yes | — |
| Cache isolation | per-tenant option | — | which tenants need it |

---

## 4. The design for shilp

### 4.1 Tenant policy

Extends the tenants table from [01 §4.5](01-authentication.md):

| Setting | Example | Enforced at |
|---|---|---|
| `allowed_models` | `assistant:prod, embed:prod` | model resolution → 403 |
| `rpm`, `tpm`, `max_in_flight` | 30, 20,000, 2 | rate limiter ([02](02-rate-limiting-and-quotas.md)) |
| `monthly_tokens` | 5,000,000 | quotas ([02 §4.6](02-rate-limiting-and-quotas.md)) |
| `max_prompt_tokens`, `max_output_tokens` | 8,192, 1,024 | before generation → 400 / clamp |
| `weight` | 1 | fair queueing ([03 §4.7](03-admission-control.md)) |
| `moderation` | `off`, `input`, `input+output` | moderation hook |
| `store_prompts`, `retention_days` | `false`, 0 | prompt log |
| `cache_isolation` | `shared` or `dedicated` | routing / cache reset |

### 4.2 Request caps — the cost-attack fix

With `max_prompt_tokens = 8,192` and `max_output_tokens = 1,024`, the worst single request on the reference machine drops from ~56 s to about 8,192 / 2,000 + 1,024 / 98.7 ≈ 4 + 10 ≈ **14 s** of machine time (computed). Combined with an in-flight cap of 1–2 per tenant, no tenant can occupy more than one or two slots for more than that.

### 4.3 Moderation hook

```python
class Verdict(NamedTuple):
    allowed: bool
    category: str | None = None

class Moderator(Protocol):
    async def check_input(self, tenant: str, messages: list[ChatMessage]) -> Verdict: ...
    async def check_output(self, tenant: str, text: str) -> Verdict: ...
```

Implementations:

- `NoModerator` (default).
- `HttpModerator` — calls an external classifier endpoint.
- `ModelModerator` — uses a small safety-classifier model **served by shilp itself** as a second model (e.g. a Llama Guard–style GGUF), which keeps text on the same machine. It needs the classifier to be resident *at the same time* as the model being moderated: with today's default `SHILP_MAX_RESIDENT=1`, the loader refuses to load a second model while the first is streaming, so output moderation would fail with 503. Either set `SHILP_MAX_RESIDENT ≥ 2` and budget memory for both, or run the classifier in a separate shilp process and call it through `HttpModerator` — the cleaner choice once there are replicas.

Input rejection → **400** with `{"error": {"type": "invalid_request_error", "code": "content_policy_violation"}}`. Output: hold-back window of `SHILP_MODERATION_HOLDBACK` tokens (e.g. 64); on a hit, end the stream with `finish_reason: "content_filter"`. Moderation verdicts are metered (category and count, never text).

### 4.4 Data isolation

- **Text never stored by default** — keep `server/telemetry.py` free of text columns; a test asserts the schema has none.
- **Opt-in prompt log** per tenant (`store_prompts`), in a separate store from telemetry, encrypted at rest, with `retention_days` enforced by a daily purge and a per-tenant delete command (`shilp tenants purge acme`).
- **Usage visibility** — `GET /v1/usage` returns only the caller's tenant ([06 §4.5](06-metering-and-billing.md)).
- **Cache isolation** — for tenants marked `dedicated`: route to a dedicated replica pool ([05](05-replicas-and-load-balancing.md)), or, on a single process, reset the engine's cache when the tenant changes (costs the prefix-cache benefit for everyone on that model). With an upstream batching engine, slots can be pinned per tenant ([04](04-batching-and-concurrency.md)).
- **Host hygiene** — encrypted swap, core dumps disabled or restricted, logs without bodies.

### 4.5 Abuse detection and response

- **Anomaly alerts** per key: token rate above, say, 5× its own 7-day p95; a sudden jump in 4xx; requests from new networks.
- **Automatic responses**: temporary suspension after repeated 429 violations in a window; a hard stop at the quota.
- **Audit log** (separate from telemetry): key created, revoked, scopes changed, admin endpoint called, tenant policy changed — who, when, from where.

---

## 5. Step-by-step implementation

| Step | Change | Roadmap |
|---|---|---|
| 1 | Admin endpoints off by default, admin key | P1.2 ([01](01-authentication.md)) |
| 2 | Request caps: body size, messages, prompt and output tokens, deadline | P1.3 ([02](02-rate-limiting-and-quotas.md)) |
| 3 | Tenants and keys; policy table | P2.1 |
| 4 | Enforce `allowed_models`, prompt/output caps per tenant | P2.1 |
| 5 | Per-tenant rate limits, in-flight caps, quotas | P2.2, P2.5 |
| 6 | Fair queueing by tenant weight | P2.4 |
| 7 | Moderation hook: `NoModerator`, `HttpModerator`, `ModelModerator`; hold-back window | P2.6 |
| 8 | Cache isolation option | P2.8 |
| 9 | Opt-in prompt log with retention and purge | new item under P2 |
| 10 | Anomaly alerts, auto-suspension, audit log | new item under P2 |

---

## 6. Testing and verification

- **Noisy neighbour**: a two-tenant load test where tenant A floods; tenant B's p95 time-to-first-token stays within its SLO (Phase 2 exit criterion in [08-service-roadmap.md](../08-service-roadmap.md)).
- **Model allowlist**: a key for tenant A gets 403 on a model only tenant B may use.
- **Caps**: an oversized prompt gets 400 naming the limit; `max_tokens` above the ceiling is clamped or rejected per policy.
- **Moderation**: an input classified as disallowed returns `content_policy_violation`; an output hit ends the stream with `finish_reason: "content_filter"`.
- **No text at rest**: a schema test fails if a text column is added to telemetry; with `store_prompts=false`, no prompt text appears anywhere under `SHILP_STATE_DIR` after a test run.
- **Purge**: after `shilp tenants purge`, no rows or files for that tenant remain.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Moderation service down | per-tenant choice of fail-open (log and allow) or fail-closed (503); alert either way |
| Classifier false positives | return the category; allow per-tenant thresholds; review samples |
| Opt-in logs retained too long | purge job with alerting on failure; retention enforced by date, not by disk space |
| Tenant policy misconfiguration | validate on save; default to the safer value |
| Isolation assumed but not configured | `cache_isolation` visible in `shilp tenants show`; documented in the tenant contract |

---

## 8. Status

| Item | Status |
|---|---|
| No prompt or answer text stored by default | ✅ |
| Context-window check (a hard cap on prompt size) | ✅ |
| Cancellation when the client disconnects (while generating) | 🟡 |
| Admin endpoints separated and off by default | ⬜ P1.2 |
| Request caps (body, messages, output tokens, deadline) | ⬜ P1.3 |
| Tenants, policy, model allowlists | ⬜ P2.1 |
| Per-tenant limits, fair queueing, quotas | ⬜ P2.2, P2.4, P2.5 |
| Moderation hook | ⬜ P2.6 |
| Cache isolation | ⬜ P2.8 |
| Opt-in prompt log with retention; audit log; anomaly alerts | ⬜ Phase 2 |
