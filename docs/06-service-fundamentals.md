# 06 — Inference service fundamentals

**The parts of a model-serving system, explained from first principles: what each one is, how it works, what is different about it for LLMs, and where shilp stands today.**

[00-theory.md](00-theory.md) explains what one forward pass computes. [05-deploy.md](05-deploy.md) does the capacity arithmetic. This document sits between them: it names the *components* a real inference service is built from, so that the deployment guide's advice and the [service roadmap](08-service-roadmap.md) have a shared vocabulary.

Every section ends with **In shilp today**, using the same legend as the roadmap:

| Mark | Meaning |
|---|---|
| ✅ | implemented and tested |
| 🟡 | partly there — the gap is stated |
| ⬜ | not implemented — the roadmap says how |

---

## 1. The anatomy of a request

A hosted LLM API looks like one HTTP call. Behind it, a request crosses a chain of components, and each one exists because something specific goes wrong without it:

```
client
  │  HTTPS
  ▼
edge ─────────── TLS termination, request size limits, IP filtering
  │
  ▼
gateway ──────── authentication → tenant identity
  │              rate limits and quotas (requests and tokens per minute)
  ▼
router / LB ──── pick a replica: least outstanding requests, cache affinity, health
  │
  ▼
admission ────── bounded queue per model; 429 + Retry-After when full; deadlines
  │
  ▼
scheduler ────── which waiting requests run in the next step (batching, fairness)
  │
  ▼
engine ───────── tokenize → prefill → decode loop (KV cache) → detokenize
  │
  ▼
stream back ──── SSE / NDJSON; usage on the last frame; error frame on failure
  │
  ▼
metering ─────── one durable usage record per request → billing, quotas, analytics
```

| Layer | In shilp today |
|---|---|
| Edge (TLS, size limits) | ⬜ — use a reverse proxy ([05-deploy.md §12](05-deploy.md)) |
| Gateway (auth, rate limits, quotas) | ⬜ |
| Router / load balancer | ⬜ — single process; `/health` reports residency for an external LB |
| Admission (bounded queue, 429) | ⬜ — requests wait on the engine lock without a bound |
| Scheduler (batching, fairness) | ⬜ — one request at a time per model |
| Engine | ✅ llama.cpp (GGUF, Metal/CUDA offload), transformers, fake |
| Streaming | ✅ both dialects; status codes before the first byte, error frames after |
| Metering | 🟡 one telemetry row per request that reaches generation, with no tenant or key on it |

The rest of this document takes the layers one at a time, bottom up — starting with the one that makes LLM serving different from every other web service.

---

## 2. The KV cache — why LLM serving is stateful

### What it is

A transformer generates one token at a time, and each new token *attends* to every token before it. Attention needs, for every earlier position, two vectors per layer: a **key** (what that position offers to be matched against) and a **value** (what it contributes if matched). The new token produces a **query**, compares it with all the keys, and takes a weighted sum of the values.

Without a cache, generating token *n* means recomputing the keys and values of all *n − 1* earlier tokens — every step redoes the whole prompt, and a 500-token answer costs the work of ~125,000 token-passes. The **KV cache** stores each token's keys and values the first time they are computed, so every later step computes only its own. That removes the repeated re-processing — each step runs the network over one new token instead of the whole sequence — and it is the reason decode is feasible at all. Attention itself still compares the new token with every cached one, so each step costs a little more as the context grows; that term is what eventually makes very long contexts expensive ([05-deploy.md §4.5](05-deploy.md)).

### Why it dominates serving

The cache is not small, and it belongs to one conversation:

```
KV bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes_per_element
KV bytes per session = KV bytes per token × tokens in the context
```

For the models in [05-deploy.md §4.1](05-deploy.md), that is 26 KiB to 512 KiB per token — 0.1 to 2 GiB for one 4k-token conversation. Three consequences shape everything else in this document:

1. **Memory per user.** The weights are shared by every request; the cache is not. How many users fit is `free memory ÷ KV per session`, which is why context length is a capacity decision, not a quality setting.
2. **Requests are not interchangeable.** A request mid-generation owns cache state on one device. It cannot be moved to another replica cheaply, and evicting its cache means recomputing it.
3. **Reuse is valuable.** If two requests start with the same tokens, the cached keys and values for that prefix are identical. Reusing them skips prefill for the shared part — measured at ~19× faster time-to-first-token on a 1.7k-token prefix ([05-deploy.md §4.6](05-deploy.md)).

### How serving systems manage it

| Technique | What it does | Where it lives |
|---|---|---|
| Grouped-query / multi-query attention | fewer KV heads than query heads; 4–32× smaller cache | model architecture |
| Preallocation | reserve `n_ctx` tokens of cache per sequence up front | llama.cpp, shilp |
| Paged attention | allocate the cache in small blocks on demand, like virtual memory, so no sequence wastes its unused reservation | vLLM |
| Prefix caching | keep caches of common prefixes and attach them to new requests | llama.cpp (last request), vLLM, SGLang (radix tree of prefixes) |
| KV quantisation | store keys/values at 8 or 4 bits | llama.cpp, vLLM |
| Sliding-window attention | layers that only attend to the last *W* tokens need only *W* tokens of cache | Gemma 3, Mistral |
| Offload | move cold caches to CPU memory or disk and back | vLLM, LMCache |

**In shilp today**
- ✅ One llama.cpp context per model, sized by the model's `num_ctx` parameter (default 4096).
- ✅ Prefix reuse with the previous request on that model, provided by `llama-cpp-python`.
- ✅ Prompts longer than the context are rejected with a 400 before any work is done.
- ⬜ One cache slot only: two conversations with different prefixes evict each other on every switch.
- ⬜ KV quantisation, flash attention and sliding-window caches are not exposed as settings.

---

## 3. Concurrency, parallelism and batching

These three words are used interchangeably and mean different things.

- **Concurrency** — several requests are *in progress* at once. They may be taking turns.
- **Parallelism** — several computations *execute at the same instant*, on different cores or devices.
- **Batching** — several requests share *one* computation: one pass over the weights advances all of them.

For LLM decode, batching is the one that matters, and the reason is arithmetic intensity. A decode step for one sequence reads every weight once and does about two floating-point operations per weight — so the step takes as long as reading the model from memory, and the arithmetic units sit mostly idle ([00-theory.md §2.4](00-theory.md)). Advance *B* sequences in the same step and the weights are still read once; the extra arithmetic was free capacity. Aggregate throughput rises almost linearly with *B* until the step becomes compute-bound or the KV caches no longer fit.

### Kinds of batching

| Kind | How it works | Weakness |
|---|---|---|
| **Static** | collect *B* requests, run them together to completion | everyone waits for the longest answer; new arrivals wait for the whole batch |
| **Dynamic** | wait up to *t* ms to fill a batch, then run it | still runs to completion; short answers held hostage by long ones |
| **Continuous** (iteration-level) | the scheduler re-forms the batch at *every token step*: finished sequences leave, waiting ones join | needs per-sequence cache management — which is what paged attention provides |
| **Chunked prefill** | split a long prompt's prefill into chunks interleaved with other sequences' decode steps | a little extra scheduling; prevents one long prompt from freezing everyone's stream |
| **Prefill/decode disaggregation** | run prefill and decode on different machines, ship the cache between them | network transfer of the cache; worth it at large scale |

Continuous batching is what made LLM serving economical (Orca, OSDI 2022; vLLM, 2023). It also explains the shape of a batched server's latency: per-stream speed falls gently as more sequences share each step, while aggregate throughput climbs. Because people read at roughly 5–8 tokens per second, a stream can slow from 90 tok/s to 15 tok/s without anyone noticing — that slack is what batching spends.

### Serial, measured

Without batching, concurrency becomes waiting. Measured with this repository's load-test script ([07-operate.md §2](07-operate.md)) on one shilp process:

| Simulated users | Throughput | Output tok/s | p50 time to first token |
|---|---|---|---|
| 1 | 0.81 req/s | 87 | 29 ms |
| 2 | 0.80 req/s | 87 | 1.4 s |
| 4 | 0.82 req/s | 87 | 3.7 s |
| 8 | 0.83 req/s | 88 | 8.5 s |

Throughput is flat; every extra user adds one service time (~1.2 s) to everyone's wait.

**In shilp today**
- ✅ Concurrency at the HTTP layer: the event loop never blocks; each request's generation runs on a worker thread.
- ✅ Correctness under concurrency: one lock per model's KV cache; a model is never evicted mid-stream (reference counts).
- ⬜ Batching. `llama-cpp-python`'s high-level API runs one sequence at a time. The roadmap adds an engine that forwards to `llama-server --parallel N` (continuous batching on the same GGUF files, Metal included), then vLLM for CUDA.

---

## 4. Request queues and admission control

### What a queue is for

Every server has a queue whether it is designed or not — if work arrives faster than it is served, it waits *somewhere*: in a socket backlog, a thread pool, or a lock. A designed queue makes three things explicit: **how long** work may wait, **in what order** it is served, and **what happens when it is full**.

### Sizing it: the queue is a latency budget

From Little's law, the number of requests waiting is the arrival rate times the wait. Turn it around: if your latency budget for waiting is *B* seconds and one request takes *S* seconds of service, a serial server should hold at most about `B ÷ S` waiting requests. Anything beyond that will miss its deadline anyway; rejecting it immediately with a `429` and `Retry-After` is kinder than letting it wait and fail — and it keeps the requests you *do* accept fast. An unbounded queue converts overload into slowness for everyone.

### Ordering

| Policy | Good for | Risk |
|---|---|---|
| FIFO | simplicity, fairness by arrival | one long prompt blocks everyone behind it (head-of-line blocking) |
| Priority classes | interactive before batch jobs | starvation of the low class without aging |
| Weighted fair queueing per tenant | multi-tenant fairness: each tenant gets its share regardless of how much it sends | needs tenant identity (§7) |
| Shortest-job-first | lower mean latency | needs an output-length prediction; starves long jobs |
| Deadline-aware | drop requests whose client has already given up | needs the client's deadline (or a 499 signal) |

### The LLM twist: requests are not equal

A web request costs roughly the same as the next one. An LLM request's cost spans three orders of magnitude: 20 prompt tokens and 10 output tokens versus 30,000 prompt tokens and 2,000 output tokens. So queues and limits in LLM serving are often measured in **tokens**, not requests: admit a request if its *estimated* tokens (`prompt_tokens + max_tokens`) fit in the remaining budget, then correct the estimate when it finishes.

**In shilp today**
- 🟡 An implicit queue: requests wait on the model's engine lock. It is unbounded, has no timeout, and Python's lock does not guarantee FIFO order.
- ✅ Requests for a *different* model while one is busy get an immediate 503, not a wait.
- 🟡 Abandoned requests are cancelled and recorded as status 499 — but only once they are *generating*. A request whose client leaves while it is still waiting for the model lock stays in line, then pays for its prefill before it notices. A cancellable async admission queue (roadmap P1.4) fixes both.
- ⬜ A bounded per-model queue with `429 Retry-After` and a queue-depth gauge — roadmap Phase 1.

---

## 5. Rate limiting and quotas

A queue protects the *server* from total load. A rate limiter protects the server — and other users — from *one client*. A quota caps a client's usage over a long period (a day, a month), usually for billing.

### The algorithms

| Algorithm | How it works | Behaviour |
|---|---|---|
| **Token bucket** | a bucket of capacity *C* refills at *r* tokens/second; each request takes tokens; empty bucket → reject | allows bursts up to *C*, sustained rate *r*. The usual choice |
| **Leaky bucket** | requests enter a queue that drains at a fixed rate | smooths traffic to a constant rate; adds delay instead of rejecting |
| **Fixed window** | count requests per calendar minute | simple; allows 2× the limit across a window boundary |
| **Sliding window** | count over the trailing 60 s (log or weighted counters) | accurate; a little more state |
| **Concurrency limit** | at most *k* requests in flight per client | the right limit for long-running requests: caps the *resources held*, not the arrival rate |

### The LLM twist: limit tokens as well as requests

Because request cost varies so much, LLM APIs limit **requests per minute (RPM)** *and* **tokens per minute (TPM)**, often plus a concurrency cap. Output length is unknown at admission, so the usual design reserves `prompt_tokens + max_tokens` from the token bucket when the request starts and refunds the unused part when it ends. This is also why a server-side ceiling on `max_tokens` matters: without one, any client can reserve — or consume — the whole context.

### Telling the client

A rejected request should say when to retry. The widely used convention is `429 Too Many Requests` with a `Retry-After` header, plus headers describing the budget — OpenAI's API, for example, returns `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests` and the same three for `-tokens`. Clients that respect them back off smoothly instead of retrying into a wall.

### Across replicas

One process can keep counters in memory. With several replicas, either the counters live in a shared store (Redis with an atomic increment or a small Lua script is the common pattern) or each replica gets a fixed share of the budget. The shared store is exact; the split is simpler and approximately right when the load balancer spreads traffic evenly.

**In shilp today**
- ⬜ No rate limiting or quotas of any kind.
- ⬜ No server-side ceiling on `max_tokens`: a client may ask for up to the whole context.
- ✅ The only hard per-request limit is the context window (400 when exceeded).
- Roadmap: per-key RPM/TPM token buckets and an in-flight cap (Phase 2), a `max_tokens` ceiling (Phase 1).

---

## 6. Replicas and load balancing

### Replicas

A **replica** is an independent server process holding its own copy of the model. Replicas give you two things: throughput (N replicas serve roughly N times the load) and availability (one can fail or restart while others serve). They cost one model's worth of memory each, and each has its own KV caches.

A crucial property of the OpenAI and Ollama APIs makes replication easy: **they are stateless**. The client sends the whole conversation with every request; the server keeps nothing between turns. Any replica can serve any turn of any conversation. The KV cache is state, but only as an *optimisation* — losing it costs time, never correctness.

### Load-balancing algorithms

| Algorithm | Fit for LLM serving |
|---|---|
| Round robin | poor — request costs vary 1000×, so equal counts mean unequal load |
| Least connections / least outstanding requests | good — routes to the replica with the least work in flight |
| Power of two choices | good and cheap — pick two replicas at random, send to the less loaded |
| Consistent hashing on tenant or prefix | keeps each replica's prefix cache warm; pair with a load bound so a hot key cannot overload one replica ("consistent hashing with bounded loads") |

The tension is **cache affinity versus balance**: hashing by system prompt maximises prefix-cache hits ([05-deploy.md §4.6](05-deploy.md) measured a ~19× time-to-first-token difference), while least-loaded routing minimises queueing. Production routers combine them — prefer the replica that has the prefix unless it is much busier than the others.

### Health, readiness and draining

- **Liveness** — is the process alive? Failing it restarts the process.
- **Readiness** — should it receive traffic? A model server is not ready until its model is loaded and warm.
- **Draining** — on deploy or scale-down, stop sending new requests to a replica and let its in-flight streams finish before stopping it. Streams can last tens of seconds, so drain timeouts must be longer than for typical web services.

**In shilp today**
- ✅ `/health` answers while generations run and lists resident models — usable as a readiness signal after a warmup request (models load lazily, so `resident` is empty until the first request).
- ✅ Requests are stateless; any replica could serve any turn.
- 🟡 Running several replicas works if each has its own writable `SHILP_HOME` (models and telemetry live under the same directory, so a shared read-only store is not possible yet), and usage must be aggregated across replicas by hand.
- ⬜ No built-in router; no shared metering; no drain endpoint beyond uvicorn's graceful shutdown on `SIGTERM`.

---

## 7. Authentication and authorisation

### Why it blocks any exposure

Every other control in this document — quotas, rate limits, fair queueing, metering, billing, abuse detection — needs to know **who** is calling. Without authentication there is no "who": one anonymous user is indistinguishable from a thousand. That is why authentication is not one feature among many but the precondition for exposing the service at all. An unauthenticated model server on a public address is a free GPU for anyone who finds it, and — in shilp's case specifically — a remote "delete this model" button (§8).

### API keys, done properly

API keys are the norm for machine-to-machine APIs. The details that matter:

- **Generate** at least 32 random bytes from a cryptographic source; add a short public prefix (e.g. `shilp_live_…`) so a leaked key is recognisable in logs and by secret scanners.
- **Store only a hash** (SHA-256 is fine for high-entropy keys) and show the key to the user exactly once.
- **Compare in constant time** (`hmac.compare_digest`) to avoid timing leaks.
- **Scope** keys: which models, which endpoints (inference versus admin), which tenant.
- **Rotate and revoke**: support two active keys per client so rotation needs no downtime; record last-used time to find dead keys.
- **Never log the key**, only its prefix or id.

### Other mechanisms

- **OAuth 2.0 / OpenID Connect** when the caller is a *person* using your app — your app then calls the model API with a service credential.
- **JWTs** when an upstream gateway has already authenticated and passes a signed identity downstream.
- **mTLS** for service-to-service traffic inside a private network.

### Where to enforce it

At a gateway (one place for every service, often with rate limiting built in) or as a dependency in the application (one less moving part, and the identity is available to metering). Either way, **authentication** answers "who is this?" and **authorisation** answers "may they do this?" — e.g. inference keys must not be able to call `/api/pull` or `/api/delete`.

**In shilp today**
- ⬜ No authentication. The server binds to `127.0.0.1` by default precisely because of this.
- ⬜ Admin endpoints (`/api/pull`, `/api/delete`) are on the same port and as open as inference.
- Interim: a reverse proxy with one shared bearer key ([05-deploy.md §12](05-deploy.md)). Roadmap Phase 1: `SHILP_API_KEY` bearer check and admin endpoints disabled unless explicitly enabled; Phase 2: hashed per-tenant keys with scopes.

---

## 8. Multi-tenancy and abuse controls

### Tenancy

A **tenant** is the unit that owns usage: a customer, a team, an application. Multi-tenancy is serving several tenants from shared infrastructure without one affecting another.

| Isolation level | What is shared | Use when |
|---|---|---|
| Shared pool | replicas, model, caches | many small tenants; cheapest; needs per-tenant limits |
| Dedicated replicas | hardware pool only | a tenant with steady load or strict latency needs |
| Dedicated deployment | nothing | regulatory or contractual isolation |

The failure mode of a shared pool is the **noisy neighbour**: one tenant's 30k-token prompts or burst of traffic raises everyone else's time to first token. The defences are per-tenant concurrency caps, token-based rate limits (§5) and weighted fair queueing (§4).

Shared caches are a subtle shared resource. A prefix cache never returns another tenant's text, but its *timing* reveals whether a prefix was recently seen — a request that hits the cache answers ~19× sooner. If tenants' prompts must be secret from each other, partition caches by tenant or isolate tenants per replica.

### Abuse controls

| Threat | Control |
|---|---|
| Anonymous use, leaked keys | authentication; key prefixes for secret scanning; revocation; last-used tracking |
| Flooding | per-key RPM/TPM limits, in-flight caps, bounded queues |
| Expensive requests (huge prompts, `max_tokens` at the context limit) | input caps (bytes, messages, prompt tokens), a server-side `max_tokens` ceiling, a generation timeout |
| Slow clients holding streams open | per-request deadlines; cancellation when the client disconnects |
| Destructive or resource-heavy admin actions (pulling models, deleting them) | separate admin endpoints, admin-only keys, or disable them on public replicas |
| Harmful content in prompts or outputs | moderation classifiers on input and output, per tenant policy |
| Prompt injection | an application-layer concern (the model server cannot know intent) — but say so, and do not grant the model tools by default |
| Runaway cost | per-tenant budgets and alerts on token-rate anomalies |

**In shilp today**
- ✅ Context-window check (400) — the one hard input limit.
- 🟡 Cancellation when the client disconnects: a stream that is generating stops at the next token; a request still waiting for the model does not (§4).
- ✅ `num_predict` defaults to 512 output tokens per request — but the client can raise it.
- ⬜ No tenants, no per-tenant limits, no fair queueing, no body-size or message-count limits, no generation timeout, no moderation.
- ⬜ **`/api/pull` and `/api/delete` are unauthenticated.** On an exposed port, anyone can fill the disk with downloads or delete models. Keep shilp on loopback behind a proxy that only forwards the inference paths until the roadmap's admin separation lands.

---

## 9. Metering and billing

### Metering is not logging

A **meter** is the authoritative record of what a tenant consumed. Logs and metrics may be sampled, dropped or aggregated; the meter may not, because money and quotas depend on it. One usage record per request:

| Field | Why |
|---|---|
| request id | idempotency: a record delivered twice is counted once |
| tenant / key id | who pays |
| model (exact version) | prices differ per model |
| prompt tokens, of which cached | input price, and the cached-input discount |
| completion tokens | output price |
| status and finish reason | abandoned (499) and failed requests are billed by policy, not by accident |
| timestamps, latency | SLO credits, disputes |

### The pipeline

```
request ends ──► usage event (request id, tenant, tokens…) ──► durable append-only log
                                                                  │  at-least-once delivery
                                                                  ▼
                                    aggregation per tenant × model × hour (dedupe by request id)
                                                                  │
                                    ┌─────────────────────────────┼──────────────────────┐
                                    ▼                             ▼                      ▼
                              quota enforcement             invoices (price book,     analytics,
                              (feeds rate limits)           versioned)                cost per tenant
```

Two rules keep it correct: write the event *before* acknowledging completion where possible (or from the same `finally` that closes the stream), and make every downstream step idempotent on the request id, because "exactly once" is achieved by "at least once + dedupe".

### Why tokens are priced the way they are

The prices of hosted APIs follow the physics of §3, and shilp's own measurements show why:

- **Output tokens cost more than input tokens.** Prefill processes the prompt in parallel; decode produces one token per pass over the weights. Measured on one M3 Pro stream with Gemma-3-1B: prefill ran at ~2,000 tokens/s and decode at ~98 tokens/s — each output token took ~20× the device time of an input token. Batching narrows the gap at scale, which is why typical output/input price ratios are smaller than 20×, but the direction is set by this.
- **Cached input is discounted.** A prefix-cache hit skips prefill for the shared part (~19× faster in the measurement), so the provider's cost for those tokens is mostly the memory to keep them.
- **Long context costs more than linearly at the extreme.** Attention is quadratic beyond tens of thousands of tokens ([05-deploy.md §4.5](05-deploy.md)), and the cache holds memory for the whole conversation.

For a self-hosted service the same records answer the internal question: what does each team's usage cost in GPU-hours, and is that cheaper than a hosted API ([05-deploy.md §10](05-deploy.md))?

**In shilp today**
- ✅ One telemetry row per request that reaches generation, in SQLite: surface, endpoint, model, status as the client saw it (including 499, 400 and 503), prompt and completion tokens, time to first token, total time, tok/s. `prompt_tokens` counts the rendered prompt, chat template included.
- 🟡 `completion_tokens` is counted by re-tokenizing the generated text, which is close to but not guaranteed to equal the tokens the model produced; requests rejected before generation (404, 422) are not recorded.
- ✅ Prometheus counters for requests (by status) and tokens (by direction).
- 🟡 Good enough for capacity planning and per-model cost, not for billing: no tenant or key, no request id, no cached-token count, and a per-process file rather than a durable shared log.
- ⬜ No price book, aggregation or invoicing.

---

## 10. Putting it together

A minimal service that is safe to expose to *a team* needs §7 (one key), the input/output caps of §8, and the bounded queue of §4. A service that is safe to expose to *customers* needs everything in this document. The order is not arbitrary — each layer depends on the one before:

```
authentication ─► tenant identity ─► rate limits & quotas ─► fair queueing ─► metering ─► billing
                                         │
                       bounded queue ◄───┘   batching & replicas scale the capacity all of this protects
```

[08-service-roadmap.md](08-service-roadmap.md) turns this into a phased plan, with what is implemented, what is partial and what is missing, item by item. [07-operate.md](07-operate.md) covers running it once it exists: SLOs, load testing, rollout, observability and runbooks.

## Glossary

**Admission control** — deciding at the door whether a request may enter, so accepted work stays fast. **Arithmetic intensity** — operations per byte of memory read; low for single-sequence decode, which is why batching is nearly free. **Authentication / authorisation** — who is calling / what they may do. **Chunked prefill** — splitting a long prompt's prefill so it interleaves with other streams' decode. **Continuous batching** — re-forming the batch at every token step. **Head-of-line blocking** — a slow item at the front of a FIFO delaying everything behind it. **Idempotency** — processing the same event twice has the effect of processing it once. **KV cache** — the stored keys and values of earlier tokens, per layer, per sequence. **Meter** — the authoritative, non-sampled usage record. **Noisy neighbour** — one tenant degrading others on shared infrastructure. **Paged attention** — allocating the KV cache in blocks on demand. **Prefix cache** — re-using the KV cache of a shared prompt prefix. **Quota** — a usage cap over a long period. **Rate limit** — a cap on usage per short window. **Readiness** — whether an instance should receive traffic now. **RPM / TPM** — requests / tokens per minute. **Tenant** — the unit that owns usage, limits and bills. **Token bucket** — a limiter that allows bursts up to a capacity and a sustained refill rate. **Weighted fair queueing** — serving queues in proportion to weights, so no tenant can crowd out another.
