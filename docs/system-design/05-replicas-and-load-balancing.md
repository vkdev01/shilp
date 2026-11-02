# 05 — Replicas and load balancing

**Status:** 🟡 the API is stateless; readiness, draining and shared state are missing · **Blocks:** availability and capacity beyond one process · **Depends on:** shared state from [02](02-rate-limiting-and-quotas.md) and [06](06-metering-and-billing.md); best combined with [04](04-batching-and-concurrency.md) · **Roadmap:** P3.3 (shared metering), P3.4 (routing), P3.5 (containers, readiness, drain) · **Effort:** 3–5 days of code, plus infrastructure

---

## 1. The problem

One shilp process is one **capacity ceiling** and one **failure domain**:

- **Capacity.** The worked example in [05-deploy.md §4.7](../05-deploy.md) — a 200-person team assistant at a peak of ~0.42 req/s — needs about **three serial replicas** to keep 95% of first tokens under 2 s (or one batching box, [04](04-batching-and-concurrency.md)).
- **Availability.** Every deploy, crash or out-of-memory kill is a full outage. A restart also empties the prefix cache and needs a model load before the first answer.

Replicas fix both, and LLM APIs make replication easy — but only once a few pieces of state move out of the process and the load balancer knows when a replica is actually able to serve.

---

## 2. Concepts

### Stateless requests, stateful optimisations

The OpenAI and Ollama chat APIs are **stateless**: every request carries the whole conversation. Any replica can serve any turn of any conversation; nothing breaks if consecutive turns land on different replicas. The KV and prefix caches are state, but only as an **optimisation** — losing them costs time, never correctness.

What shilp keeps, and where it must live with replicas:

| State | Today | With replicas |
|---|---|---|
| Model weights and manifests | `$SHILP_HOME/blobs`, `manifests` | read-only copy per replica, distributed ahead of time |
| Loaded models, KV and prefix cache | process memory | stays per replica — hence cache-aware routing |
| Telemetry | `$SHILP_HOME/logs/shilp.db` | per replica, shipped to a shared store ([06](06-metering-and-billing.md)) |
| Rate-limit counters | — (planned in memory) | shared store, e.g. Redis ([02](02-rate-limiting-and-quotas.md)) |
| API keys | — (planned SQLite) | shared database, or a replicated read-only copy ([01](01-authentication.md)) |

### "Each replica loads its own copy of the weights"

True across machines, and the reason replicas cost memory linearly. On **one** machine the picture is subtler:

- llama.cpp **memory-maps** the GGUF file, so two processes reading the same file share the operating system's page cache for the read-only weights;
- but each process has its **own KV cache, its own GPU buffers** when layers are offloaded, and — the decisive part — they share the **same memory bandwidth**, which is what limits decode.

So two replicas on one machine do not double throughput; they split it. Replicas add capacity when they run on **separate machines or separate GPUs**. On a single box, batching is the lever.

### Load-balancing algorithms for LLM traffic

LLM requests vary in cost by 100–1000×, which changes which algorithms work:

| Algorithm | Behaviour | Fit |
|---|---|---|
| Round robin | equal *counts* per replica | poor — one replica can get three long requests in a row |
| Random | similar to round robin | poor |
| **Least outstanding requests** | send to the replica with the fewest requests in flight | good — a direct proxy for queue length |
| **Power of two choices** | pick two replicas at random, send to the less loaded | almost as good, cheap, no global state |
| **Consistent hashing with bounded load** | hash a key (tenant, system prompt) to a replica, but spill over when it is busier than average × (1 + ε) | keeps prefix caches warm without hot spots |

### Cache affinity, with numbers

The prefix cache turns a 1,712-token prompt's time-to-first-token from **823 ms into 43 ms** when the start of the prompt repeats (measured). Two tenants alternating on one replica evicted each other every time (945 ms vs 37 ms). Across three replicas with random routing, a tenant's next request lands on "its" warm replica only about one time in three; with affinity routing, nearly always after warm-up. For prompt-heavy workloads (RAG, long system prompts) that is the difference between ~0.9 s and ~0.05 s to first token — a bigger effect than adding a replica.

The trade-off: affinity concentrates load; least-outstanding spreads it. Production routers combine them — prefer the replica with the prefix, unless it is noticeably busier than the rest.

### Liveness, readiness, draining

| Signal | Question | Failing means |
|---|---|---|
| **Liveness** | is the process alive? | restart it |
| **Startup** | has it finished loading? | wait, do not restart yet |
| **Readiness** | should it receive traffic **now**? | take it out of rotation, do not restart |

A model server is **not ready** until its model is loaded *and warm* — a replica that passes readiness before its warm-up request sends the first users to a cold model. **Draining** is readiness in reverse: on shutdown, fail readiness, let the load balancer stop sending, finish in-flight streams, then exit. LLM streams last tens of seconds, so drain timeouts must be far longer than for typical web services.

### Rolling deploys, with numbers

Three replicas each running at the SLO-safe utilisation of ~0.30 ([07-operate.md §1](../07-operate.md)). Take one out to replace it and the other two absorb its traffic: 0.30 × 3 / 2 = **0.45** each — past the point where the SLO held. So deploy with **surge** (start the new replica, wait for readiness, then stop an old one: `maxSurge: 1, maxUnavailable: 0`), or keep one replica of headroom (N + 1).

---

## 3. Design options

| Layer | Options | Notes |
|---|---|---|
| Load balancer | nginx, HAProxy, Envoy, a cloud LB, a Kubernetes ingress/gateway | must not buffer streams; must support least-outstanding; health-gated |
| Routing by model | one pool per model behind a hostname or path; or an L7 router that reads `model` from the JSON body | per-model pools are simpler and avoid model-swap thrash |
| Weight distribution | baked into the image; a shared read-only volume; synced from object storage at start | content-addressed blobs make sync and verification trivial |
| Shared state | Redis for counters; Postgres (or an event log) for usage; SQLite replicated read-only for keys | each replaces one in-process component |

**Recommendation:** per-model pools behind Envoy or nginx with least-outstanding routing; weights synced from object storage into a read-only volume; Redis and Postgres for shared state; Kubernetes (or systemd + a load balancer) for lifecycle.

---

## 4. The design for shilp

### 4.1 Topology

```
                   clients
                      │ HTTPS
                      ▼
        ┌──────────────────────────────┐
        │ gateway: TLS, auth, IP limits│
        │ routes by model → pool       │
        └───────┬──────────────┬───────┘
     least outstanding         │ (+ prefix/tenant affinity, bounded)
        ┌───────┴──────┐ ┌─────┴────────┐
        ▼              ▼ ▼              ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ shilp   │  │ shilp   │  │ shilp   │   one model pool; each replica:
   │ replica │  │ replica │  │ replica │   /ready, drain, own caches
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        ├── read-only models volume (blobs + manifests, synced, sha256-verified)
        ├── Redis: rate-limit buckets, in-flight leases
        └── Postgres / event log: usage events → aggregates → billing
```

### 4.2 Readiness and real preloading

Today `/health` returns 200 whenever the process is up, and models load lazily on the first request — `shilp serve --model X` only sets the *default* model, despite its "preload" help text.

Design:

- `shilp serve --model X --preload` loads X at startup and sends one warm-up generation (optionally with a configured system prompt, so the prefix cache starts warm).
- New `GET /ready` returns **200** only when: the preloaded model is resident, warm-up finished, and the server is not draining. Otherwise **503** with a reason.
- `GET /health` stays as liveness.

### 4.3 Draining

On `SIGTERM`:

1. Set `draining = True` → `/ready` returns 503 → the load balancer stops sending new requests.
2. Keep serving in-flight streams (and anything already admitted).
3. uvicorn's graceful shutdown waits for open connections up to `timeout_graceful_shutdown` — exposed as `SHILP_DRAIN_TIMEOUT` (default 120 s) and passed to `uvicorn.run(...)` in `cli.py`.
4. In Kubernetes, add a short `preStop` sleep (5–10 s) so the endpoint removal reaches every proxy before the process stops accepting, and set `terminationGracePeriodSeconds` above the drain timeout.

### 4.4 Separate models from state

Today `$SHILP_HOME` holds both the model store and the writable telemetry database, so a shared **read-only** model volume is impossible. Split them:

| Variable | Contents | Replica access |
|---|---|---|
| `SHILP_MODELS_DIR` (default `$SHILP_HOME`) | `blobs/`, `manifests/`, `recipes/` | read-only |
| `SHILP_STATE_DIR` (default `$SHILP_HOME`) | `logs/shilp.db`, caches, key cache | writable, per replica |

Defaults keep today's single-directory behaviour. Replicas never pull at request time — admin endpoints stay off ([01](01-authentication.md)) — and a deploy step syncs blobs from object storage, verifying each blob's SHA-256 against its file name (the store is already content-addressed, so verification is free).

### 4.5 Reference configurations

**nginx** (per-model pool, least outstanding, no buffering):

```nginx
upstream shilp_gemma {
    least_conn;
    server 10.0.0.11:8000 max_fails=2 fail_timeout=10s;
    server 10.0.0.12:8000 max_fails=2 fail_timeout=10s;
    server 10.0.0.13:8000 max_fails=2 fail_timeout=10s;
}
server {
    listen 443 ssl;
    location ~ ^/(v1/(chat/)?completions|v1/embeddings|v1/models|api/(chat|generate|embed|tags))$ {
        proxy_pass http://shilp_gemma;
        proxy_http_version 1.1;
        proxy_buffering off;            # stream tokens as they come
        proxy_read_timeout 300s;        # longer than the longest answer
    }
}
```

Open-source nginx only does *passive* health checks (`max_fails`); use Envoy, HAProxy or Kubernetes readiness for active `/ready` checks.

**Kubernetes** (essentials of the Deployment):

```yaml
strategy: {type: RollingUpdate, rollingUpdate: {maxSurge: 1, maxUnavailable: 0}}
template:
  spec:
    terminationGracePeriodSeconds: 150
    containers:
    - name: shilp
      args: ["serve", "--model", "assistant:prod", "--preload", "--host", "0.0.0.0"]
      env: [{name: SHILP_MODELS_DIR, value: /models}, {name: SHILP_STATE_DIR, value: /state}]
      startupProbe:   {httpGet: {path: /ready,  port: 8000}, periodSeconds: 5, failureThreshold: 60}
      readinessProbe: {httpGet: {path: /ready,  port: 8000}, periodSeconds: 5}
      livenessProbe:  {httpGet: {path: /health, port: 8000}, periodSeconds: 10}
      lifecycle: {preStop: {exec: {command: ["sleep", "10"]}}}
      volumeMounts: [{name: models, mountPath: /models, readOnly: true}, {name: state, mountPath: /state}]
```

A plain Kubernetes `Service` spreads connections without regard to load; put a gateway with least-request balancing (Envoy Gateway, Istio, or a cloud L7 load balancer) in front for LLM traffic.

### 4.6 Scaling the pool

- **Size** from the SLO-safe throughput of one replica (measured with `scripts/loadtest.py`), plus one for headroom: `replicas = ceil(peak_rate / safe_rate_per_replica) + 1`. For the team-assistant example with serial replicas: ceil(0.42 / 0.24) + 1 = **3**.
- **Autoscale** on requests in flight or queue depth per replica, never on CPU. Remember the lag: provisioning a machine, syncing weights, loading and warming can take minutes, so keep minimum replicas ≥ 1 and target ~50–70% utilisation.
- **Pools per model**, each sized and scaled on its own traffic.

---

## 5. Step-by-step implementation

| Step | Change | Files |
|---|---|---|
| 1 | `--preload` that really loads and warms; `/ready` endpoint | `cli.py`, `server/app.py`, `engine/loader.py` |
| 2 | Draining on `SIGTERM`; `SHILP_DRAIN_TIMEOUT` → `timeout_graceful_shutdown` | `server/app.py` (lifespan), `cli.py` |
| 3 | `SHILP_MODELS_DIR` / `SHILP_STATE_DIR` split with backward-compatible defaults | `config.py`, `store.py`, `server/telemetry.py` |
| 4 | `shilp sync` (or a documented script) to fetch blobs from object storage and verify digests | `cli.py` / `scripts/` |
| 5 | Shared state: Redis limiter ([02](02-rate-limiting-and-quotas.md)), usage shipping ([06](06-metering-and-billing.md)), key store access ([01](01-authentication.md)) | those designs |
| 6 | Reference configs (nginx, Envoy, Kubernetes, systemd) | `deploy/` (new), `docs/` |
| 7 | Container images: CPU and CUDA; weights never in the image | `Dockerfile`s (P3.5) |

---

## 6. Testing and verification

- `/ready` is 503 before preload finishes, 200 after warm-up, 503 after `SIGTERM`.
- **Drain test:** start a long stream, send `SIGTERM`, assert the stream completes with `[DONE]` and new connections are refused after the drain window.
- A replica runs with a **read-only** models directory and a separate writable state directory.
- **Scaling test:** open-loop load test against 1 and against N replicas behind the balancer; SLO-safe throughput scales close to N×.
- **Rolling deploy under load:** no failed requests and no SLO breach with `maxSurge: 1, maxUnavailable: 0`.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Traffic sent to a cold replica | readiness only after warm-up |
| Deploys truncate answers | draining; grace period longer than the longest stream |
| Rolling deploy overloads the remaining replicas | surge, or N + 1 capacity |
| Hot replica from affinity routing | bounded-load hashing; fall back to least outstanding |
| Replicas disagree on limits or usage | shared Redis and usage store; never per-replica counters for billing |
| Weight sync corrupts a file | content-addressed names; verify SHA-256 before marking ready |
| Proxy buffering or short timeouts | `proxy_buffering off`, timeouts above the longest answer, tested through the proxy |

---

## 8. Status

| Item | Status |
|---|---|
| Stateless request handling | ✅ |
| Liveness (`/health`, answers during generation, lists resident models) | ✅ |
| Content-addressed blobs (easy to sync and verify) | ✅ |
| Graceful shutdown | 🟡 uvicorn default; no drain flag, no configured timeout |
| Real preload + warm-up, `/ready` | ⬜ P3.5 |
| Read-only models dir separate from writable state | ⬜ P3.5 |
| Shared rate limits / usage / keys | ⬜ P2.2, P3.3 |
| Reference LB, Kubernetes and container configs | ⬜ P3.4, P3.5 |
