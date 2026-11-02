# 00 — Current state and capabilities

**What shilp is today, what it can do, how well, and exactly where it stops — the baseline every design in this folder starts from.**

---

## 1. In one sentence

shilp is a **single-process model server** that downloads models from Hugging Face, can reshape them, and serves them through **both the Ollama and the OpenAI HTTP APIs** with streaming, telemetry and a browser chat page — correct and observable for **one user or a small trusted group on one machine**, and not yet safe to expose beyond that.

On the readiness ladder of [08-service-roadmap.md](../08-service-roadmap.md) it is at **L0 — local**.

---

## 2. Architecture as built

```
                       shilp CLI (typer)                       browser
                            │                                     │
     ┌──────────────────────┼─────────────────────┐               │ /  /chat
     ▼                      ▼                     ▼               ▼
┌─────────┐          ┌─────────────┐       ┌──────────────────────────────────────┐
│ hub.py  │          │  modify/    │       │ server/  (FastAPI, one uvicorn worker)│
│ pull    │          │  convert    │       │                                      │
│ from HF │          │  quantize   │       │  routes_ollama.py   /api/*           │
└────┬────┘          │  surgery    │       │  routes_openai.py   /v1/*            │
     │               │  merge-lora │       │  ui.py              / , /chat        │
     │               └──────┬──────┘       │  generate.py  one generation path,   │
     ▼                      ▼              │               prime() before headers │
┌──────────────────────────────────────┐   │  stream.py    worker thread ↔ async, │
│ store.py — content-addressed blobs,  │◄──┤               cancellation flag      │
│ manifests (name:tag), recipes        │   │  telemetry.py SQLite row per request │
│ ~/.shilp  (SHILP_HOME)               │   │  metrics.py   Prometheus text        │
└──────────────────────────────────────┘   └───────────────┬──────────────────────┘
                                                           │ loader.acquire(model)
                                                           ▼
                                           ┌──────────────────────────────────────┐
                                           │ engine/loader.py — resident models,  │
                                           │ LRU eviction, reference counts       │
                                           ├──────────────────────────────────────┤
                                           │ engine/llamacpp.py  GGUF, Metal/CUDA │
                                           │ engine/transformers_engine.py  dense │
                                           │ engine/fake.py  tests and UI work    │
                                           └──────────────────────────────────────┘
```

About 4,800 lines of Python in `src/shilp`, 124 tests, three scripts (`smoke_test.py`, `loadtest.py`, `bare_inference.py`).

---

## 3. Capabilities

### 3.1 Model lifecycle (CLI)

| Command | Does |
|---|---|
| `shilp pull hf:org/repo[-GGUF:quant]` | download from the Hub into the content-addressed store; `--dry-run` lists files and sizes |
| `shilp ls`, `show`, `rm`, `gc` | list, inspect, remove a tag, delete unreferenced blobs |
| `shilp build -f Shilpfile` | derive a model: parameters, template, system prompt; `FROM` an existing model shares its blobs (aliases such as `assistant:prod`) |
| `shilp convert`, `quantize` | safetensors → GGUF, re-quantise (needs a llama.cpp checkout) |
| `shilp merge-lora`, `surgery prune / vocab / rope / template` | structural edits on Hugging Face weights |
| `shilp serve` | run the API server |
| `shilp ps`, `usage`, `doctor`, `version` | resident models, usage summary, environment check |

### 3.2 HTTP API

| Surface | Endpoints |
|---|---|
| OpenAI | `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, `GET /v1/models`, `GET /v1/models/{id}` |
| Ollama | `POST /api/chat`, `POST /api/generate`, `POST /api/embed`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `POST /api/pull`, `POST /api/delete` |
| Ops | `GET /health`, `GET /metrics`, `GET /api/usage` |
| UI | `GET /` (status page), `GET /chat` (chat client for both surfaces) |

Behaviour worth knowing:

- **Streaming** in both dialects (SSE with `[DONE]`, NDJSON with `done: true`), from one shared generation path.
- **Errors before the first token** — unknown model, busy model, prompt longer than the context — come back as real status codes even for `stream: true`, because routes await the first token before sending headers. **Errors after it** end the stream with an error frame in that dialect.
- **Cancellation** — when a client disconnects mid-answer, generation stops at the next token and the model is released; recorded as status 499.
- **Token usage** on every response; prompt tokens count the rendered prompt including the chat template.

### 3.3 Engines

| Engine | Format | Notes |
|---|---|---|
| `llamacpp` (default) | GGUF | quantised, mmap'd, all layers offloaded to Metal/CUDA by default (`SHILP_GPU_LAYERS`) |
| `transformers` | safetensors directory | dense weights, PyTorch; for experiments and new architectures |
| `fake` | none | deterministic; for tests and UI work |

### 3.4 Observability

- **Per-request rows** in `$SHILP_HOME/logs/shilp.db` for requests that reach generation: surface, endpoint, model, status, prompt and completion tokens, time to first token, total time, tok/s. The status is what the client saw, except that a failure after streaming has started is recorded as 500 (the client got 200 plus an error frame) and 499 marks a client that left. No prompt or answer text is stored.
- **Prometheus** at `/metrics`: requests by status, tokens by direction, mean durations, models resident.
- **Load testing**: `scripts/loadtest.py` — closed and open loop, percentiles, SLO gate.

### 3.5 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SHILP_HOME` | `~/.shilp` | store, manifests, telemetry |
| `SHILP_HOST` / `SHILP_PORT` | `127.0.0.1` / `8000` | read into settings, but **`shilp serve` uses its own `--host` / `--port` flags** (same defaults) and ignores these — pass the flags |
| `SHILP_MAX_RESIDENT` | `1` | models kept in memory |
| `SHILP_ENGINE` | `llamacpp` | `llamacpp`, `transformers`, `fake` |
| `SHILP_THREADS` | `0` | llama.cpp threads (0 = auto) |
| `SHILP_GPU_LAYERS` | `-1` | layers on the GPU (-1 = all) |
| `SHILP_LLAMACPP_DIR`, `HF_TOKEN` | — | toolchain for convert/quantize; Hub token |

---

## 4. How well it performs (measured earlier, not re-run)

One shilp process, M3 Pro (18 GB), Gemma-3-1B Q4_K_M, Metal:

| Quantity | Value |
|---|---|
| Decode speed, one stream | 98.7 tok/s |
| Time to first token, short prompt | 27 ms |
| Prefill rate, long prompts | ~2,000 tok/s |
| Same 1.7k-token prefix repeated | 823 ms → 43 ms time to first token |
| Throughput, any number of users | ~0.81 req/s, ~87 output tok/s (flat) |
| Four requests at once | first tokens at 0, 1.6, 3.2, 4.7 s; each streams at full speed |
| Load meeting "95% of first tokens < 2 s" | ~0.24 req/s (≈30% of throughput) |

The shape matters more than the numbers: **throughput does not grow with users**, and extra users wait in line. That is the defining limit of the current engine path ([04](04-batching-and-concurrency.md)).

---

## 5. Where it stops, topic by topic

| Topic | In place | Missing | Design |
|---|---|---|---|
| **1. Authentication** | binds to `127.0.0.1` by default | any caller check; admin endpoints (`/api/pull`, `/api/delete`) are as open as inference | [01](01-authentication.md) |
| **2. Rate limiting & quotas** | a hard context-window check (400) | per-key requests/tokens per minute, in-flight caps, quotas, `max_tokens` ceiling | [02](02-rate-limiting-and-quotas.md) |
| **3. Queue & admission** | requests for the same model wait on the engine lock; a different model while one is busy → 503; abandoned *generating* requests stop | a bound, a fast 429 with `Retry-After`, FIFO order, removal of clients who leave while *waiting*, queue metrics | [03](03-admission-control.md) |
| **4. Batching** | correct concurrency (no blocking, no eviction mid-stream) | more than one sequence per model at a time | [04](04-batching-and-concurrency.md) |
| **5. Replicas & LB** | stateless API; `/health` lists resident models | a readiness endpoint, drain on shutdown, shared counters and usage, read-only model store separate from writable state | [05](05-replicas-and-load-balancing.md) |
| **6. Metering & billing** | one row per request that reaches generation, with tokens and true status | key/tenant and request id on rows, rows for rejections, exact output counts, aggregation, price book, invoices | [06](06-metering-and-billing.md) |
| **7. Tenancy & abuse** | no prompt text stored anywhere | tenants, isolation, input caps, admin separation, moderation hook, abuse detection | [07](07-multi-tenancy-and-abuse.md) |

### Smaller known gaps (tracked in the roadmap)

- `finish_reason` is always `stop`, even when output was cut off by the length limit.
- `shilp serve` ignores `SHILP_HOST` / `SHILP_PORT`; only the `--host` / `--port` flags set the bind address.
- `/api/pull` downloads on the event loop, so other requests stall while a pull runs.
- `--revision` is ignored for GGUF pulls.
- Completion tokens are counted by re-tokenizing the output text — close, not guaranteed exact.
- Ollama routes reject requests without `Content-Type: application/json` (Ollama accepts them).
- `load_duration` is always 0 on the Ollama surface.
- A loaded model is cached by name until eviction or restart, so repointing an alias needs a restart.

---

## 6. What "done" looks like for each level

| Level | Who can use it | Needs from this folder |
|---|---|---|
| **L0 Local** — today | you | — |
| **L1 Team** | a trusted team behind a proxy | 01 (single key), 03 (bounded queue), input caps from 07 |
| **L2 Multi-tenant** | several teams or customers | 01 (per-tenant keys), 02, 06 (metering), 07 |
| **L3 Scaled** | real traffic | 04 (batching engine), 05 (replicas, shared state) |
| **L4 Provider** | paying customers | 06 (billing) plus the operational practice in [07-operate.md](../07-operate.md) |

The build order and estimates are in [08-implementation-roadmap.md](08-implementation-roadmap.md).
