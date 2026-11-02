# Step 3 — Serving it

One engine, two HTTP surfaces, and the numbers to tell you whether it is working.

---

## Start it

```bash
shilp serve --model google_gemma-3-1b-it-gguf:q4_k_m    # or: make serve
# → http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` in a browser for a status page: stored models, which one is resident, and the endpoints.

Open `http://127.0.0.1:8000/chat` for a chat client. It calls the public API and nothing else, so it doubles as a manual test bench:

- **API surface toggle** — send through Ollama `/api/chat` (NDJSON) or OpenAI `/v1/chat/completions` (SSE); the conversation carries over when you switch.
- **Settings** — model, system prompt, temperature, max tokens, streaming on/off. They persist in `localStorage`.
- **Per-reply stats** — completion tokens, tok/s, time to first token, wall time; *Stop* aborts mid-stream, and the server stops generating too (the request is logged as status 499).
- **Last request** — the exact call as a copyable `curl`, plus the raw response body or stream.
- **Theme** — light and dark, following the OS until you toggle.

Markup lives in `server/templates/` (Jinja2) and styling and scripts in `server/static/`; `server/ui.py` only gathers the data each page renders.

```bash
shilp serve --engine fake --port 8000     # no model, deterministic output — for UI work
SHILP_MAX_RESIDENT=2 shilp serve          # keep two models warm, if you have the RAM
```

| Env var | Default | Meaning |
|---|---|---|
| `SHILP_HOME` | `~/.shilp` | store, manifests, telemetry |
| `SHILP_HOST` / `SHILP_PORT` | `127.0.0.1` / `8000` | bind address — currently honoured only through `shilp serve --host / --port`; the variables are read into settings but not used by `serve` |
| `SHILP_MAX_RESIDENT` | `1` | models kept in memory at once |
| `SHILP_ENGINE` | `llamacpp` | `llamacpp`, `transformers` or `fake` |
| `SHILP_THREADS` | `0` (engine decides) | llama.cpp thread count |
| `SHILP_GPU_LAYERS` | `-1` (all) | layers offloaded to Metal/CUDA; `0` keeps everything on the CPU |
| `SHILP_LLAMACPP_DIR` | — | llama.cpp checkout, for convert/quantize |
| `HF_TOKEN` | — | Hub token for gated repos |

---

## Choosing an engine

Two real backends sit behind the same `Engine` protocol, so everything else in
this document — both API surfaces, streaming, telemetry, eviction — is identical
whichever you pick.

```bash
shilp serve --model qwen:q4_k_m                     # llama.cpp (default)
shilp serve --model qwen:main --engine transformers # PyTorch, dense weights
shilp serve --engine fake                           # no model at all
```

| | `llamacpp` | `transformers` |
|---|---|---|
| Weights | one GGUF file, quantised | a safetensors directory, dense |
| Get a model | `shilp pull hf:org/model-GGUF:q4_k_m` | `shilp pull hf:org/model` |
| Memory, 7B | ~4.4 GiB (Q4_K_M) | ~14 GiB (fp16) |
| Decode, 7B on M2 Pro | ~43 tok/s ceiling | ~13 tok/s ceiling |
| Install | compiles C++ (3–10 min) | `pip install -e ".[base]"` |
| Apple GPU | Metal | no |
| New architectures | after a GGUF port | day one |
| Debug a forward pass | read C++ | `print` the tensor |

**Why both ship.** `llamacpp` is the right default: quantisation is what makes a
7B model fit on a laptop, and no Python path can match it. `transformers` exists
because the boundary in `engine/base.py` should be demonstrable rather than
asserted — one new file, no changes to the server — and because it is the only
option for an architecture the GGUF ecosystem has not ported yet.

The trade-off is quantified in [00-theory.md](00-theory.md) §5 and §8; the
one-line summary is **3–7× the memory and 3–7× slower decode**, because decode is
bandwidth-bound and dense weights are 3–7× larger.

A model is not portable between the two: a GGUF cannot be read by `transformers`,
and a safetensors directory cannot be served by llama.cpp without
`shilp convert`. Each engine names the fix when handed the wrong shape.

---

## The Ollama surface

Faithful enough that Ollama's own clients work. Streaming is NDJSON: one JSON object per line, the last one carrying `done: true`.

### Chat

```bash
curl -N localhost:8000/api/chat -d '{
  "model": "qwen2.5-1.5b-instruct-gguf:q4_k_m",
  "messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is a KV cache?"}
  ],
  "options": {"temperature": 0.4, "num_predict": 128}
}'
```

```
{"model":"qwen…","created_at":"2026-11-01T09:12:03Z","message":{"role":"assistant","content":"A"},"done":false}
{"model":"qwen…","created_at":"…","message":{"role":"assistant","content":" cache"},"done":false}
…
{"model":"qwen…","created_at":"…","message":{"role":"assistant","content":""},"done":true,
 "done_reason":"stop","total_duration":1841200000,"prompt_eval_count":24,
 "eval_count":58,"eval_duration":1612000000}
```

`"stream": false` returns a single object of the same final shape.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | conversational completion |
| `POST` | `/api/generate` | raw completion, no chat formatting |
| `POST` | `/api/embed` | embeddings |
| `GET` | `/api/tags` | list stored models |
| `POST` | `/api/show` | one model's Shilpfile, template, metadata |
| `POST` | `/api/pull` | download from the Hub through the API |
| `POST` | `/api/delete` | remove a model, garbage-collect blobs |
| `GET` | `/api/ps` | models resident in memory |
| `GET` | `/api/usage` | aggregate latency and throughput |

Durations are integer **nanoseconds**, matching Ollama. `load_duration` is the acquire time (usually 0 for a warm model); `eval_duration` is generation time, excluding prompt processing.

---

## The OpenAI surface

The same engine behind `/v1`, so any OpenAI client works by changing one URL.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="qwen2.5-1.5b-instruct-gguf:q4_k_m",
    messages=[{"role": "user", "content": "What is a KV cache?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

| Method | Path |
|---|---|
| `POST` | `/v1/chat/completions` (SSE when `stream: true`) |
| `POST` | `/v1/completions` |
| `POST` | `/v1/embeddings` |
| `GET` | `/v1/models`, `/v1/models/{id}` |

Streaming uses `data: {…}` frames terminated by `data: [DONE]`. `usage` is present in the final chunk — many providers omit it while streaming, and clients that want to count tokens should not have to guess.

Not implemented, and the API says so rather than silently ignoring it: `n > 1`, `logprobs`, and tool calling.

### Wiring existing tools

| Tool | Setting |
|---|---|
| `openai` SDK (Python/JS) | `base_url="http://127.0.0.1:8000/v1"` |
| LangChain | `ChatOpenAI(base_url=…, api_key="x", model="<ref>")` |
| LlamaIndex | `OpenAILike(api_base="http://127.0.0.1:8000/v1", …)` |
| Continue / Open WebUI | provider `openai`, base URL `http://127.0.0.1:8000/v1` |
| Ollama clients | point at `http://127.0.0.1:8000` directly |

---

## How a request is actually served

```
request → resolve model ref → merge params (manifest defaults + request)
        → loader.acquire(ref)            one resident model, LRU, refcounted
        → engine.chat(...)               blocking generator, llama.cpp
        → worker thread drains it        server/stream.py
        → first token awaited            before any header is sent
        → route serialises to NDJSON or SSE
        → finally: telemetry row, metrics, refcount--
```

Four things this buys, each of which is a bug avoided:

1. **The event loop is never blocked.** `llama.cpp` blocks the thread it runs on, so it runs on a worker. `/health` still answers while a generation is streaming.
2. **A model is never evicted mid-stream.** `refs > 0` protects it; a second request for a different model gets a 503 rather than a truncated answer.
3. **Loading happens once.** Two simultaneous requests for a cold model wait on one lock and share one load, rather than racing to load it twice.
4. **Failures are recorded — with the status the client saw.** Telemetry is written in `finally`, so a failure during generation is a row: `400` for a prompt that does not fit, `503` for a busy model, `499` for a client that left, `500` for a real fault. Requests rejected *before* generation — an unknown model (404) or a body that fails validation (422) — are not recorded yet; the access log has them.
5. **Errors have a status code, even when streaming.** The route waits for the first token before it sends headers, so anything that fails first — including a prompt longer than `num_ctx` — is an ordinary JSON error. A failure after that ends the stream with an error frame (`{"error": …}` on NDJSON, `data: {"error": …}` and no `[DONE]` on SSE) rather than a silently cut connection.
6. **An abandoned answer stops.** When the client disconnects, the worker stops at the next token and closes the llama.cpp generator, which frees the model for whoever is waiting. Requests for the same model otherwise run one at a time, each at full speed — see [05-deploy.md §4.3](05-deploy.md) for what that means under load.

---

## The API contract — and hardening it for real clients

A client integrates with a contract, not with your code: what it may send, what it gets back, and what each failure looks like. This table is the contract as it stands, with what a public endpoint needs on top. Legend as in the [service roadmap](08-service-roadmap.md): ✅ in place, 🟡 partial, ⬜ missing.

### What a client may send

| Aspect | Today | Needed before exposure |
|---|---|---|
| Content type | ✅ JSON bodies; anything else is a 422 | ⬜ accept Ollama requests without `Content-Type`, as Ollama does |
| Body size | ⬜ unlimited | a cap at the proxy (Caddy `request_body { max_size 1MB }`, nginx `client_max_body_size 1m`) and in the app |
| Prompt length | ✅ counted after the chat template; over `num_ctx` → 400 naming both numbers, even when streaming | — |
| Number of messages | ⬜ unlimited | a cap, so a client cannot send 10,000 empty turns |
| Output length | 🟡 `num_predict` defaults to 512, but `max_tokens` / `options.num_predict` from the client override it up to the context | ⬜ a server-side ceiling per model or per key |
| Generation time | ⬜ no deadline; generation ends at a stop token or the length limit | ⬜ a per-request deadline that cancels the generation |
| Sampling | ✅ temperature, top-p, top-k, min-p, repeat penalty, stop sequences, `seed` | — |
| Determinism | 🟡 `seed` plus `temperature: 0` repeats on the same build and hardware; not across builds | document it; do not promise bit-exact output |
| Structured output (JSON mode, schemas) | ⬜ | llama.cpp supports grammar-constrained decoding; expose `response_format` / `format` |
| Tool calls, `n > 1`, `logprobs` | ⬜ | see the [project roadmap](04-roadmap.md) |

### What it gets back

| Aspect | Today |
|---|---|
| Streaming framing | ✅ NDJSON (Ollama) and SSE with `[DONE]` (OpenAI) |
| Token usage | ✅ on the final frame and in non-streaming bodies; prompt tokens include the chat template |
| Finish reason | 🟡 always `stop`, even when the answer was cut off by the length limit — truncation is invisible until this is fixed |
| Request id | ⬜ OpenAI responses carry a generated `id`, but there is no `X-Request-ID` header and the id is not logged |
| Rate-limit headers | ⬜ there is no rate limiting yet ([06-service-fundamentals.md §5](06-service-fundamentals.md)) |
| CORS | ⬜ no CORS headers: the built-in `/chat` page works (same origin), a browser app on another origin cannot call the API. When added, use an explicit origin allowlist, never `*` on an authenticated API |

### Status codes and error bodies

| Status | Meaning in shilp | Retry? |
|---|---|---|
| 400 | invalid request, or the prompt does not fit the context window | no — fix the request |
| 404 | unknown model | no |
| 422 | body failed schema validation (or was not JSON); FastAPI's `{"detail": [...]}` body | no |
| 429 | *(not emitted yet — reserved for rate limits and a full queue)* | yes, after `Retry-After` |
| 499 | *(recorded in telemetry, never sent)* the client disconnected mid-answer | — |
| 500 | engine failure | maybe, once |
| 502 | the Hugging Face Hub failed during `/api/pull` | yes, with backoff |
| 503 | the model is busy and another one is resident (`SHILP_MAX_RESIDENT`) | yes, with backoff |

Error bodies today are `{"error": "<message>", "type": "<kind>"}` on both surfaces (except 422s, above). That is Ollama's shape. OpenAI's is nested — `{"error": {"message": …, "type": …, "code": …}}` — so OpenAI SDKs still raise the right exception class (they branch on the status code) but may not show the message cleanly. 🟡 Matching each surface's shape is a small Phase 1 item. Errors *inside* a stream already use each dialect's own shape.

Once a stream has started, its status is fixed at 200, so a failure is reported as a final error frame and the stream ends without `done` / `[DONE]` — clients should treat that as a failed request ([architecture.md §5](architecture.md)).

### Admin endpoints are part of the attack surface

`/api/pull` downloads arbitrary models and `/api/delete` removes them. They exist for Ollama compatibility and are **not authenticated**. Until admin separation lands ([08-service-roadmap.md](08-service-roadmap.md), Phase 1), expose only the inference paths through your proxy — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags` — and keep everything else on loopback.

---

## Observability

### Per request, in SQLite

```bash
shilp usage
```

```json
{
  "requests": 42,
  "prompt_tokens": 1204,
  "completion_tokens": 8831,
  "avg_ttft_ms": 148.2,
  "avg_tok_per_s": 46.7,
  "best_tok_per_s": 58.1,
  "by_model": [{ "model": "qwen…:q4_k_m", "requests": 42, "avg_tok_per_s": 46.7 }]
}
```

`ttft_ms` is time to first token — the number a user feels. `tok_per_s` is completion tokens over generation time, excluding load and prompt processing, so it does not move when your prompt gets longer.

### Prometheus

```bash
curl -s localhost:8000/metrics
```

```
# TYPE shilp_requests_total counter
shilp_requests_total{endpoint="/api/chat",model="qwen…:q4_k_m",status="200",surface="ollama"} 41
shilp_requests_total{endpoint="/api/chat",model="qwen…:q4_k_m",status="499",surface="ollama"} 1
# TYPE shilp_tokens_total counter
shilp_tokens_total{direction="completion",model="qwen…:q4_k_m"} 8831
# TYPE shilp_time_to_first_token_ms summary
shilp_time_to_first_token_ms_sum{model="qwen…:q4_k_m"} 6224
shilp_time_to_first_token_ms_count{model="qwen…:q4_k_m"} 42
# TYPE shilp_models_resident gauge
shilp_models_resident 1
```

Hand-written rather than using the Prometheus client library: four counters and two summaries do not justify a dependency, and writing the exposition format makes clear what is counted. A summary with only `_sum` and `_count` gives you the **mean**; for p95 time to first token, query the per-request rows in `logs/shilp.db` — [05-deploy.md §11](05-deploy.md) has the SQL.

---

## Making it fast on a Mac

The defaults are reasonable; these are the knobs that actually matter.

1. **Build with Metal, and let it offload.** `CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[engine]"` (what `make setup` does on Apple Silicon) compiles the GPU backend; `SHILP_GPU_LAYERS=-1`, the default, puts every layer on it. **Measured** on an M3 Pro with Gemma-3-1B Q4_K_M: CPU-only decodes at 84.5 tok/s and Metal at 98.7 — decode is bandwidth-bound and both share the same memory — but time to first token drops from 178 ms to 27 ms, and a 3,107-token prompt prefills in 1.6 s instead of 9.2 s. The GPU's win is prefill. Look for `offloaded N/N layers` when the model loads with verbose logging, or simply a short-prompt TTFT in tens of milliseconds.
2. **Quantization beats everything else.** Q4_K_M on 7B is ~4.4 GB and fast; F16 is 15 GB and swaps.
3. **`num_ctx` costs memory linearly.** The KV cache is allocated up front. 32k context on a 7B model can be several GB on its own — set it in the Shilpfile to what you actually need, not to the maximum the model supports.
4. **Threads.** Leave `SHILP_THREADS=0` and let llama.cpp pick (it uses physical cores). Oversubscribing to logical cores is slower, not faster.
5. **Keep the model warm.** The first request after a load pays the mmap and KV allocation. Send a one-token warmup at startup if p99 latency matters.
6. **Put the stable part of the prompt first.** llama.cpp re-uses the KV cache for whatever prefix the new prompt shares with the previous one. **Measured**: a 1,712-token prompt took 823 ms to its first token cold and 43 ms when only the final question changed. A system prompt that starts with a timestamp, or two users with different system prompts taking turns, throws that away on every request.

### Rough expectations

| Machine | Model | Quant | tok/s |
|---|---|---|---|
| M1/M2 8 GB | 1.5B | Q4_K_M | 40–60 |
| M2 Pro 16 GB | 7–8B | Q4_K_M | 15–25 |
| M1 8 GB | 7B | Q4_K_M | 8–12 |
| x86 CPU, 8 cores | 1.5B | Q4_K_M | 8–15 |
| x86 CPU, 8 cores | 7B | Q4_K_M | 2–4 |
| M3 Pro 18 GB (**measured**) | Gemma-3-1B | Q4_K_M | 98.7 (Metal) / 84.5 (CPU) |

Ballparks, not benchmarks. `shilp usage` reports your real numbers, which is why the telemetry exists.

---

## Deployment notes

- **It binds to `127.0.0.1` deliberately.** There is no authentication. If you expose it, put it behind a reverse proxy that authenticates, and say so in the README of whatever you deploy.
- **One worker, not `--workers 4`.** The mmap'd weights would be shared through the page cache, but each worker holds its own KV cache, its own GPU buffers and its own lock, and all of them compete for the same memory bandwidth — more memory for no more throughput. Concurrency on one box comes from batching (`llama-server` slots), not from process replication; see [05-deploy.md](05-deploy.md).
- **Docker** works but adds ~1 GB of image for the toolchain; the roadmap covers it.

## Next

[Service fundamentals](06-service-fundamentals.md) — the components of an inference service, and which ones shilp has.

[Deployment and scaling](05-deploy.md) — what changes when other people call this, and what limits it.

[Operating it](07-operate.md) — SLOs, load testing, rollout, observability, runbooks.

[Service roadmap](08-service-roadmap.md) — implemented vs planned, phase by phase. [Project roadmap](04-roadmap.md) — everything else.
