# Step 3 — Serving it

One engine, two HTTP surfaces, and the numbers to tell you whether it is working.

---

## Start it

```bash
shilp serve --model qwen2.5-1.5b-instruct-gguf:q4_k_m
# → http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` in a browser for a status page: stored models, which one is resident, and the endpoints.

```bash
shilp serve --engine fake --port 8000     # no model, deterministic output — for UI work
SHILP_MAX_RESIDENT=2 shilp serve          # keep two models warm, if you have the RAM
```

| Env var | Default | Meaning |
|---|---|---|
| `SHILP_HOME` | `~/.shilp` | store, manifests, telemetry |
| `SHILP_HOST` / `SHILP_PORT` | `127.0.0.1` / `8000` | bind address |
| `SHILP_MAX_RESIDENT` | `1` | models kept in memory at once |
| `SHILP_ENGINE` | `llamacpp` | `llamacpp` or `fake` |
| `SHILP_THREADS` | `0` (engine decides) | llama.cpp thread count |
| `SHILP_LLAMACPP_DIR` | — | llama.cpp checkout, for convert/quantize |
| `HF_TOKEN` | — | Hub token for gated repos |

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
        → route serialises to NDJSON or SSE
        → finally: telemetry row, metrics, refcount--
```

Four things this buys, each of which is a bug avoided:

1. **The event loop is never blocked.** `llama.cpp` blocks the thread it runs on, so it runs on a worker. `/health` still answers while a generation is streaming.
2. **A model is never evicted mid-stream.** `refs > 0` protects it; a second request for a different model gets a 503 rather than a truncated answer.
3. **Loading happens once.** Two simultaneous requests for a cold model wait on one lock and share one load, rather than racing to load it twice.
4. **Failures are recorded.** Telemetry is written in `finally`, so a 500 is a row with `status=500` — the rows you most want.

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
shilp_requests_total{endpoint="/api/chat",model="qwen…:q4_k_m",surface="ollama"} 42
# TYPE shilp_tokens_total counter
shilp_tokens_total{direction="completion",model="qwen…:q4_k_m"} 8831
# TYPE shilp_time_to_first_token_ms summary
shilp_time_to_first_token_ms_sum{model="qwen…:q4_k_m"} 6224
shilp_time_to_first_token_ms_count{model="qwen…:q4_k_m"} 42
# TYPE shilp_models_resident gauge
shilp_models_resident 1
```

Hand-written rather than using the Prometheus client library: four counters and two summaries do not justify a dependency, and writing the exposition format makes clear what is counted.

---

## Making it fast on a Mac

The defaults are reasonable; these are the knobs that actually matter.

1. **Metal, or it is CPU-only.** `CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[engine]"`. Without it you will see ~4 tok/s instead of ~50 and wonder why.
2. **Quantization beats everything else.** Q4_K_M on 7B is ~4.4 GB and fast; F16 is 15 GB and swaps.
3. **`num_ctx` costs memory linearly.** The KV cache is allocated up front. 32k context on a 7B model can be several GB on its own — set it in the Shilpfile to what you actually need, not to the maximum the model supports.
4. **Threads.** Leave `SHILP_THREADS=0` and let llama.cpp pick (it uses physical cores). Oversubscribing to logical cores is slower, not faster.
5. **Keep the model warm.** The first request after a load pays the mmap and KV allocation. Send a one-token warmup at startup if p99 latency matters.

### Rough expectations

| Machine | Model | Quant | tok/s |
|---|---|---|---|
| M1/M2 8 GB | 1.5B | Q4_K_M | 40–60 |
| M2 Pro 16 GB | 7–8B | Q4_K_M | 15–25 |
| M1 8 GB | 7B | Q4_K_M | 8–12 |
| x86 CPU, 8 cores | 1.5B | Q4_K_M | 8–15 |
| x86 CPU, 8 cores | 7B | Q4_K_M | 2–4 |

Ballparks, not benchmarks. `shilp usage` reports your real numbers, which is why the telemetry exists.

---

## Deployment notes

- **It binds to `127.0.0.1` deliberately.** There is no authentication. If you expose it, put it behind a reverse proxy that authenticates, and say so in the README of whatever you deploy.
- **One worker, not `--workers 4`.** Each worker would load its own copy of the model and hold its own KV cache. Concurrency comes from the loader queue, not from process replication.
- **Docker** works but adds ~1 GB of image for the toolchain; the roadmap covers it.

## Next

[Roadmap](04-roadmap.md) — what is deliberately missing and what to build next.
