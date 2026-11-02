# Architecture and the decisions behind it

This is the document I would want if I inherited the repo. It is organised as decisions, because the structure follows from about eight choices and very little else.

---

## 1. Layers, and the rule that each one only talks downwards

```
cli ──▶ hub / modify ──▶ store ──▶ engine ──▶ server
```

| Layer | Knows about | Must not know about |
|---|---|---|
| `store` | files, digests, manifests | HTTP, llama.cpp, the Hub |
| `hub` | the Hub client | how a model is served |
| `modify` | llama.cpp's tools, safetensors | serving |
| `engine` | one loaded model, token streams | HTTP, request bodies |
| `server` | FastAPI, both API shapes | how weights are loaded |

The payoff is concrete: the server can be tested with an engine that has no model behind it, and the store can be tested with no engine at all. When a layer starts importing upwards, that is the signal that a responsibility is in the wrong place.

## 2. Content-addressed storage

**Decision.** Weights are stored as `blobs/sha256-<hex>`; a manifest lists layers with digests.

**Why.** Three properties come free:

- *Dedupe.* Quantizing `Q4_K_M` and `Q5_K_M` from the same F16 file stores the F16 once while both exist. More usefully, deriving a model with new parameters copies zero bytes — the layers are shared.
- *Integrity.* A manifest cannot point at bytes that changed, because the name is the hash. A corrupted download fails at import, not at inference.
- *Atomicity.* `put_blob` writes then renames. A reader sees the old file or the new one, never half of either.

**Cost.** Writing a manifest means a garbage-collection pass to reclaim orphans, so `store.delete()` calls `gc()`. That is a real cost at scale and irrelevant at this scale.

This is OCI's model, and Ollama's. Copying a good idea is not a weakness; inventing a worse one to be original would be.

## 3. A manifest is the model, not a directory

**Decision.** Everything needed to serve a model lives in one JSON document: layers, default sampling parameters, chat template, system prompt, metadata, parent digest.

**Why.** It makes "change the model" a small, reversible, inspectable operation. Deriving `qwen:creative` from `qwen:base` with `temperature 0.9` writes one JSON file and no weights. The chat template is per-tag, so two tags over identical weights can behave completely differently — which is usually the real problem people have with local models.

**Cost.** A template in the manifest can disagree with the template baked into the GGUF. The manifest wins, and `shilp show` prints both so the disagreement is visible rather than mysterious.

## 4. Recipes: provenance as a first-class artifact

**Decision.** Every mutating command appends a step to `recipes/<name>/<tag>.json`.

**Why.** The reshape step is where reproducibility dies in most projects. "Which base model, at which revision, quantized how, with which template?" is unanswerable six weeks later. A recipe answers it, and it is the file that turns a demo into something defensible.

**Cost.** Recipes are advisory, not enforceable — nothing prevents hand-editing a manifest out of sync with its recipe. A content digest over the recipe would fix that if it ever mattered.

## 5. Streaming is the primitive; non-streaming is a fold over it

**Decision.** `Engine.chat()` and `Engine.generate()` return `Iterator[Chunk]`. A non-streaming response drains the same iterator.

**Why.** One code path means one place for token counting, one place for cancellation, one place for the "done" semantics. The alternative — two paths — is how you end up with streaming and non-streaming responses that report different token counts.

`Chunk.done=True` arrives exactly once, last, carrying `prompt_tokens` and `completion_tokens`. Both API surfaces depend on that contract: Ollama puts them in its final JSON object, OpenAI in a trailing chunk with empty choices.

**Streaming changes when errors can be reported.** Once headers are sent the status is fixed at `200`, so the routes `prime` the stream: they await the first chunk *before* returning the response. Everything that can fail early — model resolution, loading, a 503 from the loader, a prompt longer than the context window — then reaches the ordinary exception handlers and becomes a status code. A failure after the first token ends the stream with an error frame in that surface's dialect. The cost is that headers wait for the first token, which is the moment the client is waiting for anyway.

## 6. The blocking-to-async bridge

**Decision.** `server/stream.py` runs the blocking generator in a worker thread and hands chunks back through an unbounded `asyncio.Queue`.

**Why it is needed.** `llama.cpp`'s Python binding blocks the calling thread for the whole generation. Running it on the event loop would freeze every other request, including `/health`.

**Why the queue is unbounded.** A bounded queue has two failure modes and no upside: `put_nowait` from the worker raises `QueueFull` (dropping tokens silently) or blocks the worker while the loop is busy (deadlock-shaped). Backpressure comes from the client instead — if the HTTP peer stops reading, the socket write in the route blocks, the generator is not drained, and the worker stalls. That is the correct place for it.

**Cancellation.** A queue only carries data one way, so stopping needs a second channel: a `threading.Event`. When the consumer goes away — the client disconnected and Starlette cancelled the response — the bridge sets the flag; the worker sees it at the next chunk, stops, and closes the source generator *from its own thread*, which runs the engine's `finally`, closes the llama.cpp stream and releases the engine lock.

**Why it is not optional.** This was first written as "the worker finishes the generation and the output is discarded — wasted work, not a correctness bug". Measuring it showed the cost was not the wasted tokens but the **lock**: the abandoned generation kept the model, so the next request waited for the rest of an answer nobody would read — 6.9 s in the measurement (600 tokens requested, client gone after 0.5 s), 216 ms after the fix. In a serial server, wasted work *is* latency for someone else.

**Remaining cost.** The flag is checked between chunks, so a cancellation during a long prefill takes effect only when the first token arrives.

## 7. One resident model, LRU eviction, reference counts

**Decision.** `ModelLoader` holds up to `SHILP_MAX_RESIDENT` engines (default 1), evicts the least recently used idle one, and refuses to evict a model with `refs > 0`.

**Why.** The constraint is memory, not cleverness. A 7B Q4 model is ~4.4 GB resident; a laptop cannot hold two, and thrashing between them is worse than queueing. The reference count exists because evicting a model mid-stream closes its KV cache under a live request — the kind of bug that shows up as a truncated answer, once a week, in production.

**Why `acquire` is a context manager.** So the release cannot be forgotten, and so `refs` is decremented even when the route raises.

**Cost.** A second request for a *different* model while the first is streaming gets a 503 rather than waiting. That is honest — the alternative is a request that hangs for 30 seconds and then fails. It is a *real* 503 even for `stream: true`, because routes await the first chunk before sending headers (§5): a streaming response that had already said `200` could only have cut the connection.

## 8. Two API surfaces, one translation point

**Decision.** Ollama's shapes and OpenAI's shapes are separate pydantic models, translated in the routes, over a shared `server/generate.py`.

**Why not one unified internal shape.** Because the two surfaces genuinely differ in observable ways:

| | Ollama | OpenAI |
|---|---|---|
| Stream framing | NDJSON, one JSON object per line | SSE, `data: …` then `data: [DONE]` |
| Terminator | final object with `done: true` | `[DONE]` sentinel |
| Durations | integer nanoseconds | seconds / `created` unix time |
| Token counts | `prompt_eval_count`, `eval_count` | `usage.{prompt,completion,total}_tokens` |
| Model naming | `name:tag` | opaque `id` |

Pretending they are the same produces a schema that lies to one client. Translating at the edge, visibly, means a bug in one surface is a bug in one route.

## 9. Telemetry as a table, not a dashboard

**Decision.** One SQLite row per request, written after the response finishes; Prometheus text for counters.

**Why.** `ttft_ms` and `tok_per_s` are the two numbers that decide whether a local model is usable, and neither is visible without instrumenting. A table is queryable, greppable and free; a dashboard is a week of work and a dependency.

**Why write-behind.** The write happens in `finally`, after the stream closes, so a failed request is still recorded — with `status=500`. Failures are the rows you want most.

**Cost.** SQLite writes are synchronous in the async path. At this request rate that is microseconds; at 100 rps it would need a queue and a writer task.

## 10. Optional dependencies are imported lazily, and their absence is a sentence

**Decision.** `huggingface_hub`, `llama_cpp`, `torch`, `transformers`, `peft` and `safetensors` are imported inside the function that needs them, and a missing one raises `MissingDependency` with the exact command to install it.

**Why.** `import shilp` must never require 2 GB of torch. And a stack trace ending in `ModuleNotFoundError: No module named 'peft'` is a worse experience than one sentence naming the fix.

---

## Request lifecycle, end to end

```
POST /api/chat
  │
  ├─ pydantic validates the body                      server/schemas.py
  ├─ state.resolve_model("") → "tiny:q4"              server/state.py
  ├─ manifest = store.resolve(ref)                    store.py
  ├─ params  = GenParams.merge(manifest.params, opts) engine/base.py
  │
  ├─ loader.acquire(ref)                              engine/loader.py
  │     ├─ resident? reuse : evict LRU idle model, then build
  │     └─ refs += 1                                  (blocking build in a thread)
  │
  ├─ engine.chat(messages, params) → Iterator[Chunk]  engine/llamacpp.py
  │     └─ blocking; drained in a worker thread       server/stream.py
  │
  ├─ per chunk: rec.first_token() on the first text
  ├─ per chunk: serialise to NDJSON or SSE            routes_*.py
  │
  ├─ final chunk: rec.finish(tokens)
  └─ finally: telemetry.record(rec); metrics counters; refs -= 1
```

## Failure modes, and what each one looks like

| Failure | HTTP | Where it is decided |
|---|---|---|
| Unknown model ref | 404 `model_not_found` | `StoreError` handler |
| Malformed ref (`hf:` in a model field) | 400 `bad_request` | `RefError` handler |
| Model has no GGUF yet | 503 `model_unavailable` | `ModelNotReady` |
| All slots busy with a live stream | 503 `model_unavailable` | `ModelNotReady` |
| `llama-cpp-python` not installed | 500 `missing_dependency` + install command | `MissingDependency` |
| Hub unreachable / gated repo | 502 `hub_error` | `HubError` |
| `llama-quantize` failed | 500 `tool_failed` + last 12 lines of stderr | `ToolError` |

One table, one place to change, both surfaces behave the same way.
