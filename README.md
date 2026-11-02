# shilp

**Pull a model from Hugging Face, change it, serve it over an API that Ollama and OpenAI clients already understand — with your own code doing the serving.**

`shilp` (from *shilp* — to craft, to sculpt) is a small model-serving stack in about 4,100 lines you can read in an afternoon. It is not a wrapper around `ollama serve`. It is the thing Ollama is: a content-addressed model store, an inference engine, and an HTTP API — built from scratch so that every part of the pipeline is yours.

```bash
shilp pull hf:bartowski/google_gemma-3-1b-it-GGUF:q4_k_m        # 1. download
shilp build -f Shilpfile                                         # 2. reshape (params, template, quant, surgery)
shilp serve --model google_gemma-3-1b-it-gguf:q4_k_m             # 3. serve
curl localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"model":"google_gemma-3-1b-it-gguf:q4_k_m",
       "messages":[{"role":"user","content":"hello"}]}'           # 4. use it
```

---

## Why this exists

Anybody can call an API. The interesting engineering is underneath it: what a model actually is on disk, what happens between a prompt and a token, and how you expose that as a service that does not fall over.

Three things made this worth building rather than using Ollama:

1. **Ollama is a black box at exactly the layer I wanted to learn.** Which file gets loaded, how the chat template is applied, where the KV cache lives, why the second request is faster than the first — all of that is invisible from outside.
2. **The reshape step has no equivalent in Ollama.** Ollama can import a GGUF and change its parameters. It cannot convert a Hugging Face repo, prune transformer layers, truncate a vocabulary, merge a LoRA adapter, and hand you a recipe of what it did.
3. **The API surface is a real design problem.** Ollama's API and OpenAI's API describe the same operation with different types — nanoseconds and `done: true` versus seconds and a `[DONE]` sentinel. Serving both from one engine, honestly, is the exercise.

---

## The four steps

| Step | Command | What it does |
|---|---|---|
| 1. Download | `shilp pull` | Fetch a repo from the Hub into a content-addressed store, revision-pinned |
| 2. Reshape | `shilp build`, `convert`, `quantize`, `merge-lora`, `surgery` | Change the model: format, precision, parameters, template, structure |
| 3. Serve | `shilp serve` | One engine, two HTTP surfaces, streaming, telemetry, a chat UI |
| 4. Observe | `shilp ps`, `shilp usage`, `/metrics` | What is resident, how fast, how much |

---

## Quickstart

### One command

```bash
git clone https://github.com/vkdev01/shilp && cd shilp
make up
```

`make up` is `make setup` → `make pull` → `make serve`: a venv, the dependencies, a `.env`, a doctor report, a ~800 MB model (Gemma 3 1B), and a server on `:8000`. It detects Apple Silicon and builds llama.cpp with Metal, and the server offloads every layer to the GPU by default — on an M3 Pro that is the difference between 178 ms and 27 ms to the first token. Decode speed moves far less, and that asymmetry is the most useful thing to understand about local inference (see [Numbers that matter](#numbers-that-matter)).

`make help` lists every target. Requires Python 3.10+ and, on a Mac, Xcode command line tools (the engine compiles C++ at install time).

### Or step by step

```bash
make setup    # venv, deps, .env, then report what this machine has
make test     # 124 tests, no model, no network (7 skip without torch)
make pull     # Gemma-3-1B-it Q4_K_M, ~800 MB
make serve    # http://127.0.0.1:8000 — chat UI at /chat
```

Override what gets pulled and served without editing anything:

```bash
make up MODEL=hf:bartowski/Llama-3.2-3B-Instruct-GGUF:q4_k_m \
        REF=llama-3.2-3b-instruct-gguf:q4_k_m
```

### Without a model, or without make

The whole HTTP surface runs against a deterministic fake engine, so you can see the API before downloading anything:

```bash
make demo                        # or: pip install -e ".[dev]" && SHILP_ENGINE=fake shilp serve
```

```bash
curl -s localhost:8000/api/tags
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"model":"","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

Manual equivalent of `make setup`, if you would rather see each step:

```bash
python3 -m venv .venv && source .venv/bin/activate
CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[engine,dev]"   # Apple Silicon
# pip install -e ".[engine,dev]"                             # CPU only
cp .env.example .env
shilp doctor
```

Then:

```bash
# Ollama surface — streaming NDJSON
curl -N localhost:8000/api/chat -H 'content-type: application/json' -d '{
  "model": "google_gemma-3-1b-it-gguf:q4_k_m",
  "messages": [{"role": "user", "content": "Explain KV cache in two sentences."}]
}'

# OpenAI surface — SSE, so any OpenAI client works
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "google_gemma-3-1b-it-gguf:q4_k_m",
  "messages": [{"role": "user", "content": "Explain KV cache in two sentences."}],
  "stream": true
}'
```

Point an OpenAI SDK at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="google_gemma-3-1b-it-gguf:q4_k_m",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

---

## What it looks like inside

```
                       shilp CLI
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   ┌─────────┐     ┌────────────┐     ┌────────────┐
   │  hub.py │     │  modify/   │     │  server/   │
   │  pull   │     │  convert   │     │  FastAPI   │
   │  from   │     │  quantize  │     │  ┌──────┐  │
   │  HF Hub │     │  surgery   │     │  │/api  │  │  Ollama shape
   └────┬────┘     │  merge     │     │  ├──────┤  │
        │          └─────┬──────┘     │  │/v1   │  │  OpenAI shape
        └────────────────┼────────────┴──└──┬───┘──┘
                         ▼                  │
                 ┌───────────────┐          │
                 │  store.py     │          │
                 │  blobs/       │          │
                 │  manifests/   │◀─────────┘
                 │  recipes/     │
                 └───────┬───────┘
                         ▼
                 ┌───────────────┐
                 │  engine/      │
                 │  loader (LRU) │
                 │  llama.cpp    │
                 └───────────────┘
```

A model on disk is **content-addressed**: weights live in `blobs/sha256-<hex>`, and a manifest names them. Two tags that share a file store it once, and a manifest can never point at bytes that changed underneath it — the filename *is* the checksum. Same idea as OCI images, which is why the manifest format will look familiar.

```json
{
  "schemaVersion": 1,
  "name": "qwen2.5-1.5b-instruct-gguf",
  "tag": "q4_k_m",
  "layers": [{ "mediaType": "application/vnd.shilp.gguf", "digest": "sha256:…", "size": 1117320736, "name": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" }],
  "params": { "temperature": 0.6, "num_ctx": 8192 },
  "template": null,
  "metadata": { "source": "hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF", "quantization": "Q4_K_M" }
}
```

Every reshape step also appends to `recipes/<name>/<tag>.json`, so a model can explain how it was made:

```json
{ "steps": [
  { "step": "pull", "spec": "hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF", "revision": "main" },
  { "step": "quantize", "quant_type": "Q4_K_M" },
  { "step": "build", "params": { "temperature": 0.6 }, "template": true }
] }
```

---

## Code map

```
src/shilp/
  cli.py            typer commands, one per pipeline step
  config.py         settings from the environment; no config-file format
  store.py          blobs, manifests, recipes, garbage collection
  hub.py            Hugging Face download, behind a protocol so tests need no network
  modelfile.py      the Shilpfile format (FROM / PARAMETER / TEMPLATE / SYSTEM)
  export.py         materialise a manifest back into a plain directory
  modify/
    toolchain.py    find llama.cpp, run its binaries, explain what is missing
    convert.py      HF safetensors -> GGUF
    quantize.py     GGUF precision reduction, with the trade-offs written down
    surgery.py      prune layers, truncate vocab, rope scaling, chat template
    merge.py        fold a LoRA adapter into the base weights
  engine/
    base.py         Engine protocol, GenParams, Chunk, ContextOverflow
    llamacpp.py     the real backend (GGUF, mmap, Metal/CUDA offload)
    transformers_engine.py  the same protocol on plain PyTorch — no C++, no GGUF
    fake.py         deterministic backend, so the API is testable and demoable
    loader.py       one resident model, LRU eviction, reference counting
    template.py     render Ollama-style templates to a prompt
  server/
    app.py          create_app, error translation, health, metrics
    ui.py           browser pages: status at /, chat client at /chat
    templates/      Jinja2 pages (base, status, chat)
    static/         theme CSS (light + dark) and the chat/theme scripts
    routes_ollama.py  /api/chat, /api/generate, /api/tags, /api/pull, …
    routes_openai.py  /v1/chat/completions, /v1/models, /v1/embeddings
    generate.py     the single generation path both surfaces share
    stream.py       bridge a blocking generator to an async one, cancellable
    telemetry.py    one SQLite row per request
    metrics.py      Prometheus text, hand-written, no client library
```

Every module opens with a doc comment saying what it is for and, where it matters, what it deliberately does not do.

---

## Two engines, one protocol

`shilp serve --engine <name>` picks the backend. Nothing above it changes — same loader, same API surfaces, same telemetry.

| | `llamacpp` (default) | `transformers` |
|---|---|---|
| Weights | one GGUF file, quantised, mmap'd | a safetensors directory, dense |
| Get a model | `shilp pull hf:org/model-GGUF:q4_k_m` | `shilp pull hf:org/model` |
| Memory, 7B | ~4.3 GiB at Q4_K_M | ~14.2 GiB at fp16 |
| Install | compiles C++ (3–10 min) | `pip install -e ".[base]"` |
| Apple GPU | Metal | no |
| New architectures | after a GGUF port | day one |
| Debug a forward pass | read C++ | `print` the tensor |

`fake` is the third: no model, deterministic output, and the whole HTTP surface works, which is how the 124 tests run without weights or a network.

The `transformers` engine exists to make the boundary demonstrable rather than asserted — one new file, and `scripts/bare_inference.py` is the same code as a standalone script you can run without shilp at all.

---

## Numbers that matter

Everything below is either measured on the machine named or computed from a model's published config. The two regimes are the key idea: **prefill is compute-bound, decode is bandwidth-bound**, so a GPU helps the first enormously and the second barely at all.

**Measured** — Gemma-3-1B Q4_K_M on an M3 Pro, same build, layers on CPU versus on Metal:

| | CPU only | Metal | Ratio |
|---|---|---|---|
| Decode (bandwidth-bound) | 84.5 tok/s | 98.7 tok/s | 1.17× |
| Time to first token, short prompt | 178 ms | 27 ms | 6.7× |
| Prefill, 3,107-token prompt | 9.15 s | 1.59 s | 5.8× |

**Measured** — the same server under load: four simultaneous requests each streamed at full speed, but their first tokens arrived at 0, 1.6, 3.2 and 4.7 s, and aggregate throughput stayed at one stream's worth. A prompt that repeats a 1,712-token prefix cost 43 ms instead of 823 ms.

**Computed** — what a model costs to hold and how fast it can possibly go:

| Model | fp16 | Q4_K_M | KV per token | 8k context |
|---|---|---|---|---|
| Qwen2.5-1.5B | 2.88 GiB | 0.88 GiB | 28 KiB | 0.22 GiB |
| Qwen2.5-7B | 14.18 GiB | 4.34 GiB | 56 KiB | 0.44 GiB |

```
resident = weights + (KV_per_token × context × concurrent_sessions) + runtime
decode ceiling ≈ memory_bandwidth / model_size_in_bytes
concurrent_requests ≈ arrival_rate × latency
```

Three consequences worth internalising. Quantisation is the largest single lever: 3.3× less memory *and* 3.3× more throughput at Q4_K_M versus fp16. The KV cache is per session, so **20 users at 8k context cost 8.75 GiB** — as much as the weights. And 10 requests/second at 8 seconds each means **80 simultaneous generations**, which is why a serial server plateaus at a handful of users.

---

## Testing

```bash
make test      # 124 unit tests, no model, no network (7 skip without torch)
make smoke     # boots a real uvicorn server and exercises every endpoint over HTTP
make loadtest  # ARGS="--rate 0.5 --duration 60" against a running server
make lint      # ruff, if installed
```

The unit tests run against the `fake` engine, so they are fast and hermetic. The smoke test catches what in-process tests cannot: a stream that buffers instead of streaming, a wrong media type, a shutdown that hangs. The load test measures what neither can: latency under concurrency.

Three fixtures show the design is honest about dependencies: `FakeHub` implements the two-method hub protocol so `pull` is tested without the network; `tiny_hf_model` builds a two-layer safetensors model with numpy so layer pruning runs on real tensor files in milliseconds; and the fake engine lets the entire serving surface be tested with no weights at all.

---

## Running it as a service

shilp is a good *team* server and a good *learning* stack. It is not yet a public provider, and the gap is specific rather than vague:

| | Status |
|---|---|
| Two API dialects, streaming, usage reporting | implemented |
| Error codes that distinguish retryable from fatal | implemented |
| Cancellation — a client disconnect stops generation (logged as 499) | implemented |
| Telemetry, Prometheus metrics, TTFT and tok/s | implemented |
| Authentication, quotas, rate limiting | **missing** |
| A request queue with admission control | **missing** |
| Continuous batching | **missing** — `llama-cpp-python` is single-sequence |
| Metering and billing, multi-tenancy | **missing** |

It binds to `127.0.0.1` by default because of the first gap. Before exposing it, put a proxy in front that authenticates and terminates TLS, and set `flush_interval -1` on the streaming route — a proxy that buffers turns streaming into a 20-second wait followed by a wall of text.

The honest capacity note: on one box, requests serialise, so throughput is flat as concurrency rises. The next step on the same GGUF files is `llama-server --parallel N`, which batches (Metal included); on a CUDA box, vLLM. The engine protocol is shaped so either arrives as one file.

---

## What this is not

Being clear about the edges is more useful than claiming none:

- **Not multi-user.** No auth, no per-user quota.
- **Not concurrent by design.** One model resident at a time, one mutable KV cache, so requests take turns: each streams at full speed, the rest wait.
- **No continuous batching.** Measured above: four streams start 1.6 s apart.
- **No tool calling.** The chat template is rendered, the tokens come back; nothing parses `tool_calls`.
- **Sharded GGUFs are rejected** with a message that says so, rather than half-supported.
- **Surgery is lossy and unvalidated.** `prune_layers` will happily make your model worse. It reports what it did; measuring the damage is your job, and an eval harness is the top item on the roadmap.

---

## Documentation

The long-form docs live in a **private repository, [`shilp-docs`](https://github.com/vkdev01/shilp-docs)** — theory, pipeline detail, deployment and the system-design set. Access on request.

| Document | Covers |
|---|---|
| `00-theory.md` | What inference computes: tokens, the forward pass, KV cache, templates, formats, quantisation, the prefill/decode split, and a from-scratch implementation with no llama.cpp |
| `01-pull.md` · `02-reshape.md` | Getting models from the Hub; converting, quantising, pruning, merging |
| `03-serve.md` · `05-deploy.md` | The API surfaces and engines; capacity planning, reliability, security, cost, a reference deployment |
| `06-service-fundamentals.md` · `07-operate.md` · `08-service-roadmap.md` | Building blocks of an inference service; SLOs, load testing, rollout; what is implemented, partial and missing |
| `architecture.md` · `04-roadmap.md` | Design decisions and their trade-offs; what is deliberately not built |
| `system-design/` | One design each for authentication, rate limiting, admission control, batching, replicas, metering, multi-tenancy — with worked numbers and an implementation roadmap |
| `system-design/interview/` | A beginner system-design track ending in a mock LLM-inference interview |

---

## Licence

MIT.
