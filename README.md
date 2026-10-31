# shilp

**Pull a model from Hugging Face, change it, serve it over an API that Ollama clients already understand — with your own code doing the serving.**

`shilp` (from *shilp* — to craft, to sculpt) is a small model-serving stack in about 4,100 lines you can read in an afternoon. It is not a wrapper around `ollama serve`. It is the thing Ollama is: a content-addressed model store, an inference engine, and an HTTP API — built from scratch so that every part of the pipeline is yours.

```bash
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m    # 1. download
shilp build -f Shilpfile                                 # 2. reshape (params, template, quant, surgery)
shilp serve --model qwen2.5-1.5b-instruct-gguf:q4_k_m         # 3. serve
curl localhost:8000/api/chat -d '{"model":"qwen2.5-1.5b-instruct-gguf:q4_k_m",
  "messages":[{"role":"user","content":"hello"}]}'       # 4. use it
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

| Step | Command | What it does | Doc |
|---|---|---|---|
| 1. Download | `shilp pull` | Fetch a repo from the Hub into a content-addressed store | [docs/01-pull.md](docs/01-pull.md) |
| 2. Reshape | `shilp build`, `convert`, `quantize`, `merge-lora`, `surgery` | Change the model: format, precision, parameters, template, structure | [docs/02-reshape.md](docs/02-reshape.md) |
| 3. Serve | `shilp serve` | One engine, two HTTP surfaces, streaming, telemetry | [docs/03-serve.md](docs/03-serve.md) |
| 4. Observe | `shilp ps`, `shilp usage`, `/metrics` | What is resident, how fast, how much | [docs/03-serve.md](docs/03-serve.md#observability) |

Design decisions and their trade-offs are in [docs/architecture.md](docs/architecture.md). What is not built yet is in [docs/04-roadmap.md](docs/04-roadmap.md).

---

## Quickstart

### One command

```bash
git clone https://github.com/vkdev01/shilp && cd shilp
make up
```

`make up` is `make setup` → `make pull` → `make serve`: a venv, the dependencies, a `.env`, a doctor report, a ~1 GB model, and a server on `:8000`. It detects Apple Silicon and builds llama.cpp with Metal — the difference between ~50 tok/s and ~4.

`make help` lists every target. Requires Python 3.10+ and, on a Mac, Xcode command line tools (the engine compiles C++ at install time).

### Or step by step

```bash
make setup    # venv, deps, .env, then report what this machine has
make test     # 78 tests, no model, no network
make pull     # Qwen2.5-1.5B-Instruct Q4_K_M, ~1 GB
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
curl -s localhost:8000/api/chat -d '{"model":"","messages":[{"role":"user","content":"hi"}],"stream":false}'
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
curl -N localhost:8000/api/chat -d '{
  "model": "qwen2.5-1.5b-instruct-gguf:q4_k_m",
  "messages": [{"role": "user", "content": "Explain KV cache in two sentences."}]
}'

# OpenAI surface — SSE, so any OpenAI client works
curl -N localhost:8000/v1/chat/completions -d '{
  "model": "qwen2.5-1.5b-instruct-gguf:q4_k_m",
  "messages": [{"role": "user", "content": "Explain KV cache in two sentences."}],
  "stream": true
}'
```

Point an OpenAI SDK at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen2.5-1.5b-instruct-gguf:q4_k_m",
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
    base.py         Engine protocol, GenParams, Chunk
    llamacpp.py     the real backend
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
    stream.py       bridge a blocking generator to an async one
    telemetry.py    one SQLite row per request
    metrics.py      Prometheus text, hand-written, no client library
```

Every module opens with a doc comment saying what it is for and, where it matters, what it deliberately does not do.

---

## Testing

```bash
make test     # 78 unit tests, no model, no network
make smoke    # boots a real uvicorn server and exercises every endpoint over HTTP
make lint     # ruff, if installed
```

The unit tests run against the `fake` engine, so they are fast and hermetic. The smoke test is the one that catches what in-process tests cannot: a stream that buffers instead of streaming, a wrong media type, a shutdown that hangs.

Two test fixtures are worth knowing about, because they show the design is honest about dependencies:

- `FakeHub` implements the two-method hub protocol, so `pull` is tested without touching the network.
- `tiny_hf_model` builds a two-layer safetensors model with numpy, so layer pruning is tested on real tensor files in milliseconds.

---

## What this is not

Being clear about the edges is more useful than claiming none:

- **Not multi-user.** There is no auth and no per-user quota. It binds to `127.0.0.1` by default for that reason.
- **Not concurrent by design.** One model is resident at a time and a GGUF has one mutable KV cache, so requests serialise on the engine lock. That is the correct trade-off for a laptop and the wrong one for a server.
- **No continuous batching.** llama.cpp's Python bindings do not expose it. A CUDA box should use vLLM, and the engine protocol is shaped so that backend can be added as one file.
- **No tool calling.** The chat template is rendered, the tokens come back; nothing parses `tool_calls`. [Roadmap](docs/04-roadmap.md).
- **Sharded GGUFs are rejected** with a message that says so, rather than half-supported.
- **Surgery is lossy and unvalidated.** `prune_layers` will happily make your model worse. It reports what it did; measuring the damage is your job, and the roadmap has the eval harness that would automate it.

---

## Numbers, honestly

On an M2 Pro with a 1.5B Q4_K_M model, expect roughly 40–60 tok/s and ~150 ms to first token for a short prompt. A 7B Q4_K_M lands nearer 15–20 tok/s. These are ballparks — `shilp usage` reports your actual figures, which is the point of building the telemetry.

The reason a laptop can do this at all is quantization plus mmap: a 4.4 GB file is not read into RAM, it is mapped, and the OS pages in what the model touches. Resident memory ends up close to the file size rather than several times it.

---

## Licence

MIT.
