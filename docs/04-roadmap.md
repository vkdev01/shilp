# Roadmap — what is missing, and why

A roadmap is only useful if it says what is *not* there. Each item below is a real gap in the current code, with the reason it was left and the shape the fix would take.

This is the *project* roadmap. Everything needed to run shilp as an inference service — authentication, limits, admission control, rate limiting, metering, batching engines, replicas, containers — is tracked item by item, with status and acceptance criteria, in the **[service roadmap](08-service-roadmap.md)**. Items 2, 3 and 7 below are summarised there in more detail.

---

## 1. A real evaluation harness

**Gap.** `surgery prune` and `quantize` change quality and shilp cannot measure the change. Right now the honest advice is "test it yourself", which is not engineering.

**Fix.** `shilp eval <ref> --suite gsm8k,lite --n 200`, reporting accuracy per task against the parent model, stored in the recipe. A small suite (50–200 items) that runs in minutes on a laptop would be enough to catch a bad prune or an over-aggressive quant. This is the highest-value item on the list, because it turns every other lossy operation from "trust me" into "measured".

## 2. Continuous batching: a `llama-server` engine, then vLLM

**Gap.** One model, one request at a time. **Measured** with four simultaneous requests: each streamed at full speed, but their first tokens arrived at 0, 1.6, 3.2 and 4.7 s, and aggregate throughput stayed at one stream's worth. On a laptop with one user that is correct. With several it is the ceiling ([05-deploy.md §4.3](05-deploy.md)).

**Fix.** Three options, in order of cost, and the engine protocol was shaped for all of them:

- **An engine that forwards to a local `llama-server --parallel N`.** It reads the same GGUF blobs shilp already stores, does continuous batching across slots, and runs on Metal — the only batching option on Apple Silicon. One file: start or attach to the server, translate `Chunk`s from its streaming API.
- **An `Engine` for vLLM**, which brings paged attention and continuous batching on NVIDIA GPUs. Roughly one file, plus a `gpu` extra in `pyproject.toml`.
- A micro-batch window inside the loader on top of `llama-cpp-python` — not recommended: the binding's high-level API is single-sequence, so this would mean reimplementing what `llama-server` already does.

## 3. Admission control

**Gap.** There is no bound on waiting. Requests for a busy model queue on the engine lock indefinitely, so overload shows up as ever-growing time to first token rather than as a fast, retryable rejection. The utilisation table in [05-deploy.md §4.4](05-deploy.md) is the argument: past ~80% of capacity, latency grows without limit.

**Fix.** A bounded `asyncio.Semaphore` per model in the loader (`SHILP_MAX_WAITING`, default a few requests), and a `429` with `Retry-After` when it is full — raised before the first byte, which the routes' stream priming already supports. Expose the number of waiting requests as a gauge in `/metrics`, which gives the queue-depth metric the deployment guide currently has to approximate from SQLite.

**Done since this was written: cancellation.** A disconnect now stops the worker at the next token and releases the model (it used to make the next request wait 6.9 s; now 216 ms). What remains is cancelling *during prefill*: the flag is only checked between tokens, so a long prompt still finishes prefill after its client has gone. llama.cpp's abort callback is the fix when the binding exposes it.

## 4. Tool calling

**Gap.** No `tool_calls` parsing on either surface. A model that emits a JSON tool call returns text.

**Fix.** Parse the model's tool-call syntax (per-family: Hermes JSON, Llama's `<|python_tag|>`, Qwen's `<tool_call>`) into OpenAI's `tool_calls` shape, and add a `tools` passthrough to the template renderer. Needs a per-architecture registry, which is why it is not a five-line change.

## 5. Sharded GGUFs

**Gap.** `select_files` rejects multi-part GGUF repos with a clear message. Larger models increasingly ship that way.

**Fix.** Download all shards, verify the split count and index, store them as separate layers of one manifest, and hand llama.cpp the first path (it resolves siblings itself).

## 6. Layer surgery on GGUF directly

**Gap.** Surgery happens on safetensors, which requires the float model — 3 GB for 1.5B, 15 GB for 7B. You cannot prune a model you could otherwise run.

**Fix.** GGUF is a documented container: a header, a metadata KV block, tensor info, then tensor data. Dropping layers means rewriting the tensor-info table and the offsets, and decrementing `block_count`. Doable and fiddly; worth doing after the eval harness exists so the result can be measured.

## 7. Authentication and multi-tenancy

**Gap.** No auth, binds to localhost. Deliberate, but it means the thing cannot be shared.

**Fix.** An optional bearer token (`SHILP_API_KEY`) checked by a dependency, plus per-key usage rows so `shilp usage --key` works. Small; skipped only because a single-user local server does not need it and half-built auth is worse than none.

## 8. A Docker image and a `compose` file

**Gap.** No container. `pip install` plus a llama.cpp build is two steps too many for a demo.

**Fix.** Multi-stage build: compile llama.cpp in one stage, copy the binary and `convert_hf_to_gguf.py` into a slim runtime, mount `SHILP_HOME` as a volume so models survive the container. Apple Silicon users should still run natively — Metal does not pass through to Docker.

## 9. A fine-tuning pipeline that closes the loop

**Gap.** `merge-lora` exists but nothing produces an adapter. The pipeline is download → reshape → serve, with the training step assumed to happen elsewhere.

**Fix.** `shilp finetune --base <ref> --data <jsonl> --method lora|grpo`, emitting an adapter that `merge-lora` consumes. This is the natural next project, and it connects to the reasoning-model work: GRPO on a small model, evaluated by the harness in item 1, served by the engine that already exists. That would make the four steps a closed loop: *pull → tune → quantize → serve → measure → tune again*.

## 10. Memory knobs for concurrency

**Gap.** The levers that decide how many sessions fit in memory are not exposed: llama.cpp's KV-cache type (`q8_0` roughly halves the cache), flash attention (required for a quantised V cache), and sliding-window caches for models such as Gemma 3, whose most layers only need a 512-token window — the build measured here logs `using full-size SWA cache`.

**Fix.** Manifest parameters (`PARAMETER kv_cache_type q8_0`, `PARAMETER flash_attn true`) passed through to `Llama(...)`, so they live with the model in its Shilpfile, alongside `num_ctx`. Measure each with the eval harness (item 1) before making it a default.

## 11. Small things

- `shilp pull --resume` reporting byte progress through a callback instead of three coarse phases.
- A `--json` flag on every CLI command, for scripting.
- Speculative decoding (a small draft model plus a large target), which llama.cpp supports and the loader could own.
- Histograms instead of summaries in `/metrics`, so dashboards can show p95 time to first token without querying SQLite.
- `load_duration` on the Ollama surface is always 0; report the loader's acquire time, as the route's docstring promises.
- Accept Ollama requests without a `Content-Type: application/json` header, as Ollama does — `curl -d` without `-H` currently gets a 422 on `/api/*`.
- An OpenAI-compatible `/v1/audio/*` is out of scope; that is a different kind of model.

---

## What I would do next, in order

1. **Evaluation harness.** Everything lossy becomes defensible.
2. **Admission control.** Small, and it turns overload from "everything is slow" into "a few requests are told to retry".
3. **A `llama-server` engine.** Real concurrency on the hardware this project targets, and proof the engine abstraction was worth having; vLLM after it for CUDA.
4. **Fine-tuning pipeline.** Turns four steps into a loop.
5. **Docker.** Distribution, not engineering — do it when someone else needs to run this.
