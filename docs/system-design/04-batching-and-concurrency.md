# 04 — Concurrency and batching

**Status:** ⬜ one sequence at a time per model · **Blocks:** throughput beyond one stream — this is the architectural limit · **Depends on:** nothing; changes how [03](03-admission-control.md) is sized · **Roadmap:** P3.1 (`llama-server` engine), P3.2 (vLLM engine) · **Effort:** 1–2 weeks for P3.1 including tests; P3.2 a few days more on a CUDA machine

---

## 1. The problem

### What happens with N users today

shilp drives llama.cpp through `llama-cpp-python`, whose high-level API runs **one sequence per model at a time**. With N concurrent requests, they take turns. **Measured** on one M3 Pro with Gemma-3-1B:

| Users | Throughput | Output tok/s (all users) | Each stream, once started | Median wait for first token |
|---|---|---|---|---|
| 1 | 0.81 req/s | 87 | 92 tok/s | 29 ms |
| 2 | 0.80 req/s | 87 | 91 tok/s | 1.4 s |
| 4 | 0.82 req/s | 87 | 91 tok/s | 3.7 s |
| 8 | 0.83 req/s | 88 | 91 tok/s | 8.5 s |

**Aggregate throughput is flat.** Adding users adds waiting, not output.

### "Each user gets 1/N" — true on average, wrong in experience

A common way to describe this is "N streams each get ~1/N of the throughput". Averaged over a busy period that is right: 16 users sharing a 7B model on an M2 Pro (~43 tok/s ceiling) receive about 43 / 16 ≈ **2.7 tok/s each** on average. But no one actually sees a 2.7 tok/s stream. Requests run **one after another at full speed**, so the cost is paid as waiting:

| 16 users at once | First user | 16th user |
|---|---|---|
| 1B on M3 Pro (~1.23 s per answer) | starts immediately | waits 15 × 1.23 ≈ **18 s** for the first token |
| 7B on M2 Pro (~6 s per 150-token answer at ~25 tok/s) | starts immediately | waits 15 × 6 ≈ **90 s** |

No queueing policy fixes this. A queue ([03](03-admission-control.md)) decides *who* waits and turns hopeless waits into fast 429s, but total throughput is set by the engine. To serve more people per machine, the engine must run **several sequences at once**.

---

## 2. Concepts

### Why one stream wastes the hardware

Each decode step produces one token per sequence. To do it, the device reads **every weight** once and performs about **2 floating-point operations per weight**. Reading 0.81 GB of weights at 150 GB/s takes **5.4 ms**; the arithmetic for a 1B model (≈2 GFLOP) takes a fraction of a millisecond on a GPU. The step is **memory-bandwidth-bound**: the arithmetic units sit idle most of the time ([00-theory.md §2.4](../00-theory.md)).

**Arithmetic intensity** — operations per byte read — is the number that says so: ~2 FLOP per weight, and a Q4 weight is ~0.6 bytes, so ~3–4 FLOP/byte. A modern GPU needs tens to hundreds of FLOP/byte before compute becomes the limit.

### What batching does

Advance **B** sequences in the same step. The weights are still read **once**; only the arithmetic and each sequence's own KV cache scale with B:

```
step_time(B) ≈ (weight_bytes + B × context × KV_bytes_per_token) / bandwidth      (while memory-bound)
aggregate tok/s = B / step_time(B)        per-stream tok/s = 1 / step_time(B)
```

**Computed** for Gemma-3-1B on the M3 Pro (0.81 GB weights, 26 KiB KV per token, 1k-token contexts), scaling the memory floor by the **measured** single-stream efficiency (5.6 ms floor vs 10.1 ms measured = 55%):

| Batch B | Step time (est.) | Aggregate tok/s | Per stream | KV memory at 4k context |
|---|---|---|---|---|
| 1 | 10.1 ms | ~99 (measured: 98.7) | 99 | 0.10 GiB |
| 4 | ~11 ms | ~360 | ~90 | 0.41 GiB |
| 8 | ~12 ms | ~650 | ~80 | 0.81 GiB |
| 16 | ~15 ms | ~1,070 | ~67 | 1.62 GiB |

These are **first-order estimates**, not measurements: they assume the step stays memory-bound, and at larger B the GPU's compute, kernel launch overhead and scheduling will cut into them. The direction is not in doubt — the same hardware serves several times more tokens per second when it batches — and every per-stream figure stays far above human reading speed (~5–8 tok/s), which is the slack batching spends.

For a 7B model (4.68 GB) on the same machine, the ceiling per stream is ~31 tok/s and the aggregate at B = 8 is ~210 tok/s (realistic ~115): batching helps proportionally just as much, but each stream is slower to begin with, so B is limited by keeping per-stream speed above ~15 tok/s.

### Kinds of batching

| Kind | How | Weakness |
|---|---|---|
| **Static** | gather B requests, run to completion together | everyone waits for the longest answer |
| **Dynamic** | wait up to *t* ms to fill a batch, then run | same: short answers held hostage |
| **Continuous** (iteration-level) | the batch is re-formed at **every token step**: finished sequences leave, waiting ones join | needs per-sequence KV management — the reason for paged attention |
| **Chunked prefill** | a long prompt's prefill is split into pieces interleaved with other sequences' decode | prevents one long prompt from freezing every stream |
| **Prefill/decode disaggregation** | prefill and decode on different machines, KV shipped between | large-scale only |

Continuous batching is what made LLM serving economical (Orca, OSDI 2022; vLLM, SOSP 2023).

### What limits B

1. **KV memory.** Each running sequence needs its own cache: `B × context × KV_per_token`. Paged attention (vLLM) allocates it in blocks on demand; `llama-server` preallocates `context / slots` per slot.
2. **Per-stream latency.** Each step gets slower as B grows; choose B so per-stream speed stays above your stream-speed SLO.
3. **Compute.** Eventually the arithmetic catches up with memory; past that point, more B adds latency without adding throughput.
4. **Prefill interference.** New arrivals' prefills share the device with everyone's decode; chunked prefill bounds the stall.

### Prefix caching with batching

A batching server can keep **several** prefixes warm — one per slot in `llama-server`, a radix tree of prefixes in SGLang, block-level caching in vLLM. That fixes the measured weakness of the current engine, where two tenants with different system prompts evicted each other's cache on every switch (945 ms vs 37 ms time-to-first-token, [05-deploy.md §4.6](../05-deploy.md)).

---

## 3. Design options

| Option | Hardware | Effort | Notes |
|---|---|---|---|
| **A. `llama-server` engine** — run llama.cpp's own server with `--parallel N`, forward to it | Apple Silicon (Metal), CUDA, CPU | 1–2 weeks | same GGUF blobs shilp already stores; continuous batching across slots; the only batching option on a Mac |
| **B. vLLM engine** | NVIDIA (CUDA), AMD (ROCm) | days, given A | paged attention, automatic prefix caching, chunked prefill; best concurrency per GPU; safetensors models (shilp's raw-pull path) |
| **C. Batch inside `llama-cpp-python`** with the low-level `llama_batch` API | any | weeks | re-implements a scheduler that `llama-server` already has; not recommended |
| **D. SGLang / TGI** | CUDA | days, given A | alternatives to B with similar properties |

A and B have the same shape — *start an inference server, forward to its OpenAI-compatible API* — so the design builds **one generic forwarding engine** and two launchers.

**Recommendation:** A first (works on the hardware this project targets), B for CUDA deployments.

---

## 4. The design for shilp

### 4.1 Shape

```
shilp (API, auth, limits, queue, metering)
   │  loader.acquire(model)
   ▼
UpstreamEngine ── implements Engine (chat/generate/embed/info/close), slots = N
   │  HTTP, streaming, on 127.0.0.1
   ▼
llama-server -m <blob.gguf> -c <N × ctx> --parallel N -ngl <layers>     (P3.1)
vllm serve <model dir> --max-num-seqs N                                  (P3.2)
```

shilp keeps everything a service needs — both API dialects, authentication, rate limits, admission, telemetry, the model store — and delegates the one thing it cannot do well in Python: scheduling many sequences on the accelerator.

### 4.2 The engine

New module `src/shilp/engine/upstream.py` (sketch):

```python
class UpstreamEngine:
    """Forwards to an OpenAI-compatible inference server it starts and owns."""

    def __init__(self, manifest: Manifest, launcher: Launcher, *, slots: int):
        self.ref, self.manifest, self.slots = manifest.ref, manifest, slots
        self._proc, self._base = launcher.start()          # subprocess + http://127.0.0.1:<port>
        self._http = httpx.Client(base_url=self._base, timeout=None)
        launcher.wait_ready(self._http)                    # GET /health until 200

    def chat(self, messages, params) -> Iterator[Chunk]:
        body = {"messages": [m.to_dict() for m in messages], "stream": True,
                "stream_options": {"include_usage": True}, **openai_sampling(params)}
        with self._http.stream("POST", "/v1/chat/completions", json=body) as res:
            if res.status_code != 200:
                raise EngineError(f"upstream {res.status_code}: {res.read()[:200]!r}")
            usage = {}
            for line in res.iter_lines():
                ...                                         # parse SSE → yield Chunk(text=...)
            yield Chunk(done=True, finish_reason=reason,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"))
        # Leaving the `with` closes the connection: a cancelled request stops its slot.

    def close(self) -> None:
        self._http.close(); self._proc.terminate(); ...     # wait, then kill after a grace period
```

Things that fall out of this shape:

- **It fits the existing bridge.** The engine stays a synchronous `Iterator[Chunk]`, so `server/stream.py` runs it on a worker thread exactly as it runs llama.cpp today — including cancellation: when shilp's stop flag closes the generator, the HTTP connection closes and the upstream stops that sequence. (`llama-server` cancels a task when its client disconnects; confirm on the pinned version with the test in §6.)
- **Exact token counts.** Usage comes from the upstream's own counters, which also removes today's re-tokenizing approximation for completion tokens.
- **Templates.** With no manifest template, the upstream applies the GGUF's chat template (`llama-server --jinja`). With a manifest template (a Shilpfile `TEMPLATE`), shilp renders the prompt itself and calls the raw `/completion` endpoint — the same split the llama.cpp engine makes today.
- **No engine lock.** Concurrency is the upstream's job; the loader's reference counting still prevents eviction mid-stream.

### 4.3 The launcher for `llama-server` (P3.1)

```
llama-server -m $SHILP_HOME/blobs/sha256-<digest> \
             -c <slots × num_ctx> --parallel <slots> \
             -ngl <SHILP_GPU_LAYERS> --host 127.0.0.1 --port <free port> --jinja --metrics
```

- **Context is divided between slots**: `-c 16384 --parallel 4` gives each slot 4,096 tokens. shilp computes `-c` from the manifest's `num_ctx` so a model's per-request context does not change when batching is switched on.
- **Binary**: `SHILP_LLAMA_SERVER` if set, else `llama-server` on `PATH` (on macOS, `brew install llama.cpp` provides it), else `$SHILP_LLAMACPP_DIR/build/bin/llama-server`. `shilp doctor` reports which one it found.
- **Lifecycle**: one upstream process per resident model, started on first use (like today's lazy load), stopped on eviction and at shutdown. If the process exits unexpectedly, the engine is marked failed, the loader drops it, and the next request starts a fresh one.
- **Flags change between llama.cpp releases**; pin the version in the deployment and keep the launcher's flag list in one place.

### 4.4 The launcher for vLLM (P3.2)

`vllm serve <model directory> --max-num-seqs <N> --port <free port>`, against the safetensors directory that the loader already materialises for the `transformers` engine. vLLM handles paged attention, prefix caching and chunked prefill itself; it runs on CUDA/ROCm, not Metal.

### 4.5 Admission with N slots

The queue in [03](03-admission-control.md) takes its slot count from configuration via `loader.slots_for(manifest)` — 1 for `llamacpp`, `SHILP_SLOTS` for an upstream engine — because admission happens before the engine is loaded; the launcher passes the same N to `--parallel`, so the two always agree. With N slots and a wait budget B, the bound becomes `max_waiting ≈ N × B / S`.

### 4.6 Choosing N — worked example

Gemma-3-1B on the M3 Pro, SLO: stream speed ≥ 20 tok/s, contexts up to 4k:

| Constraint | Limit on N |
|---|---|
| Memory: ~10 GB free for KV at 26 KiB/token × 4k = 0.1 GiB per slot | ~100 — not binding |
| Per-stream speed ≥ 20 tok/s (estimated ~67 at N = 16) | comfortably ≥ 16 |
| Diminishing returns as compute binds | find with the load test |

Start at **N = 4**, run `scripts/loadtest.py --concurrency 1,2,4,8` against it, and raise N while aggregate throughput still grows and per-stream speed and p95 time-to-first-token stay inside the SLO.

### 4.7 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SHILP_ENGINE` | `llamacpp` | add `llamaserver` and `vllm` |
| `SHILP_SLOTS` | `4` for upstream engines | concurrent sequences per model |
| `SHILP_LLAMA_SERVER` | found on `PATH` | path to the binary |
| `SHILP_UPSTREAM_START_TIMEOUT` | `120` s | how long to wait for the upstream's `/health` |

---

## 5. Step-by-step implementation

| Step | Change | Files |
|---|---|---|
| 1 | `UpstreamEngine` against an already-running OpenAI-compatible server (no launcher yet): SSE parsing, usage, errors, cancellation | `engine/upstream.py` (new) |
| 2 | Map `GenParams` to OpenAI/llama-server sampling fields | `engine/upstream.py` |
| 3 | `LlamaServerLauncher`: binary discovery, free port, flags, readiness wait, terminate/kill | `engine/upstream.py` |
| 4 | Register `llamaserver` in `ENGINES` / `load_engine`; loader passes slots; drop the engine lock for upstream engines | `engine/__init__.py`, `engine/loader.py` |
| 5 | Admission queue takes `engine.slots` | `engine/admission.py` (from [03](03-admission-control.md)) |
| 6 | `shilp doctor` reports the binary and version | `cli.py` |
| 7 | Docs and a measured throughput curve | `docs/` |
| 8 (P3.2) | `VllmLauncher` on the safetensors directory | `engine/upstream.py` |

---

## 6. Testing and verification

- **Unit, no binary needed:** `UpstreamEngine` against an in-process fake OpenAI server (an ASGI app or `httpx.MockTransport`) — streaming text, usage on the last frame, an upstream 500, a malformed frame, and **cancellation**: closing the generator closes the upstream request (the fake records it).
- **Launcher with a stub binary:** a tiny Python script standing in for `llama-server` that serves `/health` and one route — tests port selection, readiness timeout, and kill-on-close in CI without llama.cpp.
- **Integration (manual or nightly):** a real `llama-server` with a small GGUF; the smoke test passes unchanged against `SHILP_ENGINE=llamaserver`.
- **Acceptance (load test):** closed-loop curve with 1, 2, 4, 8 users shows **aggregate throughput rising** — e.g. at 4 users ≥ 2.5× the single-user figure — with per-stream speed ≥ 20 tok/s and p95 time-to-first-token at 4 users below 1 s.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Upstream process crashes | engine marked failed, loader drops it, next request restarts it; requests in flight get an error frame |
| Port collision | bind to port 0 or probe for a free port; pass it explicitly |
| Upstream slower to start than expected | readiness wait with a timeout; `/health` of shilp reports the model as not ready until it is |
| Flag or API changes between llama.cpp releases | pin the version; launcher flags in one place; integration test per upgrade |
| KV memory exhausted at high N | size N from memory first (§4.6); vLLM's paged attention degrades more gracefully |
| One long prompt stalls all slots | input caps ([02](02-rate-limiting-and-quotas.md)); chunked prefill (vLLM; llama-server processes prompts in batch-size chunks) |
| Two processes per model (shilp + upstream) to supervise | shilp owns the child's lifecycle; container deployments run them in one container |

---

## 8. Status

| Item | Status |
|---|---|
| Concurrency-safe server (no event-loop blocking, no eviction mid-stream) | ✅ |
| Engine protocol that a forwarding engine can implement | ✅ (`engine/base.py`) |
| Cancellation plumbing that a forwarding engine can reuse | ✅ (`server/stream.py`) |
| More than one sequence per model at once | ⬜ P3.1 |
| `llama-server` engine (Metal/CUDA/CPU) | ⬜ P3.1 |
| vLLM engine (CUDA) | ⬜ P3.2 |
| Multi-prefix caching | ⬜ comes with P3.1 / P3.2 |
