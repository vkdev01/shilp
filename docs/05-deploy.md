# 05 — Serving your own model like a provider

**From "a model on Hugging Face" to "an endpoint my app can call like any hosted API" — and what limits it once real traffic arrives.**

The other documents cover what inference *is* ([00-theory.md](00-theory.md)), how to get a model ([01-pull.md](01-pull.md)), how to change it ([02-reshape.md](02-reshape.md)) and how to run it locally ([03-serve.md](03-serve.md)). This one is about the step after that: putting it on a machine that other people's code calls, and being honest about what breaks. If terms like admission control, token buckets or continuous batching are new, read [06-service-fundamentals.md](06-service-fundamentals.md) first.

The short version, since it is the thing most guides bury: **a laptop serves one user well and ten users badly, and no amount of tuning changes the shape of that curve.** The reasons are arithmetic, and they are in §4.

---

## 1. What "like a provider" actually means

Clients do not integrate with *your* server. They integrate with a contract, and they notice immediately when it is missing. If you want `openai-python`, LangChain, LlamaIndex, Continue, Open WebUI or a Vercel app to work unchanged, these are the obligations:

| Obligation | Why the client cares |
|---|---|
| **Endpoint shapes** — `POST /v1/chat/completions`, `GET /v1/models` | hard-coded in SDKs |
| **SSE streaming** — `data: {…}` frames, terminated by `data: [DONE]` | the client's parser waits for the sentinel; without it, it hangs until timeout |
| **`usage` with token counts** | billing, context-budget tracking, cost display |
| **A model id that resolves** | the client sends whatever string you told it to |
| **Error codes with meaning** — 400 vs 429 vs 500 vs 503 | retry logic branches on the code, not the message |
| **Retry semantics** — a 429 or 503 is retryable, a 400 is not | otherwise clients retry your bugs forever |
| **Stable latency** | a p99 of 40 s reads as an outage |

shilp implements the first five in both dialects — Ollama's NDJSON and OpenAI's SSE — from one generation path, so a bug in one surface is a bug in one route rather than two implementations that disagree. What it does **not** do, and what you must add before exposing it, is everything in §8 and §9.

One detail of the error contract is easy to get wrong with streaming, and it is worth knowing why shilp does it the way it does. A streaming response commits to `200` the moment its headers leave. So shilp waits for the **first token before sending headers**: anything that fails earlier — unknown model (404), a busy model (503), a prompt longer than the context window (400) — comes back as a real status code with a JSON body. Anything that fails *after* the first token can no longer change the status, so it ends the stream with an error frame instead: `{"error": "…"}` on the Ollama surface (what Ollama itself sends) and `data: {"error": {…}}` with no `[DONE]` on the OpenAI surface (the OpenAI SDKs raise on it). A client that treats "stream ended without `done`/`[DONE]`" as a failure is then always right.

The honest framing: matching the API is a weekend. Matching the *operational* contract — auth, quotas, rate limits, retries, graceful degradation — is the actual work of being a provider.

---

## 2. The shortest path

Four commands, from nothing to an endpoint a client can call:

```bash
pip install -e ".[engine]"        # on Apple Silicon: CMAKE_ARGS="-DGGML_METAL=on" first
shilp pull hf:bartowski/google_gemma-3-1b-it-GGUF:q4_k_m
shilp serve --model google_gemma-3-1b-it-gguf:q4_k_m --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://your-host:8000/v1", api_key="unused")
print(client.chat.completions.create(
    model="google_gemma-3-1b-it-gguf:q4_k_m",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

That is a working private endpoint. Everything below is about what happens when the second user arrives. (The examples use the Makefile's default model; any GGUF works the same way. `--host 0.0.0.0` exposes it to your network with no authentication — read §9 before you do that anywhere but a lab.)

Two decisions to make before you start, because they are expensive to reverse:

**Which engine.** `llamacpp` is the only one to deploy. It has quantisation (the single largest lever on both memory and speed) and mmap. `transformers` is for experiments and architectures GGUF has not ported; it has no batching and 3–7× the memory. `fake` is for UI work.

**Which model.** Pick the smallest that passes your task, not the largest that fits. A 1.5B Q4_K_M answering a support-classification question is usually indistinguishable from a 70B and 50× cheaper to serve. §6 puts this first among the scaling levers for a reason.

---

## 3. Deployment topologies

| Topology | What it buys | What it costs |
|---|---|---|
| **Laptop, localhost** | zero cost, zero exposure, instant | one user, no uptime, no auth |
| **Single VM, CPU** | always on, a real IP | 5–15 tok/s for a 7B; fine for a team, not for a product |
| **Single VM, one GPU** | 30–80 tok/s single stream; the realistic starting point | GPU rental, driver/CUDA setup, one model per process |
| **One GPU, several models** | routing by task | memory, and shilp holds one model resident by default |
| **One box, `llama-server` with parallel slots** | continuous batching on the same GGUF, including on Apple Silicon | a different server process; the context is divided between slots |
| **N replicas behind a proxy** | linear throughput, rolling restarts | each replica holds its own KV cache (and, on a GPU, its own copy of the weights) |
| **GPU cluster + vLLM** | paged attention, continuous batching at high concurrency | CUDA/ROCm only — no Metal; real ops work; no longer a laptop project |

The jump that matters is the last one. Everything up to "N replicas" is a stateless web service with an unusually large binary asset; the cluster tier is a different discipline.

The `llama-server` row is the rung most guides skip, and it is the natural next step *for this project*: it reads the same GGUF files shilp stores, and its `--parallel N` slots batch several sequences per weight read — the concurrency lever of §4.3 — without leaving llama.cpp or the Mac. shilp's engine protocol is shaped so an engine that forwards to a local `llama-server` is one new file (see the [roadmap](04-roadmap.md)).

A useful test of where you are: **if a request can take 20 seconds, you are not serving a web app, you are serving a job queue that happens to speak HTTP.** Design accordingly — timeouts, streaming, and admission control matter more than request throughput.

---

## 4. Capacity planning: the four numbers

Every sizing decision reduces to four quantities. Get them wrong and no amount of tuning helps; get them right and the rest is tuning.

### 4.1 Memory: weights + cache × sessions + runtime

```
resident = weights
         + (KV_bytes_per_token × context × concurrent_sessions)
         + runtime_overhead
```

The middle term is the one people forget, because it is *per session* and grows with context. **Computed** for a 7B model (56 KiB per token):

| Concurrent sessions | Context | KV cache |
|---|---|---|
| 1 | 4,096 | 0.22 GiB |
| 10 | 4,096 | 2.19 GiB |
| 20 | 8,192 | 8.75 GiB |
| 50 | 8,192 | 21.88 GiB |
| 100 | 4,096 | 21.88 GiB |

Twenty users with generous context cost as much memory as the weights themselves. This is the single most common capacity surprise, and it is why "it ran fine in testing" and "it OOMs under load" are both true.

**The per-token figure is an architecture choice, and it varies ~20×.** It is `2 × layers × kv_heads × head_dim × bytes` (§2.3 of [00-theory.md](00-theory.md)), and *grouped-query attention* — fewer KV heads than query heads — is what moves it. **Computed** at fp16, from each model's published config:

| Model | Layers | KV heads × dim | KV per token | 4k-context sessions per GiB of KV |
|---|---|---|---|---|
| Llama-2-7B (no GQA) | 32 | 32 × 128 | 512 KiB | 0.5 |
| Llama-3-8B | 32 | 8 × 128 | 128 KiB | 2 |
| Qwen2.5-7B | 28 | 4 × 128 | 56 KiB | 4.6 |
| Gemma-3-1B (shilp's default) | 26 | 1 × 256 | 26 KiB | 9.8 |

Read the last column as *concurrency per gigabyte*. Two 7B-class models with the same weights footprint can differ 9× in how many simultaneous users fit, which makes KV-per-token a model-selection criterion for serving, not a footnote. Two further levers exist in llama.cpp and are not yet exposed by shilp: **KV-cache quantisation** (`q8_0` keys and values roughly halve the cache, at a small quality cost; a quantised V cache needs flash attention) and **sliding-window attention** for models that use it — Gemma 3 restricts most layers to a 512-token window, but the llama.cpp build measured here logs `using full-size SWA cache`, so that saving is not realised by default.

Full budget for 7B at 8k context, **computed**:

| Precision | Weights | KV | Runtime | Total | Fits 8 GB? | 16 GB? |
|---|---|---|---|---|---|---|
| fp16 | 14.19 | 0.44 | 0.7 | 15.33 GiB | no | barely |
| Q8_0 | 7.54 | 0.44 | 0.7 | 8.68 GiB | no | yes |
| Q6_K | 5.82 | 0.44 | 0.7 | 6.96 GiB | yes | yes |
| Q4_K_M | 4.34 | 0.44 | 0.7 | 5.48 GiB | yes | yes |

Note that this is the *single-session* budget. Add 0.22 GiB per additional session at 4k, and it stops fitting very quickly on a 16 GB box.

### 4.2 Throughput: bandwidth ÷ bytes read per token

```
single-stream ceiling ≈ memory_bandwidth / model_size_in_bytes
```

**Computed** for a 7B at Q4_K_M (4.66 GB):

| Machine | Bandwidth | Single-stream ceiling | Realistic |
|---|---|---|---|
| M1 | 68 GB/s | 15 tok/s | 7–10 |
| M2 Pro | 200 GB/s | 43 tok/s | 21–30 |
| RTX 4090 | ~1000 GB/s | 215 tok/s | 100–150 |

Real throughput lands at 50–70% of the ceiling. If you measure far below it, the model is not in memory (it is being re-read from disk) or the kernels are not vectorised for your CPU.

**Measured**, to check the model against a real machine: Gemma-3-1B Q4_K_M (0.81 GB) on an M3 Pro (150 GB/s) has a ceiling of ~186 tok/s and decodes at **98.7 tok/s** with every layer on the GPU — 53% of the ceiling, inside the predicted band. With the same build forced onto the CPU it decodes at 84.5 tok/s. That 17% gap is small *because* decode is bandwidth-bound and Apple's CPU and GPU share the same memory; the difference shows up in prefill instead (§4.6).

### 4.3 Concurrency: what happens when a second user arrives

This is where most intuition is wrong, and the deciding factor is not CPU versus GPU — it is **whether the engine batches**.

**Without batching (shilp today, on any hardware):** one `llama.cpp` context per model, guarded by a lock, so requests for the same model run **one after another** — in arrival order in practice, though Python's lock does not guarantee it (a designed FIFO queue is roadmap P1.4). The intuitive picture — "four users each get a quarter of the speed" — is not what happens. **Measured** on an M3 Pro with Gemma-3-1B, four identical requests fired at once:

| Request | Waited before its first token | Streamed at | Finished after |
|---|---|---|---|
| 1st | 0.0 s | 99 tok/s | 1.6 s |
| 2nd | 1.6 s | 97 tok/s | 3.2 s |
| 3rd | 3.2 s | 95 tok/s | 4.7 s |
| 4th | 4.7 s | 92 tok/s | 6.3 s |
| **Aggregate** | | **96 tok/s** — the same as one user | |

Every user gets full speed *once they start*; the cost is paid entirely as waiting, and it grows linearly with your position: `wait_k ≈ (k − 1) × service_time`. Aggregate throughput is flat — you are not serving four users faster, you are serving them in turn. Time-slicing would give the same aggregate but spread the pain evenly; FIFO gives the first user a great experience and the last a bad one, which is why **p95 time-to-first-token, not average tok/s, is the number that degrades first** under load. That is the honest reason a laptop is not a server.

**With continuous batching** (`llama-server --parallel N`, vLLM): the opposite curve. One read of the weights advances every active sequence, so aggregate throughput *rises* with load until compute saturates, while per-stream speed falls gently. A useful floor for how far to push it: people read at roughly 5–8 tokens per second, so a stream above ~10 tok/s already feels instant, and batching that trades 99 tok/s for one user into 15 tok/s each for six users is a good trade for chat. This is why a GPU's advantage is a *concurrency* advantage, not a single-request one.

shilp implements neither batching nor a request queue with admission control. `SHILP_MAX_RESIDENT=1` means a request for a *different* model while one is streaming gets a 503, and requests for the same model wait on the engine lock. For a team tool that is correct. For a product it is a ceiling, and §6 says what to do about it.

### 4.4 Concurrency you must support: Little's law

```
concurrent_requests = arrival_rate × latency
```

**Computed**, at 8 s per answer:

| Arrival rate | Concurrency |
|---|---|
| 0.5 req/s | 4 |
| 2 req/s | 16 |
| 10 req/s | 80 |
| 10 req/s at 20 s latency | 200 |

Ten requests per second sounds modest until you notice it means 80 simultaneous generations. If your design serialises them, the 80th user waits 80 × 8 s. This is the arithmetic behind "it worked in staging" — staging had one user.

Little's law tells you the average load; **utilisation** tells you when it falls over. A server that handles one request at a time, taking `S` seconds each, can sustain at most `1/S` requests per second. As arrivals approach that rate, waiting does not grow gently — for random arrivals (the M/M/1 queue) the mean time in the system is `S / (1 − ρ)`, where `ρ = arrival_rate × S`. **Computed** for a 2-second answer, so a capacity of 0.5 req/s:

| Arrival rate | Utilisation ρ | Mean time in system | of which waiting |
|---|---|---|---|
| 0.25 req/s | 50% | 4 s | 2 s |
| 0.40 req/s | 80% | 10 s | 8 s |
| 0.45 req/s | 90% | 20 s | 18 s |
| 0.475 req/s | 95% | 40 s | 38 s |

Real traffic is not exactly random, but the shape is universal: the last 20% of capacity costs more latency than the first 80%. **Plan to run a serial server below ~70% utilisation**, and treat the knee of this curve — not the throughput ceiling — as your capacity. It is also the precise argument for admission control (§6, lever 4): rejecting work above ρ ≈ 0.8 keeps the requests you do accept fast.

### 4.5 Prefill: linear in practice, quadratic in the limit

Prompt length is not free, but the usual "attention is O(n²)" framing overstates the cost at the lengths most apps use. Prefill does two kinds of work per prompt token:

```
prefill_FLOPs ≈ 2 × parameters × n                    (every weight touches every token: linear)
              + ~2 × layers × attention_width × n²    (attention between tokens: quadratic)
```

where `attention_width` is query heads × head size. The quadratic term only overtakes the linear one once `n` exceeds **parameters per layer ÷ attention width** — about **12 × d_model** for a classic decoder block. **Computed**: ~49k tokens for a 7B with `d_model = 4096`; ~26k for Gemma-3-1B, whose wide MLP (6,912) and narrow attention (4 × 256) push the crossover out. Below that, time-to-first-token grows close to **linearly** with the prompt. **Measured**, Gemma-3-1B on an M3 Pro, cold cache:

| Prompt tokens | CPU only | All layers on Metal |
|---|---|---|
| 614 | 1.93 s | 0.44 s |
| 1,712 | 4.98 s | 0.82 s |
| 3,107 | 9.15 s | 1.59 s |
| *prefill rate* | *~340 tok/s* | *~2,000 tok/s above 1k tokens* |

A constant rate across a 5× range of lengths is what linear looks like. The attention-only arithmetic still matters at the far end:

| Prompt tokens | Attention work alone |
|---|---|
| 2,048 | 1× |
| 8,192 | 16× |
| 32,768 | 256× |

so a 32k-token document is genuinely expensive, and it is why RAG systems that stuff 30 pages into a prompt feel slow even on good hardware. The fix is retrieval that returns less, not a bigger GPU. Two practical consequences:

- **Prefill is compute-bound, so this is where a GPU pays.** The 5.8× Metal speed-up above is on the same memory that only gave decode 17% (§4.2). If your users send long prompts, offloading layers is the first thing to check — shilp offloads all layers by default (`SHILP_GPU_LAYERS=-1`).
- **A repeated prefix is almost free** — see the prefix cache in §4.6.

### 4.6 Measured: one shilp process under load

Everything above is arithmetic; this is what one process actually did. M3 Pro, 18 GB, Gemma-3-1B Q4_K_M, `llama-cpp-python` 0.3.36 with Metal, default settings, timed from a separate client.

| Experiment | Result | What it shows |
|---|---|---|
| Short prompt, CPU only vs all layers on GPU | TTFT 178 ms → **27 ms**; decode 84.5 → **98.7 tok/s** | prefill is compute-bound, decode is bandwidth-bound |
| 4 identical requests at once | first tokens at 0, 1.6, 3.2, 4.7 s; each streams at 92–99 tok/s; aggregate 96 tok/s | no batching: waiting, not slowing down (§4.3) |
| 1,712-token prompt, then the same prefix with a new question | **823 ms → 43 ms** | the prefix cache skips re-reading what the model has already seen |
| Two tenants, ~1,960-token system prompts each, alternating A, A, B, A, B, B | **945, 37, 910, 942, 910, 33 ms** | one cache slot: every switch of tenant is a full prefill |
| Client disconnects 0.5 s into a 600-token answer | next request's TTFT **216 ms**; before cancellation existed it was 6.9 s — the rest of the abandoned answer (measured CPU-only, so slightly slower decode) | an abandoned answer must stop, or the queue pays for it |
| 5,009-token prompt into a 4,096 context, `stream: true` | **HTTP 400** with the token counts (was a cut connection) | errors must surface before the first byte |

The prefix-cache rows are the most useful for design, because the effect is large and free:

- `llama-cpp-python` keeps the KV of the **previous request** on that model and re-uses the longest common prefix. Anything identical at the *start* of the prompt is not recomputed: ~19× faster time-to-first-token in the measurement above.
- **Put stable content first** — system prompt, tool definitions, retrieved documents that repeat — and the user's turn last. A timestamp or request id at the top of a system prompt silently disables the cache for every request.
- **There is one cache per model, not per user.** Two tenants with different system prompts evict each other on every switch, as the alternating row shows. That is the argument for *cache-aware routing* once you have replicas: send the same tenant (or the same system prompt) to the same replica, and each replica's cache stays warm. `llama-server`'s parallel slots and vLLM's automatic prefix caching keep many prefixes at once, which is the same idea built into the engine.

To repeat any of these on your own machine, use `scripts/loadtest.py` ([07-operate.md §2](07-operate.md)); `shilp usage` and the SQL in §11 give the server's own view of the same requests.

### 4.7 Worked example: sizing a team assistant

Putting §4.1–4.6 together for one concrete service. Every input is stated so you can substitute your own; figures are **computed** from the measurements above unless marked otherwise.

**The workload (assumed).** An internal assistant for 200 engineers. At the peak hour a quarter of them are active, each sending a message every two minutes: `λ = 50 / 120 s ≈ 0.42 req/s`. A request carries a ~600-token system prompt that repeats (prefix-cached) plus ~150 new tokens, and the answer is ~150 tokens.

**The SLO.** 95% of requests reach their first token within 2 s, and streams run at ≥ 10 tok/s.

**Option A — Gemma-3-1B on one M3 Pro, as shilp runs today.**

| Quantity | Value |
|---|---|
| Service time `S` | 150 new prompt tokens at ~2,000 tok/s + 150 output tokens at ~98 tok/s ≈ **1.6 s** |
| Throughput capacity `1/S` | ≈ 0.62 req/s |
| Utilisation `λ × S` | ≈ **0.67** |
| Output tokens needed | 0.42 × 150 ≈ **62 tok/s** — below the 98 tok/s one stream decodes |
| SLO-safe utilisation, **measured** for a 2 s p95 ([07-operate.md §1](07-operate.md)) | ≈ 0.3 |

The machine has the *throughput* for this workload — it needs 62 output tok/s and can decode 98 — but a serial server cannot use it while meeting the SLO: at 67% utilisation the measured p95 time to first token was 8 s. A serial deployment needs `λ × S ÷ 0.3 ≈ 2.2`, so **3 replicas** on separate machines (replicas on one machine would share its memory bandwidth). The cheaper answer is batching: the same box with `llama-server --parallel 4` would run four streams at once, each far above the 10 tok/s floor, which is exactly the slack batching spends (§4.3). That is the case for the batching engine on the [roadmap](08-service-roadmap.md) — this workload needs concurrency, not a bigger machine.

**Option B — a 7B model (Q4_K_M, 4.68 GB) on the same M3 Pro.**

| Quantity | Value |
|---|---|
| Decode ceiling | 150 GB/s ÷ 4.68 GB ≈ 32 tok/s; **expected** 16–22 |
| Service time | ≈ 150 / 19 ≈ **7.8 s**, before any prefill |
| Utilisation | 0.42 × 7.8 ≈ **3.3** — more than three times what one stream can serve |
| KV for 4 sessions at 4k (Qwen2.5-7B, 56 KiB/token) | ≈ 0.9 GiB — memory is not the problem |

A single stream at 16–22 tok/s cannot deliver 62 output tok/s at all, so no amount of queueing discipline helps: this needs batching on a GPU with several times the bandwidth (an A10G-class card's ceiling for an 8B model is ~120 tok/s per stream, §4.8) or several machines. Whether the 7B's better answers are worth that is a product decision — but it is a 4–8× infrastructure decision, which is why §6 says to right-size the model first.

**The method generalises:** estimate `λ` at the peak, compute `S` from prefill and decode rates, check output tok/s against one stream's decode rate, then size for SLO-safe utilisation — measured with an open-loop load test, not assumed.

### 4.8 Hardware sizing

Single-stream decode ceilings (bandwidth ÷ model size), **computed** from vendor memory-bandwidth specifications and typical Q4_K_M file sizes: 1B ≈ 0.81 GB, 8B ≈ 4.9 GB, 70B ≈ 42.5 GB. "—" means the weights plus a small KV budget do not fit in the device's memory.

| Device | Memory | Bandwidth | 1B | 8B | 70B |
|---|---|---|---|---|---|
| Apple M3 Pro | 18–36 GB | 150 GB/s | 185 | 30 | — |
| Apple M3 Max | 36–128 GB | 300–400 GB/s | 494 | 81 | 9 (64 GB+) |
| Apple M2 Ultra | 64–192 GB | 800 GB/s | 988 | 163 | 19 |
| NVIDIA L4 | 24 GB | 300 GB/s | 370 | 61 | — |
| NVIDIA A10G | 24 GB | 600 GB/s | 741 | 122 | — |
| NVIDIA RTX 4090 | 24 GB | 1,008 GB/s | 1,244 | 205 | — |
| NVIDIA A100 80 GB | 80 GB | ~2,000 GB/s | ~2,500 | ~410 | ~48 |
| NVIDIA H100 SXM | 80 GB | 3,350 GB/s | ~4,100 | ~680 | ~79 |

Tok/s per stream; M3 Max figures use the 400 GB/s configuration. **Expected** real throughput is 50–70% of these (measured: 53% for the 1B on the M3 Pro).

How to read it:

1. **Fit first.** Memory needed ≈ weights + runtime (~1 GB) + KV per session × concurrent sessions. A 70B at Q4_K_M with Llama-3-70B's 320 KiB of KV per token needs ~5 GB more for four 4k-token sessions on top of its 42.5 GB of weights.
2. **Then the per-stream floor.** If the ceiling × 0.5 is below the speed your users need (≥ 10–20 tok/s for chat), the device is too slow for that model whatever else you do.
3. **Then concurrency.** Multiply by batching, not by hope: a serial server's aggregate equals one stream (§4.3); a batching server's aggregate grows with the number of sequences until compute or KV memory runs out — the large-memory, high-bandwidth parts of this table (A100, H100) earn their price there.
4. **Apple Silicon** is the cheapest way to *fit* large models — unified memory up to 192 GB — but it has a fraction of a datacentre GPU's bandwidth and compute, and vLLM does not run on it. It suits one-user and small-team serving; `llama-server`'s parallel slots are its batching option.
---

## 5. The limiting factors, ranked

In the order they actually bite, not the order they are usually listed.

**1. Memory (hard wall).** The model either fits or it does not, and there is no graceful degradation — you get an allocation failure or, worse, swap, which turns a 40 tok/s model into a 2 tok/s one. This is the only limit that is binary.

**2. Memory bandwidth (throughput).** Sets the decode ceiling. You cannot code your way past it; you can only move fewer bytes per token (quantise) or read them faster (better hardware).

**3. Absence of batching (concurrency).** Without batching, throughput is flat as concurrency rises and the extra load turns into queueing (§4.3, §4.4). This, not raw speed, is why a single-process deployment plateaus at a handful of users — on a GPU just as on a CPU.

**4. KV cache growth (concurrency × context).** Silent until it is not: memory grows linearly in *both* concurrent sessions and context length — so with the product of the two — at a per-token rate set by the architecture (§4.1). The failure is an OOM under load rather than a slow request.

**5. Prefill compute (TTFT with long prompts).** Close to linear in prompt length up to tens of thousands of tokens, quadratic beyond (§4.5), and compute-bound — so it is the cost a GPU removes best. A chat app with short prompts never notices; a document-QA app hits it immediately, unless the long part is a repeated prefix the cache can skip.

**6. Cold start (first request).** Loading a 7B takes seconds. With mmap, llama.cpp makes *loading* nearly free, but the first forward pass still faults pages in from disk — so the first request after a restart is slow, and a rolling restart means a burst of slow requests. Warm it deliberately.

**7. Single process (failure domain).** One process, one model, one crash. `uvicorn --workers 4` is not the answer. With mmap the read-only weights are shared through the OS page cache, so four workers do not quite cost four copies of the model — but each has its own KV cache, its own GPU buffers when layers are offloaded, and its own lock, and all four compete for the same memory bandwidth. You get 4× the KV memory for roughly the throughput of one.

**8. Disk I/O.** Only at cold start, and only if the page cache is cold. A network filesystem makes this much worse — keep weights on local SSD.

**9. Python overhead.** At low concurrency it is irrelevant. At high concurrency the tokenizer, sampler and JSON serialisation become measurable, and they hold the GIL. This is one reason vLLM is written the way it is.

**10. Network.** Streaming text is a few KB/s — negligible. If you add image generation or TTS, the egress bill becomes real. Not a limit for text.

---

## 6. Scaling levers, in order of cost-effectiveness

Do these in order. Most deployments stop at step 3 and never need the rest.

**1. Quantise.** Q4_K_M versus fp16 is 3.3× less memory and 3.3× more throughput, for a quality cost most tasks do not notice. Nothing else on this list is close. See [02-reshape.md](02-reshape.md).

**1b. Use the accelerator you have.** Check that layers are actually on the GPU — the log line `offloaded N/N layers` or a time-to-first-token in tens of milliseconds for a short prompt. It is free, and on Apple Silicon it cut short-prompt TTFT 6.7× and long-prompt prefill 5.8× (§4.6). shilp offloads every layer by default; `SHILP_GPU_LAYERS=0` turns it off.

**2. Cap context.** `num_ctx` is allocated up front and multiplied by concurrent sessions. If your app needs 4k, do not set 32k "in case". This is free and frequently saves more memory than quantisation on a busy box.

**3. Right-size the model per task.** Route: a 1.5B for classification, extraction, routing and summarisation; a 7B for the hard 10%. Two models on one box costs less than one big model and is often better overall. It is also the only lever that improves *both* cost and latency.

**4. Add a queue with admission control.** A bounded queue plus a 429 when full converts "everything gets slow" into "some requests are rejected quickly". Clients retry; users get an answer. Unbounded queueing is how a service dies politely. Size the bound from §4.4: with `S` seconds per answer and a latency budget `B`, admit at most about `B / S` waiting requests, and reject the rest with a `Retry-After`.

**5. Batch: `llama-server` slots first, then GPU + vLLM.** Continuous batching turns flat throughput into rising throughput. This is the single biggest architectural change available, and it is worth it only when you have concurrent demand. On the same box and the same GGUF files, `llama-server --parallel N` is the cheap first step (and the only one on Apple Silicon); on NVIDIA hardware at higher concurrency, vLLM's paged attention fits more sequences into the same memory.

**6. Replicate.** Stateless replicas behind a load balancer scale linearly, at the cost of one full model copy each. Combine with 5 — replicas of a batched server, not of a serial one. Route with the cache in mind: sending the same tenant or system prompt to the same replica keeps each replica's prefix cache warm, which round-robin destroys (§4.6).

**7. Cache.** Two kinds, both underrated. A **prefix cache** reuses the KV of a prompt's unchanged beginning: shilp gets one per model for free from `llama-cpp-python`, measured at ~19× faster time-to-first-token on a repeated 1.7k-token prefix (§4.6) — provided the stable part of the prompt comes first. A **response cache** answers repeated identical queries without the model at all; for a support bot it can absorb a large fraction of traffic. Key it on the model, the full message list and the sampling parameters, and only cache `temperature: 0` answers.

**8. Speculative decoding.** A small draft model proposes tokens, the big model verifies. 1.5–2.5× on some workloads, no quality change. Complexity for a modest, uneven win — do it last.

**9. Distil or fine-tune a smaller model.** The real answer at scale: if 80% of your traffic is one narrow task, a fine-tuned 1.5B beats a general 7B on that task, at a tenth of the cost.

---

## 7. What breaks as you grow

| Stage | Users | Bottleneck | Symptom | Fix |
|---|---|---|---|---|
| Laptop demo | 1 | none | works fine | — |
| Small team | 2–5 | serialisation | the second person waits for the first answer to finish | accept it, keep answers short, warm the prefix |
| Internal tool | 10–30 | KV cache memory + serialisation | OOM under load; p95 in the tens of seconds | queue + admission control, smaller model |
| Product, low volume | 100s/day | cold starts, failures | occasional 40 s request; a crash takes everything | replicas, health checks, warmup |
| Product, real volume | 10s/min | throughput ceiling | queue grows monotonically; timeouts | GPU + batching, or hosted API |
| Scale | 1000s/min | cost and ops | GPU bill exceeds the hosted alternative | mixed: self-host the hot model, hosted for burst |

The transition that catches people is internal tool → product. An internal tool tolerates 20-second answers and one crash a week. A product does not, and the fixes are operational (replicas, timeouts, shedding load), not model-side.

---

## 8. Reliability: what actually fails

**Mid-stream failure.** The model fails or the process dies halfway through a stream. The status line already said `200`, so the client has a partial answer and only the end of the stream can tell it so. shilp ends a failed stream with an error frame (§1) and never sends `done`/`[DONE]` after one; a client that treats a stream without its terminator as an error is then correct in every case, including a killed process.

**Context overflow.** A prompt longer than `num_ctx` cannot be answered at all. shilp counts the rendered prompt — chat template included — before generating and returns a 400 naming both numbers, even for `stream: true`. **Measured** before that check existed: the client saw a cut connection and the logs a 500. Validate on the client too if you can: it is cheaper to trim the oldest turns of a conversation than to round-trip a rejection.

**Cancellation.** If a user closes the tab, the answer should stop. shilp's worker checks a flag between tokens and closes the llama.cpp generator when the client goes away, which releases the model for the next request; the abandoned request is recorded with status **499**. **Measured**: before this, a disconnect 0.5 s into a 600-token answer made the next user wait 6.9 s for tokens nobody would read; after, 216 ms. Three limits remain. The flag is checked between tokens, so a request cancelled during a long *prefill* still finishes its prefill before stopping; a request whose client leaves while it is still *waiting* for the model stays in line and then pays for its prefill too (roadmap P1.4 replaces the lock with a cancellable queue). And a proxy must pass the disconnect through — one that keeps the upstream connection open after the client leaves defeats cancellation entirely.

**Timeouts, in three places.** Client timeout, proxy timeout, and server generation limit. If the proxy times out before the server finishes, the user sees a 504 while the GPU keeps working. Set them deliberately, in that order, longest last.

**Graceful shutdown.** `SIGTERM` should stop accepting, finish in-flight streams, then exit. Otherwise every deploy truncates live answers.

**Health checks that lie.** `/health` returning 200 while the model is not loaded tells the load balancer to send traffic to a replica that will fail. Report model residency in the health payload — shilp's `/health` includes `resident` for exactly this reason. Note that shilp loads lazily: `resident` is empty until the first request, so a readiness check that requires it must come *after* the warmup request in §12, or a fresh replica never becomes ready.

**Retry storms.** A client retrying a 20-second request every second multiplies load by 20. Retryable codes should be 429/503 only, with backoff, and the server should shed rather than queue.

---

## 9. Security: the part shilp does not do

Stated plainly because it is a real gap, not a TODO: **shilp has no authentication, no per-key quotas, and no rate limiting.** It binds to `127.0.0.1` by default for that reason. Exposing it directly to the internet means anyone with the URL can use your GPU.

Before it faces traffic you need, at minimum:

- **Authentication** — an API key checked by a dependency, or a reverse proxy doing it. A `SHILP_API_KEY` bearer check is small and is on the roadmap.
- **Rate limiting and quotas** — per key, per minute. Without it, one looping client consumes the box.
- **Admin endpoints closed** — `/api/pull` (downloads any model) and `/api/delete` (removes one) are as open as inference today. Forward only the inference paths through the proxy ([03-serve.md](03-serve.md) lists them) until admin separation lands ([08-service-roadmap.md](08-service-roadmap.md), P1.2).
- **TLS** — terminate at a proxy (Caddy is two lines), never serve plain HTTP to the internet.
- **Prompt-injection surface** — your server will happily execute whatever the prompt says; the risk lives in your *application*, not the inference layer, but it is worth naming since "it is just a model server" invites complacency.
- **Multi-tenancy** — with one model and no session isolation, per-tenant data separation has to be enforced above this layer. The telemetry table stores model and token counts, not prompt text — check that stays true if you extend it. The prefix cache is shared too: it never returns another tenant's text, but time-to-first-token reveals whether someone *recently sent the same prefix* (§4.6 shows a ~19× difference). That timing side channel matters only if prompts are secret from other tenants; if they are, isolate tenants per replica.
- **Secrets** — the HF token belongs in the environment, never in a manifest or a recipe.

---

## 10. Cost: self-host versus hosted API

The arithmetic, so you can decide rather than guess:

```
cost_per_million_tokens = gpu_dollars_per_hour / (aggregate_tokens_per_second × 3600 / 1e6)
```

**Computed**, assuming those hourly prices:

| GPU $/hr | Aggregate tok/s | $ per 1M tokens |
|---|---|---|
| $1.00 | 500 | $0.56 |
| $1.00 | 2,000 | $0.14 |
| $2.50 | 500 | $1.39 |
| $2.50 | 2,000 | $0.35 |
| $2.50 | 5,000 | $0.14 |

Hosted APIs for small models sit in the same range, so **self-hosting is not automatically cheaper** — and it never is at low volume, because a GPU bills whether or not anyone is asking. A single A10-class instance at ~$1/hr is ~$730/month whether you serve 10 requests or 10 million.

Where self-hosting genuinely wins:

- **Privacy or regulation** — the data cannot leave your infrastructure. This is the strongest reason and often the only one that matters.
- **A fine-tune nobody hosts** — your LoRA on a small base model.
- **Latency** — no network hop to a provider, and no shared rate limits.
- **Predictable, high, steady volume** — above roughly the point where hosted spend exceeds GPU spend plus your time.
- **Customisation** — grammar-constrained output, custom sampling, a specific quantisation, your own telemetry.

Where hosted wins: bursty traffic, no ops appetite, frontier model quality, and anything where your engineering hours cost more than the GPU. Below a few hundred requests a day, self-hosting is a hobby, and that is a legitimate thing to want — just do not cost it as if it were free.

The realistic pattern at scale is **mixed**: self-host the high-volume narrow task on a small quantised model, fall back to a hosted frontier model for the hard cases.

---

## 11. Observability a provider needs

Four numbers decide whether your service is healthy:

| Metric | Why | Where in shilp |
|---|---|---|
| **TTFT** p50/p95 | the number users feel; on a serial server it is queue wait + prefill | per request in SQLite — percentiles via the query below |
| **tok/s** per model | decode health; a drop means swapping, thermal throttling or a lost GPU offload | `shilp usage`, SQLite |
| **Saturation** | the leading indicator of everything else | average requests in flight, from SQLite (below) |
| **Rate by status code** | 400 vs 499 vs 503 vs 500 tells you *which* failure | `shilp_requests_total{status=…}` in `/metrics`, and the `status` column |

Plus two that are specific to this domain: **tokens per request** (a spike means someone is pasting documents) and **context utilisation** (how close to `num_ctx` your sessions run, which predicts OOM and 400s).

Two details of what shilp records, because they decide which questions you can answer:

- **`/metrics` gives averages, not percentiles.** The duration metrics are Prometheus *summaries* with only `_sum` and `_count`, so a dashboard can plot the mean TTFT but not p95. Percentiles come from the per-request rows, which is what they are for.
- **Status codes are what the client saw.** A rejected over-long prompt is a 400, a client that left mid-answer is a 499, a busy model is a 503 — not a blanket 500. Requests rejected before generation (unknown model 404, validation 422) do not reach the telemetry table yet. The 499 rate is worth an alert of its own: users abandoning answers is how a slow server shows up before anyone complains.

**Percentiles of time-to-first-token**, last hour, from `$SHILP_HOME/logs/shilp.db`:

```sql
WITH r AS (
  SELECT ttft_ms,
         ROW_NUMBER() OVER (ORDER BY ttft_ms) AS rn,
         COUNT(*) OVER () AS n
  FROM requests
  WHERE status = 200 AND ttft_ms > 0 AND ts > strftime('%s', 'now') - 3600
)
SELECT n,
       MAX(CASE WHEN rn <= 0.50 * n THEN ttft_ms END) AS p50_ms,
       MAX(CASE WHEN rn <= 0.95 * n THEN ttft_ms END) AS p95_ms
FROM r;
```

**Saturation and failures**, same window. `avg_in_flight` is Little's law read backwards — total time spent in the system divided by the window is the average number of requests in flight. On a server that answers one at a time, a value approaching 1 means requests have started to queue (§4.4):

```sql
SELECT COUNT(*)                                   AS requests,
       SUM(status = 499)                          AS abandoned,
       SUM(status BETWEEN 400 AND 498)            AS rejected,
       SUM(status >= 500)                         AS failed,
       ROUND(SUM(total_ms) / (3600 * 1000.0), 2)  AS avg_in_flight
FROM requests
WHERE ts > strftime('%s', 'now') - 3600;
```

What shilp does not have is a queue of its own — waiting happens on the engine lock — so there is no queue-depth gauge to scrape. `avg_in_flight` and p95 TTFT are the honest substitutes, and the absence is itself the signal that you are running a serial server.

---

## 12. A reference deployment

Minimal, boring, and correct. Adjust the model and paths.

**The service** (`/etc/systemd/system/shilp.service`):

```ini
[Unit]
Description=shilp model server
After=network.target

[Service]
Type=simple
User=shilp
Environment=SHILP_HOME=/var/lib/shilp
Environment=SHILP_MAX_RESIDENT=1
# -1 = every layer on the GPU; 0 = CPU only (ignored by a CPU-only build)
Environment=SHILP_GPU_LAYERS=-1
ExecStart=/opt/shilp/.venv/bin/shilp serve \
    --model google_gemma-3-1b-it-gguf:q4_k_m --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
# finish in-flight streams on deploy
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

Bind to loopback and let a proxy handle the outside world — it keeps auth and TLS in one place instead of in your model server.

**The proxy** (`Caddyfile`), which gives you TLS and a key check for free. `{$SHILP_API_KEY}` is substituted from Caddy's environment when the config loads, so the key never sits in the file:

```
llm.example.com {
    @authorized header Authorization "Bearer {$SHILP_API_KEY}"
    handle @authorized {
        reverse_proxy 127.0.0.1:8000 {
            flush_interval -1     # required: do not buffer the stream
        }
    }
    respond 401
}
```

`flush_interval -1` is not optional. A proxy that buffers turns streaming into a 20-second wait followed by a wall of text, and the symptom looks like a model problem. Caddy flushes `text/event-stream` on its own, but the Ollama surface streams `application/x-ndjson`, which it would buffer. One shared key is the minimum, not multi-tenant auth: per-key quotas and usage need a gateway or the roadmap's `SHILP_API_KEY` work.

**Warm it after start**, so the first real user does not pay the page-in cost:

```bash
curl -s localhost:8000/api/generate -H 'content-type: application/json' \
  -d '{"model":"","prompt":"warm","stream":false,"options":{"num_predict":1}}'
```

The empty `model` falls back to the one named by `--model`. This is also what makes `/health` report the model as resident (§8). If your clients share a long system prompt, warm with *that* prompt: the prefix cache then starts hot as well (§4.6).

**Then verify from outside**, which is the only test that counts:

```bash
curl -N https://llm.example.com/v1/chat/completions \
  -H "Authorization: Bearer $SHILP_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"google_gemma-3-1b-it-gguf:q4_k_m",
       "messages":[{"role":"user","content":"hi"}],"stream":true}'
```

If chunks arrive progressively, the whole path works. If they arrive at once, the proxy is buffering.

---

## 13. Go-live checklist

- [ ] Model chosen by task, quantised, and `num_ctx` set to what you actually need
- [ ] KV-per-token of the chosen model checked (§4.1) — it decides concurrency per GiB
- [ ] Memory budget includes KV × expected concurrent sessions, not just weights
- [ ] Layers confirmed on the GPU (`offloaded N/N layers`, short-prompt TTFT in tens of ms)
- [ ] Little's law run: `arrival_rate × latency` fits your capacity, at ≤ ~70% utilisation
- [ ] Bound on concurrency, with a 429 and `Retry-After` when exceeded
- [ ] Prompts laid out stable-first, so the prefix cache hits; no timestamps at the top
- [ ] Clients treat a stream without `done`/`[DONE]` as a failure
- [ ] Timeouts set in all three places, longest at the server
- [ ] Streaming verified *through* the proxy, not just locally
- [ ] Auth and rate limits in front
- [ ] Only inference paths forwarded by the proxy; `/api/pull` and `/api/delete` unreachable from outside
- [ ] `/health` reports model residency; the LB uses it
- [ ] Graceful shutdown drains in-flight streams
- [ ] Warmup request after every start and deploy
- [ ] p95 TTFT computed from the request rows (not the `/metrics` average), with an alert
- [ ] Alerts on the 499 (abandoned) and 5xx rates, and on `avg_in_flight` approaching 1
- [ ] Cost per 1M tokens computed, and compared against the hosted alternative
- [ ] A documented fallback for when the box dies

---

## 14. Glossary

**Admission control** — rejecting or queueing work at the door so the system stays inside its capacity. **Aggregate throughput** — total tokens/second across all users; distinct from per-stream speed. **Batching** — processing several sequences per weight read; the throughput lever. **Cache-aware routing** — sending requests that share a prefix to the same replica so its prefix cache stays warm. **Cold start** — the first request after a load, which pays page-in and cache allocation. **Continuous batching** — admitting new requests at token boundaries mid-flight; what `llama-server` slots and vLLM do. **Concurrency** — requests in flight simultaneously. **Graceful shutdown** — stop accepting, finish in-flight, then exit. **Grouped-query attention (GQA)** — several query heads sharing one key/value head; the reason modern models' KV cache is 4–32× smaller than older ones'. **KV cache** — per-session memory for attention state; the term that scales with users. **Little's law** — `concurrency = arrival_rate × latency`. **mmap** — mapping weights into virtual memory so only touched pages load, and processes share them. **p50/p95/p99** — latency percentiles; the tail is what users experience. **Prefill** — the parallel pass over the prompt; compute-bound, close to linear in prompt length until attention dominates. **Prefix cache** — re-using the KV of a prompt's unchanged beginning so it is not recomputed. **Quantisation** — reducing weight precision; the largest single lever on memory and speed. **Rate limiting** — capping requests per client per unit time. **Replica** — an independent process serving the same model; scales linearly, at one model copy each. **Saturation** — the point where added load no longer adds throughput. **Speculative decoding** — a draft model proposes, the target verifies. **Status 499** — the client closed the connection before the answer finished (nginx's convention); shilp records abandoned answers with it. **Time to first token (TTFT)** — queueing plus prefill; perceived responsiveness. **Utilisation (ρ)** — arrival rate × service time; waiting grows as `1 / (1 − ρ)` near 1. **Warmup** — a deliberate throwaway request so the first user does not pay cold-start cost.

---

## Where this leaves shilp

Being precise about the boundary, since this document is about production and shilp is not yet a production server:

**It has**: both API dialects, streaming with status codes for early failures and error frames for late ones, cancellation of abandoned answers, GPU offload by default, a context-overflow check, per-request telemetry with true status codes, Prometheus metrics, LRU residency with refcounts, a working engine swap, and honest documentation of every trade-off.

**It lacks**: authentication, rate limiting, a request queue with admission control, batching, and multi-tenancy. The first two are small and on the roadmap; admission control is a bounded semaphore around the engine; batching is a different engine — `llama-server` slots on the same GGUFs, or vLLM on CUDA.

**What that means**: shilp is a good *team* server and a good *learning* stack — the thing you run to understand what a provider does before you rent the GPU. If you need to serve the public, put it behind a proxy that authenticates, and plan the move to a batched engine before the queue starts growing.

That is not a weakness of the design. It is the same ladder every provider climbed: one model, one box, one user — then queueing, then batching, then replicas. Knowing which rung you are on is most of the engineering.

The rungs, with what each one needs and what shilp already has, are laid out in the [service roadmap](08-service-roadmap.md); the components behind them are explained in [06-service-fundamentals.md](06-service-fundamentals.md), and running the result day to day in [07-operate.md](07-operate.md).
