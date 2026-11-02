# 05 — Serving your own model like a provider

**From "a model on Hugging Face" to "an endpoint my app can call like any hosted API" — and what limits it once real traffic arrives.**

The other documents cover what inference *is* ([00-theory.md](00-theory.md)), how to get a model ([01-pull.md](01-pull.md)), how to change it ([02-reshape.md](02-reshape.md)) and how to run it locally ([03-serve.md](03-serve.md)). This one is about the step after that: putting it on a machine that other people's code calls, and being honest about what breaks.

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

The honest framing: matching the API is a weekend. Matching the *operational* contract — auth, quotas, rate limits, retries, graceful degradation — is the actual work of being a provider.

---

## 2. The shortest path

Four commands, from nothing to an endpoint a client can call:

```bash
pip install -e ".[engine]"
shilp pull hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
shilp serve --model qwen2.5-1.5b-instruct-gguf:q4_k_m --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://your-host:8000/v1", api_key="unused")
print(client.chat.completions.create(
    model="qwen2.5-1.5b-instruct-gguf:q4_k_m",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

That is a working private endpoint. Everything below is about what happens when the second user arrives.

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
| **N replicas behind a proxy** | linear throughput, rolling restarts | each replica loads its own copy of the weights |
| **GPU cluster + vLLM** | continuous batching, 10×+ throughput | real ops work; no longer a laptop project |

The jump that matters is the last one. Everything up to "N replicas" is a stateless web service with an unusually large binary asset; the cluster tier is a different discipline.

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

### 4.3 Concurrency: what happens when a second user arrives

This is where CPU and GPU diverge completely, and where most intuition is wrong.

**On CPU (llama.cpp, one slot):** requests **serialise**. There is no batching, so N concurrent streams each get roughly 1/N of the single-stream rate.

| Concurrent | M2 Pro, 7B Q4_K_M |
|---|---|
| 1 | ~43 tok/s |
| 4 | ~11 tok/s each |
| 16 | ~2.7 tok/s each |

**Aggregate throughput is flat.** You are not serving 16 users faster; you are serving them slower, together. Past a small number, each user's answer takes so long that they leave. That is the honest reason a laptop is not a server.

**On GPU with continuous batching (vLLM):** the opposite. Reading the weights once serves many sequences, so aggregate throughput rises with load and per-stream speed degrades mildly. This is why a GPU's advantage is a *concurrency* advantage, not a single-request one.

shilp implements neither batching nor a queue: `SHILP_MAX_RESIDENT=1` means a second request for a *different* model gets a 503, and requests for the same model serialise on the engine lock. For a team tool that is correct. For a product it is a ceiling, and §6 says what to do about it.

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

### 4.5 Prefill: the cost that is quadratic

Attention is O(n²) in sequence length, so prompt length is not free:

| Prompt tokens | Attention work |
|---|---|
| 2,048 | 1× |
| 4,096 | 4× |
| 8,192 | 16× |
| 32,768 | 256× |

A single 32k-token document costs ~256× the attention of a 2k prompt *before* generating anything. Time-to-first-token for long prompts is a compute problem, not a bandwidth one, and it is why RAG systems that stuff 30 pages into a prompt feel slow even on good hardware. The fix is retrieval that returns less, not a bigger GPU.
---

## 5. The limiting factors, ranked

In the order they actually bite, not the order they are usually listed.

**1. Memory (hard wall).** The model either fits or it does not, and there is no graceful degradation — you get an allocation failure or, worse, swap, which turns a 40 tok/s model into a 2 tok/s one. This is the only limit that is binary.

**2. Memory bandwidth (throughput).** Sets the decode ceiling. You cannot code your way past it; you can only move fewer bytes per token (quantise) or read them faster (better hardware).

**3. Absence of batching (concurrency).** On CPU, throughput is flat as concurrency rises. This, not raw speed, is why a single-machine deployment plateaus at a handful of users.

**4. KV cache growth (concurrency × context).** Silent until it is not: memory climbs with the square-ish of your session count × context, and the failure is an OOM under load rather than a slow request.

**5. Prefill compute (TTFT with long prompts).** Quadratic in prompt length. A chat app with short prompts never notices; a document-QA app hits it immediately.

**6. Cold start (first request).** Loading a 7B takes seconds. With mmap, llama.cpp makes *loading* nearly free, but the first forward pass still faults pages in from disk — so the first request after a restart is slow, and a rolling restart means a burst of slow requests. Warm it deliberately.

**7. Single process (failure domain).** One process, one model, one crash. `uvicorn --workers 4` is not the answer: each worker loads its own copy of the weights and its own KV cache, so four workers means 4× the memory for the same throughput on CPU.

**8. Disk I/O.** Only at cold start, and only if the page cache is cold. A network filesystem makes this much worse — keep weights on local SSD.

**9. Python overhead.** At low concurrency it is irrelevant. At high concurrency the tokenizer, sampler and JSON serialisation become measurable, and they hold the GIL. This is one reason vLLM is written the way it is.

**10. Network.** Streaming text is a few KB/s — negligible. If you add image generation or TTS, the egress bill becomes real. Not a limit for text.

---

## 6. Scaling levers, in order of cost-effectiveness

Do these in order. Most deployments stop at step 3 and never need the rest.

**1. Quantise.** Q4_K_M versus fp16 is 3.3× less memory and 3.3× more throughput, for a quality cost most tasks do not notice. Nothing else on this list is close. See [02-reshape.md](02-reshape.md).

**2. Cap context.** `num_ctx` is allocated up front and multiplied by concurrent sessions. If your app needs 4k, do not set 32k "in case". This is free and frequently saves more memory than quantisation on a busy box.

**3. Right-size the model per task.** Route: a 1.5B for classification, extraction, routing and summarisation; a 7B for the hard 10%. Two models on one box costs less than one big model and is often better overall. It is also the only lever that improves *both* cost and latency.

**4. Add a queue with admission control.** A bounded queue plus a 429 when full converts "everything gets slow" into "some requests are rejected quickly". Clients retry; users get an answer. Unbounded queueing is how a service dies politely.

**5. Move to GPU + vLLM for concurrency.** Continuous batching turns flat throughput into rising throughput. This is the single biggest architectural change available, and it is worth it only when you are CPU-bound *and* have concurrent demand.

**6. Replicate.** Stateless replicas behind a load balancer scale linearly, at the cost of one full model copy each. Combine with 5 — replicas of a batched GPU server, not of a serial CPU one.

**7. Cache.** Two kinds, both underrated: a **prompt/prefix cache** (reuse the KV for a long system prompt across requests — llama.cpp keeps a prompt cache per slot) and a **response cache** for repeated identical queries. For a support bot, a response cache can absorb a large fraction of traffic.

**8. Speculative decoding.** A small draft model proposes tokens, the big model verifies. 1.5–2.5× on some workloads, no quality change. Complexity for a modest, uneven win — do it last.

**9. Distil or fine-tune a smaller model.** The real answer at scale: if 80% of your traffic is one narrow task, a fine-tuned 1.5B beats a general 7B on that task, at a tenth of the cost.

---

## 7. What breaks as you grow

| Stage | Users | Bottleneck | Symptom | Fix |
|---|---|---|---|---|
| Laptop demo | 1 | none | works fine | — |
| Small team | 2–5 | serialisation | answers get slow when two people ask at once | accept it, or cap context |
| Internal tool | 10–30 | KV cache memory + serialisation | OOM under load; p95 in the tens of seconds | queue + admission control, smaller model |
| Product, low volume | 100s/day | cold starts, failures | occasional 40 s request; a crash takes everything | replicas, health checks, warmup |
| Product, real volume | 10s/min | throughput ceiling | queue grows monotonically; timeouts | GPU + batching, or hosted API |
| Scale | 1000s/min | cost and ops | GPU bill exceeds the hosted alternative | mixed: self-host the hot model, hosted for burst |

The transition that catches people is internal tool → product. An internal tool tolerates 20-second answers and one crash a week. A product does not, and the fixes are operational (replicas, timeouts, shedding load), not model-side.

---

## 8. Reliability: what actually fails

**Mid-stream failure.** The model crashes or the client disconnects halfway through an SSE stream. The client has a partial answer and no way to know it is partial. Mitigation: a `finish_reason` on the final frame, and a client that treats a stream without one as an error.

**No cancellation.** If a user closes the tab, the generation continues to completion — llama.cpp exposes no cancel through the Python bindings. On a busy server this wastes a measurable fraction of your capacity. shilp has this gap; the roadmap's cooperative-flag fix is the honest mitigation.

**Timeouts, in three places.** Client timeout, proxy timeout, and server generation limit. If the proxy times out before the server finishes, the user sees a 504 while the GPU keeps working. Set them deliberately, in that order, longest last.

**Graceful shutdown.** `SIGTERM` should stop accepting, finish in-flight streams, then exit. Otherwise every deploy truncates live answers.

**Health checks that lie.** `/health` returning 200 while the model is not loaded tells the load balancer to send traffic to a replica that will fail. Report model residency in the health payload — shilp's `/health` includes `resident` for exactly this reason.

**Retry storms.** A client retrying a 20-second request every second multiplies load by 20. Retryable codes should be 429/503 only, with backoff, and the server should shed rather than queue.

---

## 9. Security: the part shilp does not do

Stated plainly because it is a real gap, not a TODO: **shilp has no authentication, no per-key quotas, and no rate limiting.** It binds to `127.0.0.1` by default for that reason. Exposing it directly to the internet means anyone with the URL can use your GPU.

Before it faces traffic you need, at minimum:

- **Authentication** — an API key checked by a dependency, or a reverse proxy doing it. A `SHILP_API_KEY` bearer check is small and is on the roadmap.
- **Rate limiting and quotas** — per key, per minute. Without it, one looping client consumes the box.
- **TLS** — terminate at a proxy (Caddy is two lines), never serve plain HTTP to the internet.
- **Prompt-injection surface** — your server will happily execute whatever the prompt says; the risk lives in your *application*, not the inference layer, but it is worth naming since "it is just a model server" invites complacency.
- **Multi-tenancy** — with one model and no session isolation, per-tenant data separation has to be enforced above this layer. The telemetry table stores model and token counts, not prompt text — check that stays true if you extend it.
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

Four numbers decide whether your service is healthy, and all four are in shilp already:

| Metric | Why | Where |
|---|---|---|
| **TTFT** p50/p95 | the number users feel | `shilp usage`, `/metrics` |
| **tok/s** per model | decode health; drops mean swapping or contention | same |
| **Queue depth / saturation** | the leading indicator of everything else | not implemented — add it |
| **Error rate by code** | 429 vs 503 vs 500 tells you *which* failure | `/metrics` counters |

Plus two that are specific to this domain: **tokens per request** (a spike means someone is pasting documents) and **context utilisation** (how close to `num_ctx` your sessions run, which predicts OOM).

shilp records one SQLite row per request with TTFT, tok/s and token counts, and exposes Prometheus text at `/metrics`. What it does not have is a queue, so there is no queue depth to report — which is itself the honest signal that you are running a serial server.

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
ExecStart=/opt/shilp/.venv/bin/shilp serve \
    --model qwen2.5-1.5b-instruct-gguf:q4_k_m --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
# finish in-flight streams on deploy
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

Bind to loopback and let a proxy handle the outside world — it keeps auth and TLS in one place instead of in your model server.

**The proxy** (`Caddyfile`), which gives you TLS and a key check for free:

```
llm.example.com {
    @authorized header Authorization "Bearer {env.SHILP_PROXY_KEY}"
    handle @authorized {
        reverse_proxy 127.0.0.1:8000 {
            flush_interval -1     # required: do not buffer the SSE stream
        }
    }
    respond 401
}
```

`flush_interval -1` is not optional. A proxy that buffers turns streaming into a 20-second wait followed by a wall of text, and the symptom looks like a model problem.

**Warm it after start**, so the first real user does not pay the page-in cost:

```bash
curl -s localhost:8000/api/generate -d '{"model":"","prompt":"warm","stream":false,"options":{"num_predict":1}}'
```

**Then verify from outside**, which is the only test that counts:

```bash
curl -N https://llm.example.com/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"qwen2.5-1.5b-instruct-gguf:q4_k_m",
       "messages":[{"role":"user","content":"hi"}],"stream":true}'
```

If chunks arrive progressively, the whole path works. If they arrive at once, the proxy is buffering.

---

## 13. Go-live checklist

- [ ] Model chosen by task, quantised, and `num_ctx` set to what you actually need
- [ ] Memory budget includes KV × expected concurrent sessions, not just weights
- [ ] Little's law run: `arrival_rate × latency` fits your capacity
- [ ] Bound on concurrency, with a 429 when exceeded
- [ ] Timeouts set in all three places, longest at the server
- [ ] Streaming verified *through* the proxy, not just locally
- [ ] Auth and rate limits in front
- [ ] `/health` reports model residency; the LB uses it
- [ ] Graceful shutdown drains in-flight streams
- [ ] Warmup request after every start and deploy
- [ ] TTFT and tok/s scraped, with an alert on p95
- [ ] Cost per 1M tokens computed, and compared against the hosted alternative
- [ ] A documented fallback for when the box dies

---

## 14. Glossary

**Admission control** — rejecting or queueing work at the door so the system stays inside its capacity. **Aggregate throughput** — total tokens/second across all users; distinct from per-stream speed. **Batching** — processing several sequences per weight read; the GPU throughput lever. **Cold start** — the first request after a load, which pays page-in and cache allocation. **Continuous batching** — admitting new requests at token boundaries mid-flight; vLLM's core advantage. **Concurrency** — requests in flight simultaneously. **Graceful shutdown** — stop accepting, finish in-flight, then exit. **KV cache** — per-session memory for attention state; the term that scales with users. **Little's law** — `concurrency = arrival_rate × latency`. **mmap** — mapping weights into virtual memory so only touched pages load. **p50/p95/p99** — latency percentiles; the tail is what users experience. **Prefill** — the parallel pass over the prompt; sets TTFT and scales quadratically. **Quantisation** — reducing weight precision; the largest single lever on memory and speed. **Rate limiting** — capping requests per client per unit time. **Replica** — an independent process serving the same model; scales linearly, at one model copy each. **Saturation** — the point where added load no longer adds throughput. **Speculative decoding** — a draft model proposes, the target verifies. **Time to first token (TTFT)** — prefill plus queueing; perceived responsiveness. **Warmup** — a deliberate throwaway request so the first user does not pay cold-start cost.

---

## Where this leaves shilp

Being precise about the boundary, since this document is about production and shilp is not yet a production server:

**It has**: both API dialects, streaming, telemetry, metrics, LRU residency with refcounts, error translation, a working engine swap, and honest documentation of every trade-off.

**It lacks**: authentication, rate limiting, a request queue, batching, cancellation, and multi-tenancy. The first two are small and on the roadmap; batching is a different engine (vLLM); cancellation is a cooperative flag.

**What that means**: shilp is a good *team* server and a good *learning* stack — the thing you run to understand what a provider does before you rent the GPU. If you need to serve the public, put it behind a proxy that authenticates, and plan the move to a batched engine before the queue starts growing.

That is not a weakness of the design. It is the same ladder every provider climbed: one model, one box, one user — then queueing, then batching, then replicas. Knowing which rung you are on is most of the engineering.
