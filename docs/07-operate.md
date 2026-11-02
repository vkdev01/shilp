# 07 — Operating it as a service

**Service-level objectives, load testing, model rollout, observability, runbooks, packaging and quality — what it takes to run a model endpoint that other people depend on.**

[06-service-fundamentals.md](06-service-fundamentals.md) explains the components; [05-deploy.md](05-deploy.md) sizes them. This document is about the day-to-day: deciding what "working" means, proving it under load, changing the model without breaking clients, and knowing what to do when the numbers move. Where shilp does not have something yet, the section says so and points at the [service roadmap](08-service-roadmap.md).

Figures marked **measured** come from one shilp process on an M3 Pro (18 GB) serving Gemma-3-1B Q4_K_M with Metal, driven by `scripts/loadtest.py`. Sample sizes are small (20–70 requests per point), so treat tail percentiles as indicative, not precise.

---

## 1. Service-level objectives

### Indicators, objectives, budgets

- An **SLI** (indicator) is a measurement: "fraction of requests whose time to first token was under 2 s".
- An **SLO** (objective) is a target for it over a window: "≥ 95% of requests in any 30 days".
- The **error budget** is what the SLO allows to fail: 5% of requests in this example. While budget remains, ship changes; when it is spent, prioritise reliability.

Write SLOs as *fractions of good requests* rather than as percentiles. "p95 TTFT < 2 s" and "95% of requests have TTFT < 2 s" mean the same thing, but only the second has a budget you can count and burn.

### The indicators that matter for an LLM endpoint

| SLI | Good request | Why |
|---|---|---|
| **Availability** | status is not 5xx, and not a 429 caused by a full queue | 4xx for a bad request or an exhausted quota are the client's doing; a 429 for *lack of capacity* is the service failing and must count against it, or a saturated service could reject everything and stay "available". 499 is the client leaving — track it separately |
| **Time to first token** | first content frame within *X* s | what users perceive as "responsive"; includes queueing |
| **Stream speed** (time per output token) | ≥ *Y* tok/s after the first token | below human reading speed (~5–8 tok/s) the stream feels broken |
| **Completeness** | stream ended with `done` / `[DONE]`, not an error frame | catches mid-stream failures that a status code cannot show |
| **Truncation** | finish reason is not `length` | answers cut off by `max_tokens` — see the gap below |

### Example SLO sets

| | Team tool | Product |
|---|---|---|
| Availability | 99% | 99.9% |
| TTFT | 95% under 3 s | 99% under 1 s |
| Stream speed | 95% above 10 tok/s | 99% above 20 tok/s |
| Window | 30 days | 30 days |
| Error budget (availability) | 1% ≈ 7.2 hours of full outage | 0.1% ≈ 43 minutes |

### Alerting on burn rate

Alert on how fast the budget is being spent, not on single bad requests. The multi-window pattern from Google's SRE workbook, for a 30-day window:

| Budget spent | In | Burn rate | Action |
|---|---|---|---|
| 2% | 1 hour (confirmed over 5 min) | 14.4× | page |
| 5% | 6 hours (confirmed over 30 min) | 6× | page |
| 10% | 3 days (confirmed over 6 h) | 1× | ticket |

Burn rate = (error rate observed) ÷ (error rate the SLO allows). The short confirmation window stops a page from firing after the problem has already cleared.

### SLO capacity is not throughput capacity

The most important number for planning is how much load you can take **while meeting the SLO**, and it is much lower than how much the box can push. **Measured**, open-loop Poisson arrivals, SLO "p95 TTFT ≤ 2 s":

| Arrival rate | Utilisation | p50 TTFT | p95 TTFT | p99 TTFT | SLO |
|---|---|---|---|---|---|
| 0.16 req/s | 20% | 62 ms | 1.2 s | 1.7 s | pass |
| 0.24 req/s | 30% | 52 ms | 1.3 s | 2.2 s | pass |
| 0.32 req/s | 40% | 55 ms | 2.3 s | 3.8 s | **fail** |
| 0.40 req/s | 50% | 353 ms | 2.9 s | 4.8 s | **fail** |
| 0.57 req/s | 70% | 1.3 s | 7.8 s | 9.5 s | **fail** |
| 0.69 req/s | 85% | 1.9 s | 10.2 s | 11.8 s | **fail** |

The server's throughput ceiling was 0.81 req/s (§2), but the SLO held only up to roughly **30–37% of it**. Two things cause the gap. Random arrivals bunch up, so even at modest average load some requests arrive behind two or three others; and with a ~1.2 s service time, waiting behind two requests already costs 2.4 s. The p50 barely moves while the p95 explodes — which is why averages are the wrong thing to watch.

The levers that move SLO capacity are the ones that shorten the service time or serve requests in parallel: smaller `max_tokens`, a smaller model, prefix-cache hits, batching, more replicas ([05-deploy.md §6](05-deploy.md)).

**Gap.** shilp reports `finish_reason: "stop"` for every answer, including ones cut off by `max_tokens`, so the truncation SLI cannot be measured yet ([roadmap](08-service-roadmap.md), Phase 1).

---

## 2. Load testing

### Two ways to apply load

**Closed loop** — *N* simulated users, each sending its next request when the previous one finishes. It answers "how does the service behave with *N* users?". Its flaw: when the server slows down, the load generator slows down with it, so queueing is under-represented.

**Open loop** — requests arrive at a fixed average rate (Poisson), regardless of how the server is doing. It answers "what happens at this traffic level?" — which is the question an SLO asks. Latency must be measured from when each request was *due*, not when the generator got round to sending it. Measuring from the send time hides exactly the waiting you are trying to measure; this is called **coordinated omission**, and it is the most common reason load tests look better than production.

### The script

```bash
# closed loop: 4 users, 32 measured requests after 2 warmup requests
python scripts/loadtest.py --concurrency 4 --requests 32

# open loop at 0.3 req/s for two minutes, gated on an SLO (exit code 1 if it fails)
python scripts/loadtest.py --rate 0.3 --duration 120 --slo-ttft-p95 2000 --slo-error-rate 0.01

# the Ollama surface, machine-readable
python scripts/loadtest.py --surface ollama --rate 0.2 --duration 60 --json > run.json

make loadtest ARGS="--rate 0.3 --duration 120"
```

It streams every request and reports, separately: time to first token, end-to-end latency, time per output token (the per-stream speed), throughput in requests and output tokens per second, statuses, and per-prompt-class breakdowns. The default prompt mix is 60% short questions, 30% questions with a system prompt and 10% with a ~1.9k-token reference block that repeats across requests, as retrieved documents do. Pass `--mix short=0.2,long=0.8` to change it.

### Measured: the throughput curve

Closed loop, default mix, `max_tokens` 128:

| Users | Throughput | Output tok/s | p50 TTFT | p95 TTFT | Per-stream speed |
|---|---|---|---|---|---|
| 1 | 0.81 req/s | 87 | 29 ms | 41 ms | 92 tok/s |
| 2 | 0.80 req/s | 87 | 1.4 s | 1.5 s | 91 tok/s |
| 4 | 0.82 req/s | 87 | 3.7 s | 4.5 s | 91 tok/s |
| 8 | 0.83 req/s | 88 | 8.5 s | 9.9 s | 91 tok/s |

This is the signature of a server without batching: throughput and per-stream speed are constant, and each extra user adds one service time to everyone's wait. A batching server would show throughput rising with users and per-stream speed falling.

### Doing it properly

1. **Warm up first.** The first requests pay for paging in weights and filling caches. The script discards `--warmup` requests (default 2).
2. **Use realistic prompts.** Prompt length decides prefill time and output length decides decode time; a test of "hi" measures neither. If you have production traffic, sample its length distribution into a `--mix`, or replay real (scrubbed) prompts.
3. **Run long enough for the tail.** p95 is the 5% tail: with 40 samples it is decided by two requests. Aim for several hundred requests per point before quoting a p99. At high utilisation, also run long enough for the queue to reach steady state — in the measurements above, a 90-second run at 95% utilisation reported a *lower* p95 than one at 85%, because the queue had not finished growing when the test ended.
4. **Step the load.** Run a series of rates (or user counts) and plot p95 TTFT against load. The knee of that curve is your capacity; the SLO line crossing it is your SLO capacity.
5. **Test through the proxy.** Buffering, timeouts and connection limits live there. A test against `127.0.0.1` cannot see them.
6. **Soak.** Run at expected load for hours to catch memory growth, file-handle leaks and slow drift.
7. **Gate deploys.** The SLO flags make the script exit non-zero, so a pipeline can refuse to promote a model or config that misses the target.

---

## 3. Model versioning and rollout

Changing the model is the riskiest routine change to an LLM service: the API stays the same while every answer changes. Treat it like a code deploy.

### Identity: know exactly what is serving

shilp gives every model three layers of identity:

- **Content-addressed weights.** Blobs are stored by SHA-256 digest, so a weights file cannot change under a name.
- **A manifest digest.** `GET /api/tags` returns a `digest` per model — the hash of the manifest (weights, parameters, template, system prompt). Two models with the same digest behave identically.
- **A recipe.** How the model was produced: source repo and revision, quantisation, surgery, build steps ([02-reshape.md](02-reshape.md)).

Pin what you pull: `shilp pull hf:org/repo --revision <commit-sha>` records an exact upstream commit rather than a moving `main`. **Gap:** today the revision is honoured only for raw (safetensors) repos — for a GGUF pull such as `hf:org/repo-GGUF:q4_k_m` it is silently ignored and `main` is used. Until that is fixed ([roadmap](08-service-roadmap.md), Phase 1), record the manifest digest after pulling and treat *that* as the version.

### Aliases: give clients a stable name

Clients should not hard-code `google_gemma-3-1b-it-gguf:q4_k_m`. Give them a role name and point it at a concrete model. A Shilpfile whose `FROM` is an existing model creates a new tag that **shares the same weight blobs** — no copy:

```
# Shilpfile.assistant
FROM google_gemma-3-1b-it-gguf:q4_k_m
PARAMETER temperature 0.3
PARAMETER num_predict 384
SYSTEM """You are the internal engineering assistant. Be brief."""
```

```bash
shilp build -f Shilpfile.assistant --name assistant --tag prod      # clients call "assistant:prod"
shilp build -f Shilpfile.assistant.next --name assistant --tag canary
```

Promoting a new version is rebuilding `assistant:prod` from the new source; rolling back is rebuilding it from the old one. Keep the old blobs until the new version has proven itself — do not run `shilp gc` mid-rollout.

**Restart after repointing.** The loader caches loaded models by name, so a running server keeps serving the old weights for `assistant:prod` until it restarts (or the model is evicted). With several replicas, that makes a promotion a *rolling restart*. An unload endpoint, and reloading automatically when an alias's digest changes, are on the [roadmap](08-service-roadmap.md).

### Rollout strategies

| Strategy | How | Catches |
|---|---|---|
| **Offline evaluation** | run a fixed evaluation set against both versions before anything else | quality regressions on known tasks |
| **Shadow** | copy live requests to the new version, discard its answers, compare offline | regressions on real traffic, with zero user impact; doubles compute |
| **Canary** | route a small share (1–10%) of traffic to replicas serving the new version | errors, latency and behaviour changes on real users, limited blast radius |
| **Blue/green** | two full pools; switch the load balancer from one to the other | instant rollback; double capacity during the switch |

A sensible default for shilp-sized deployments: offline evaluation → canary on one replica behind the load balancer → promote by rolling restart → keep the previous alias target ready for a one-command rollback.

### What to compare during a canary

Error rate by status; p95 time to first token and stream speed; **output length distribution** (a template or stop-token change often shows up first as answers that are much longer or shorter); the 499 rate (users abandoning answers); finish reasons (once truncation is reported); and, if your app has them, thumbs-down and regenerate rates.

### Changes that are model changes

All of these change behaviour without changing the weights, and all of them belong in the same rollout process: the chat template, the system prompt, sampling defaults (`temperature`, `top_p`, `repeat_penalty`), stop sequences, `num_ctx`, and the quantisation level. shilp records them in the manifest, so its digest changes when any of them do.

---

## 4. Observability and privacy

### What exists

| Signal | Where | Contains |
|---|---|---|
| Per-request records | `$SHILP_HOME/logs/shilp.db` (SQLite) | surface, endpoint, model, status, prompt/completion tokens, TTFT, total time, tok/s — no prompt or answer text |
| Metrics | `GET /metrics` (Prometheus text) | request counts by status, token counts, mean durations, models resident |
| Health | `GET /health` | engine, store size, resident models |
| Access log | uvicorn's stdout | method, path, status — no bodies |

Percentile and saturation queries over the per-request records are in [05-deploy.md §11](05-deploy.md).

### A minimal dashboard

1. Requests per second, by status (200, 400, 499, 503, 500).
2. p50 and p95 time to first token.
3. p50 stream speed (tok/s).
4. Average requests in flight (saturation).
5. Prompt and completion tokens per request (p50, p95).
6. Error budget remaining, per SLO.

### Alerts worth having

- Burn-rate alerts for each SLO (§1).
- 499 rate above its baseline — users are giving up, usually before anyone complains.
- Stream speed below half its baseline — lost GPU offload, memory pressure or thermal throttling.
- Average in-flight approaching 1 on a serial server — the queue is forming.
- `/health` not answering, or `resident` empty after warmup.

### Request IDs and tracing

When a user reports a bad answer, you need to find *that* request. The standard is: accept an `X-Request-ID` header (or generate one), return it in the response, and attach it to every log line and the usage record. Tracing goes further, splitting each request into spans — queue wait, prefill, decode, streaming — so a slow request shows *where* it was slow. shilp has neither yet: the OpenAI surface generates a completion id, but it is not stored or logged. Both are Phase 1 items on the [roadmap](08-service-roadmap.md).

### Privacy: prompts are user data

Prompts and answers routinely contain personal data, credentials pasted by mistake, and confidential documents. The defaults that keep that manageable:

- **Do not store text by default.** shilp's telemetry stores counts and timings, never prompt or answer text — keep it that way if you extend it.
- **If you must log content** (debugging, evaluation): make it opt-in per tenant, set a retention period and enforce it, redact obvious identifiers before storage, restrict access, and support deletion per tenant.
- **Know where text lives transiently.** The KV and prefix caches hold the most recent prompt in memory; a crash dump or swap file can contain it. On shared hosts, encrypt swap and restrict core dumps.
- **Keep secrets out of artefacts.** The Hugging Face token belongs in the environment, never in a manifest, recipe or image layer.

---

## 5. Runbooks

Each entry is: what you see → what it usually means → how to check → what to do.

**p95 time to first token is rising; stream speed is normal.**
Queueing — load is past the SLO knee (§1). Check average in-flight and request rate ([05-deploy.md §11](05-deploy.md)). Shed or defer batch traffic, lower `max_tokens`, add a replica, or move to a batching engine. Do not raise timeouts — that only hides the queue.

**Stream speed dropped (tok/s down by a large factor).**
The model is no longer on the GPU, the machine is swapping, or it is thermally throttling. Check `SHILP_GPU_LAYERS`, that the build has Metal/CUDA (`shilp doctor`), memory pressure (`vm_stat` / `free -h`), and other processes on the GPU. Fix the cause; a restart alone will not.

**Single requests are slow even when idle.**
Long prompts without prefix-cache hits, a cold start, or CPU-only inference. Compare TTFT by prompt class in a load test; check that stable content comes first in prompts ([05-deploy.md §4.6](05-deploy.md)).

**Rising 400s mentioning the context window.**
Prompts have outgrown `num_ctx` — usually chat histories growing turn by turn. Trim the oldest turns on the client; raise `num_ctx` only if the memory budget allows (KV memory grows with it, per session).

**Rising 499s.**
Users or clients are leaving before answers finish: latency is too high, or a client/proxy timeout is shorter than generation time. Check TTFT and total-time percentiles; align timeouts (client < proxy < server).

**503 "model unavailable".**
A request asked for a different model while one was streaming, with `SHILP_MAX_RESIDENT=1`. Raise `SHILP_MAX_RESIDENT` if memory allows, or run one replica per model and route by model name.

**The process was killed (out of memory).**
Weights + KV × context × resident models exceeded memory ([05-deploy.md §4.1](05-deploy.md)). Lower `num_ctx`, `SHILP_MAX_RESIDENT`, or the quantisation level; check for other large processes.

**New requests hang, but `/health` answers.**
A long prefill is holding the model — cancellation takes effect between tokens, so a huge prompt finishes its prefill even after its client left, and requests whose clients left while queued still run their prefill when their turn comes. It clears on its own; prevent it with input caps and the cancellable admission queue (roadmap P1.3, P1.4).

**Streams arrive all at once at the end.**
A proxy is buffering. Set `flush_interval -1` (Caddy) or `proxy_buffering off` (nginx) for the inference paths.

**The model fails to load.**
Missing blob, wrong engine for the format, or a build without the expected backend. Run `shilp doctor` and `shilp show <model>`; re-pull if a blob is missing.

**Disk is full.**
Old model versions and interrupted downloads. `shilp ls`, remove unused tags with `shilp rm`, then `shilp gc` to delete unreferenced blobs.

---

## 6. Packaging and autoscaling

### Packaging

- **Keep weights out of the image.** A model is gigabytes and changes on its own schedule. Mount `SHILP_HOME` as a volume, or pre-pull into it with an init step, so images stay small and model changes do not need a rebuild.
- **Match the image to the accelerator.** CPU-only or CUDA builds of llama.cpp for Linux containers. **Metal does not pass through to Docker on macOS**, so on Apple Silicon run natively (a `launchd` or systemd service, as in [05-deploy.md §12](05-deploy.md)).
- **Start-up is three steps**: load the model, send a warmup request, *then* report ready. A readiness probe that passes before warmup sends the first users to a cold model.
- **Stop gracefully.** On `SIGTERM`: stop accepting, let in-flight streams finish, exit. Give the orchestrator a grace period longer than your longest expected stream.

shilp has no Dockerfile or Kubernetes manifests yet — that is on the [roadmap](08-service-roadmap.md).

### Autoscaling

- **Scale on load signals, not CPU.** For a GPU server, CPU utilisation is meaningless, and GPU utilisation is high whenever anything runs. Scale on requests in flight, queue depth or p95 time to first token.
- **Account for scale-out lag.** New capacity takes GPU node provisioning (often minutes), image pull, model load (a 40 GB model at 1 GB/s is 40 s before page-in) and warmup. Load arriving faster than that will queue; keep headroom — target 50–70% utilisation, not 90% — and set a minimum replica count of at least 1 for anything with an SLO.
- **Scale to zero only where cold starts are acceptable.** Development, batch jobs and internal tools tolerate a cold first request; interactive products usually do not.
- **Scale by model.** Different models have different costs and traffic; give each its own pool and its own scaling rule.

---

## 7. Quality in production

Latency and errors say whether the service is *up*. They say nothing about whether its answers are *good*, and a model change can make them worse while every dashboard stays green.

**Before release** — run a fixed evaluation set against the current and candidate versions with `temperature 0` and a fixed seed, and compare. Determinism is only approximate across hardware and builds, so compare scores, not exact strings. shilp has no evaluation harness yet; it is the top item of the [project roadmap](04-roadmap.md).

**During rollout** — the canary signals of §3: output length distribution, finish reasons, 499 rate, error frames, and user feedback if your application collects it.

**In steady state** — inputs drift even when the model does not: new document types, new languages, longer conversations. Watch prompt-length distributions and context-window rejections, and sample a small, privacy-reviewed set of conversations for human review on a schedule.

**Close the loop** — failures found in review become new evaluation items, so the next release is tested against them.

---

## Checklist: is it operable?

- [ ] SLOs written as fractions of good requests, with error budgets and burn-rate alerts
- [ ] SLO capacity measured with an open-loop load test, not inferred from throughput
- [ ] Load test in the deploy pipeline, gated on the SLO
- [ ] Clients call an alias (`assistant:prod`), not a concrete model
- [ ] Rollout procedure written: evaluate → canary → promote by rolling restart → rollback command
- [ ] Dashboard and alerts as in §4; 499 rate tracked separately
- [ ] Request IDs on every response and log line *(roadmap)*
- [ ] No prompt text stored by default; a retention policy if any is
- [ ] Runbooks reviewed by whoever is on call
- [ ] Readiness only after warmup; graceful shutdown longer than the longest stream
- [ ] Evaluation set run before every model change *(roadmap)*
