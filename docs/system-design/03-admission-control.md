# 03 — Request queue and admission control

**Status:** 🟡 an implicit, unbounded wait · **Blocks:** predictable latency under load · **Depends on:** nothing (fair queueing later needs [01](01-authentication.md)) · **Roadmap:** P1.4 (bounded FIFO queue, 429), P2.4 (fair queueing per tenant) · **Effort:** 2–4 days

---

## 1. The problem

### What happens today

When requests arrive faster than one model can serve them, shilp does this:

| Situation | Behaviour today | Where |
|---|---|---|
| Second request, **same** model | waits on the engine's `threading.Lock` until the first finishes | `engine/llamacpp.py` `_stream()` |
| Second request, **different** model, `SHILP_MAX_RESIDENT=1` | **503** immediately (`ModelNotReady`) — the resident model is busy and cannot be evicted mid-stream | `engine/loader.py` `_ensure()` |
| Tenth request, same model | also waits — there is no bound | — |
| A waiting client gives up | its request keeps its place, later runs its prefill, then notices and stops | `server/stream.py` checks the stop flag between chunks |

Three properties of that wait are wrong for a product:

1. **Unbounded.** Overload becomes ever-growing latency instead of a quick, retryable "no".
2. **Unordered.** Python's `threading.Lock` does not promise first-come-first-served.
3. **Expensive while idle.** Each waiting request already holds a worker thread blocked on the lock; a hundred waiting requests are a hundred threads.

### What it costs, in numbers

Measured service time is about **S = 1.23 s** per request (≈0.81 req/s). Suppose traffic arrives at twice that, **λ = 1.6 req/s**, for one minute:

| Policy | After 60 s |
|---|---|
| **Unbounded wait (today)** | (1.6 − 0.81) × 60 ≈ **47 requests waiting**; the next one waits ≈ 47 × 1.23 ≈ **58 s** for its first token — and every client has long since timed out, so the server spends a minute generating answers nobody reads |
| **Bounded queue, 4 waiting** | about half the requests get a **429 in milliseconds**; every accepted request starts within ≈ 4 × 1.23 ≈ **4.9 s** |

The second row serves the same number of requests per second — the box is saturated either way — but every answer it produces is still wanted, and every rejection is fast enough for the client to retry or fail over.

---

## 2. Concepts

### Every server has a queue

If work arrives faster than it is served, it waits somewhere: in the socket backlog, in a thread pool, or on a lock. *Admission control* means choosing **where** it waits, **how long**, **in what order**, and **what happens when the wait is full** — instead of letting the operating system decide.

### Little's law and utilisation

- **Little's law:** average number in the system = arrival rate × average time in the system. At 0.5 req/s and 8 s per request, 4 requests are in flight on average.
- **Utilisation:** ρ = λ × S. For random arrivals and one server (the M/M/1 model), the average time in the system is **S / (1 − ρ)**. With S = 2 s: ρ = 0.5 → 4 s, 0.8 → 10 s, 0.9 → 20 s, 0.95 → 40 s ([05-deploy.md §4.4](../05-deploy.md)).

The lesson: latency explodes long before throughput runs out. **Measured**, the SLO "95% of first tokens within 2 s" held only up to ~0.24 req/s — about 30% of the 0.81 req/s throughput ([07-operate.md §1](../07-operate.md)). A queue cannot raise that capacity; it decides what happens *above* it.

### The queue is a latency budget

If clients will wait at most **B** seconds for an answer to start and one request takes **S**, a single-slot server should hold at most

```
max_waiting = floor(B / S)
```

waiting requests. Anything beyond that will miss its deadline anyway — rejecting it now is kinder and cheaper. With S = 1.23 s:

| Wait budget B | max_waiting |
|---|---|
| 3 s | 2 |
| 6 s | 4 |
| 10 s | 8 |

### Load shedding and Retry-After

A rejected request should say when to come back. Two estimates, for two client behaviours:

```
time until the queue has room   ≈ S_avg / slots                  (the running request finishes, a waiter moves up)
time until it would likely run  ≈ (waiting + 1) × S_avg / slots
```

With 4 waiting, S_avg = 1.23 s and one slot: room opens in about **1–2 s**, but a request admitted then would start after ≈ 5 × 1.23 ≈ **6 s**. Send the first as `Retry-After` (rounded up, never below 1 s) with random jitter so a burst of rejected clients does not return in lockstep; clients that get a second 429 back off exponentially. Under sustained overload, the longer estimate is the honest one — some services send it to slow retries down.

### Ordering

| Policy | Use |
|---|---|
| **FIFO** | the default: fair by arrival, predictable |
| **Deadline / timeout** | drop a request that has waited longer than its client will — a waiting request is free to drop, a generating one is not |
| **Priority classes** | interactive before batch, with aging so batch is never starved |
| **Weighted fair queueing per tenant** | multi-tenant fairness: one tenant's burst cannot fill everyone's queue (P2.4, [07](07-multi-tenancy-and-abuse.md)) |
| **Adaptive LIFO under overload** | serve the newest first when the queue is long, because the oldest clients have probably given up — useful at very large scale, surprising to users at small scale |

### Head-of-line blocking

In a FIFO queue, one 30,000-token prompt delays everyone behind it by its whole prefill. Request limits ([02 §4.2](02-rate-limiting-and-quotas.md)) cap the worst case; batching engines with chunked prefill remove most of it ([04](04-batching-and-concurrency.md)).

### Slots

A **slot** is one request the engine can run at the same time. For shilp's llama.cpp engine today, a model has **one** slot. A batching engine has **N** (e.g. `llama-server --parallel 4`). The queue's job is the same either way: at most N running, at most `max_waiting` waiting, the rest rejected.

---

## 3. Design options

| Option | Notes |
|---|---|
| **A. At the proxy** (nginx/Envoy max connections and queue) | cannot see which model a request is for, nor tokens; coarse |
| **B. A semaphore per model** | bounds concurrency, but `asyncio.Semaphore` waiters are not removable when a client leaves, and there is no depth to report |
| **C. An explicit queue per model** — a deque of futures, slot count, bound, timeout, cancellable waits | exact control, metrics, FIFO, cancellation while queued |

**Recommendation:** C, in the loader layer, so every route and engine benefits.

---

## 4. The design for shilp

### 4.1 Where it sits

```
route ─► prime(stream_chat(...)) ─► admission.enter(model) ─► loader.acquire(model) ─► engine
                                        │      │
                         QueueFull ─────┘      └──── QueueTimeout
                         → 429 Retry-After           → 503
```

Because routes already await the first chunk before sending headers (`prime()` in `server/generate.py`), an exception raised while waiting in the queue becomes a normal status code, even for `stream: true`. That is the property that makes a streaming 429 possible.

### 4.2 The queue

New module `src/shilp/engine/admission.py` (sketch):

```python
class QueueFull(ShilpError):  ...       # → 429 with Retry-After
class QueueTimeout(ShilpError): ...     # → 503

class ModelQueue:
    """At most `slots` running, at most `max_waiting` waiting, FIFO, cancellable."""

    def __init__(self, slots: int, max_waiting: int, max_wait_s: float):
        self.slots, self.max_waiting, self.max_wait_s = slots, max_waiting, max_wait_s
        self.running = 0
        self.waiters: deque[asyncio.Future] = deque()
        self.service = Ewma(initial=1.0)        # recent seconds per request, for Retry-After

    async def enter(self, gone: Callable[[], Awaitable[bool]]) -> None:
        if self.running < self.slots and not self.waiters:
            self.running += 1
            return
        if len(self.waiters) >= self.max_waiting:
            raise QueueFull(retry_after=self.retry_after())
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        try:
            await wait_unless_gone(fut, gone, timeout=self.max_wait_s)   # raises on timeout/disconnect
        except BaseException:
            if fut in self.waiters:
                self.waiters.remove(fut)        # a client who left frees its place at once
            elif fut.done() and not fut.cancelled():
                self.leave()                     # we were granted a slot just as we left: pass it on
            raise

    def leave(self, seconds: float | None = None) -> None:
        if seconds is not None:
            self.service.update(seconds)
        while self.waiters:
            fut = self.waiters.popleft()
            if not fut.done():
                fut.set_result(None)            # hand the slot over; `running` stays the same
                return
        self.running -= 1

    def retry_after(self) -> int:
        return max(1, math.ceil(self.service.value / self.slots))   # + jitter when sent
```

`wait_unless_gone` waits for the future, a timeout, or the client disconnecting, whichever comes first. **The disconnect check is essential**: the framework does not cancel a route handler that is still waiting before it has returned a response, so the queue must poll `await request.is_disconnected()` (every ~250 ms) itself. This closes the gap where an abandoned request kept its place and ran its prefill.

### 4.3 Wiring

- `AppState` gets an `admission` registry: one `ModelQueue` per model, created on first use. The slot count must be known **before** the engine exists — admission runs before `loader.acquire()`, which is what loads the engine — so it comes from configuration, not from the engine object: `loader.slots_for(manifest)` returns 1 for the `llamacpp` and `transformers` engines and `SHILP_SLOTS` for a batching engine, decided from the manifest's engine and the settings.
- `stream_chat` / `stream_generate` in `server/generate.py` wrap `loader.acquire(ref)` in `admission.enter(...)` / `leave(...)`, and pass the elapsed service time to `leave()`.
- They need a "has the client gone?" callable, so the routes pass `request.is_disconnected` down.
- `app.py` maps `QueueFull` → **429** with `Retry-After` and a `rate_limit_error`-style body (code `server_overloaded`), and `QueueTimeout` → **503**.
- The engine's `threading.Lock` stays as a safety net; with one slot per model it is never contended.
- Worker threads are started only after admission, so waiting requests no longer each hold a thread.

### 4.4 Different models

Keep today's fast **503** when another model is resident and busy and `SHILP_MAX_RESIDENT` does not allow both. Queueing across models would mean evicting and loading weights back and forth (seconds for a small model, much longer for large ones), which wastes the very capacity the queue protects. The scaled answer is one replica pool per model, routed by model name ([05](05-replicas-and-load-balancing.md)).

### 4.5 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SHILP_MAX_WAITING` | `4` | waiting requests per model before 429 |
| `SHILP_MAX_QUEUE_WAIT` | `10` s | longest a request may wait before 503 |
| `SHILP_SLOTS` | from the engine | concurrent requests per model |

### 4.6 Metrics

| Metric | Type | Use |
|---|---|---|
| `shilp_queue_depth{model}` | gauge | the leading indicator of overload; alert when it sits at the bound |
| `shilp_queue_wait_ms{model}` | histogram | how much of time-to-first-token is waiting |
| `shilp_admission_rejected_total{model,reason}` | counter | `full`, `timeout`, `gone` |
| `shilp_requests_running{model}` | gauge | slots in use |

### 4.7 Fair queueing (P2.4)

With tenants ([07](07-multi-tenancy-and-abuse.md)), replace the single deque with one deque per tenant served by **deficit round robin**: each tenant gets a quantum of cost (e.g. estimated tokens) per round, so a tenant sending 50 requests cannot push another tenant's single request to the back. Worked example on one slot with S = 1.23 s: tenant A has 4 requests waiting, tenant B sends one. FIFO puts B fifth, ≈ 4.9 s of waiting; round robin puts B second, ≈ 1.2 s.

---

## 5. Step-by-step implementation

| Step | Change | Files |
|---|---|---|
| 1 | `ModelQueue`, `QueueFull`, `QueueTimeout`, `Ewma`, `wait_unless_gone` | `engine/admission.py` (new) |
| 2 | Registry on `AppState`; slot count from the engine | `server/state.py`, `engine/loader.py` |
| 3 | Wrap acquire with enter/leave; pass a disconnect callable from the routes | `server/generate.py`, `routes_*.py` |
| 4 | Exception handlers: 429 + `Retry-After`, 503 | `server/app.py` |
| 5 | Settings and `.env.example` | `config.py` |
| 6 | Metrics above | `server/generate.py`, `server/metrics.py` |
| 7 | Docs: replace "waits on the engine lock" wherever it appears | `docs/` |
| 8 (P2.4) | Per-tenant deques with deficit round robin | `engine/admission.py` |

---

## 6. Testing and verification

- **FIFO**: with one slot held, three waiters are admitted in arrival order.
- **Bound**: with `max_waiting=2`, the third waiter gets `QueueFull` with a `Retry-After` of at least 1 s; through HTTP it is a 429, also for `stream: true`.
- **Timeout**: a waiter past `max_wait_s` gets 503 and is removed.
- **Leaving while queued**: a waiter whose client disconnects is removed, and the engine is **never called** for it (assert on the fake engine's call log) — the test that proves the old gap is closed.
- **Slot hand-off race**: a waiter granted a slot at the moment it is cancelled passes the slot on (no leak: `running` returns to 0).
- **Load test**: `scripts/loadtest.py --rate` at 2× capacity shows bounded p95 time-to-first-token for accepted requests and a 429 share near 50%; at 0.5× capacity, no 429s.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Bound too small → 429s at moderate load | size from the wait budget; watch the rejected counter against the queue-depth gauge |
| Slot count wrong for the engine | derived from configuration by `loader.slots_for()`, checked against the engine after load (a mismatch is logged and the queue resized) |
| Bound too large → long waits, timeouts at the client | same formula; clients should send a deadline (future: honour an `X-Request-Timeout` header) |
| Slot leak (a path that never calls `leave`) | `leave` in `finally`; a test asserting `running == 0` after every scenario |
| All clients retry at the same second | jitter on `Retry-After`; exponential backoff in clients |
| One huge request blocks the queue | input caps ([02 §4.2](02-rate-limiting-and-quotas.md)); chunked prefill in a batching engine |

---

## 8. Status

| Item | Status |
|---|---|
| Event loop not blocked during generation (worker threads) | ✅ — but `/api/pull` runs its download on the event loop; it must move to a thread (or off the inference server, [01](01-authentication.md) admin separation) |
| Model never evicted mid-stream; different model while busy → 503 | ✅ |
| Cancellation of a *generating* request (499) | ✅ |
| Bounded queue with 429 + `Retry-After` | ⬜ P1.4 |
| FIFO order, queue timeout | ⬜ P1.4 |
| Removal of a request whose client leaves while *waiting* | ⬜ P1.4 |
| Queue depth / wait metrics | ⬜ P1.4 |
| Fair queueing per tenant | ⬜ P2.4 |
