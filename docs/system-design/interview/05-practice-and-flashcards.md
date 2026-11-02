# 05 — Practice questions and flashcards

**Questions to practise with, each with an answer outline to check yourself against, followed by flashcards for quick recall.** Practise out loud, timed, drawing as you go — then compare with the outline. The outlines are not the only right answers; they show what a good answer covers.

---

## Part A — Estimation drills

Answers to the three questions at the end of [02-estimation.md](02-estimation.md).

**1. Photo-sharing app** — 10 M DAU, 2 uploads and 50 views per user per day, ~2 MB per photo.

| Quantity | Working | Answer |
|---|---|---|
| Uploads/day | 10 M × 2 | 20 M |
| Upload rate | 20 M ÷ 86,400 | ~230 /s |
| Storage/day | 20 M × 2 MB | ~40 TB |
| Storage/year | × 365 | **~15 PB** (before replication) |
| View rate | 10 M × 50 ÷ 86,400 | ~5,800 /s average, ~17,000 /s peak |

*Implication:* storage is the dominant problem → object storage, plus a CDN for views; generate thumbnails asynchronously.

**2. Chat app** — 50 M DAU, 40 messages each per day, ~100 bytes per message.

| Quantity | Working | Answer |
|---|---|---|
| Messages/day | 50 M × 40 | 2 billion |
| Write rate | 2 B ÷ 86,400 | ~23,000 /s average, **~70,000 /s peak** |
| Storage/day | 2 B × 100 B | 200 GB |
| Storage/year | × 365 | ~73 TB |

*Implication:* write-heavy → a database built for high write rates (wide-column, e.g. Cassandra), sharded by conversation id; WebSockets for delivery.

**3. LLM coding assistant** — 200,000 developers, 100 completions each per day, 2,000 prompt and 50 output tokens each.

| Quantity | Working | Answer |
|---|---|---|
| Requests/day | 200 k × 100 | 20 M |
| Peak RPS | 20 M ÷ 86,400 × 3 | ~700 |
| Prompt tokens/s at peak | 700 × 2,000 | ~1.4 million |
| Output tokens/s at peak | 700 × 50 | ~35,000 |

*Implication:* prompts are **40× longer** than outputs, so **prefill dominates** — the opposite of chat. Time to first token is the user-facing metric (completions must appear in ~200–500 ms). Levers: prefix caching (the same file context is re-sent on every keystroke), smaller fast models, servers near users, cancelling stale requests when the developer keeps typing.

---

## Part B — Design questions with answer outlines

Work through each with the eight steps of [03-interview-framework.md](03-interview-framework.md). The outline lists the points a strong answer reaches.

### B1. Design a rate limiter (classic, very common)

- **Requirements:** limit requests per user/API key per window; low added latency (< 1–2 ms); works across many servers; clear 429 response.
- **Algorithm:** token bucket (bursts + sustained rate); mention sliding window and fixed window, and why token bucket.
- **Where:** in the API gateway / middleware, before expensive work.
- **Storage:** Redis, key per user, atomic update with a Lua script (or `INCR` + `EXPIRE` for fixed windows); TTL to clean up.
- **Distributed issues:** race conditions (atomicity), clock skew (use Redis time), Redis failure (fail open with local limits).
- **Response:** 429 + `Retry-After` + remaining-quota headers.
- **Extension for LLMs:** limit tokens per minute as well, with reserve-and-refund.
- Reference: [02-rate-limiting-and-quotas.md](../02-rate-limiting-and-quotas.md).

### B2. Design a URL shortener

- **Estimates:** ~40 writes/s, ~4,000 reads/s, ~3 TB over 5 years, 7-character base62 ids ([02 §3](02-estimation.md)).
- **API:** `POST /shorten {url}` → `{short}`; `GET /{short}` → 301/302 redirect.
- **Id generation:** counter + base62 (needs a distributed counter or ranges per server), or a hash of the URL (handle collisions).
- **Storage:** key-value (short → long); read-heavy → cache (Redis) + CDN.
- **Extras:** expiry, custom aliases, analytics via an async queue (clicks → Kafka → aggregates), abuse (malicious URLs).

### B3. Design a chat application (WhatsApp/Slack-like)

- **Delivery:** WebSockets for real-time; a connection service tracking which server each user is connected to.
- **Storage:** messages in a write-optimised store, sharded by conversation; message ids ordered per conversation.
- **Offline users:** store and push notifications; deliver on reconnect.
- **Group chats:** fan-out on write vs read; large groups need different handling.
- **Guarantees:** at-least-once delivery + deduplication by message id; read receipts as events.

### B4. Design a news feed (Twitter/Instagram-like)

- **Core trade-off:** fan-out on write (push posts to followers' feeds at post time — fast reads, expensive for celebrities) vs fan-out on read (build the feed when requested — cheap writes, slow reads). Hybrid: push for normal users, pull for celebrities.
- **Storage:** posts in a database; feeds as cached lists (Redis) of post ids.
- **Ranking:** a ranking service; pagination with cursors.

### B5. Design a RAG (retrieval-augmented generation) question-answering system

- **Pipeline:** ingest documents → split into chunks → embed → store in a vector database; at query time embed the question → retrieve top-k chunks → build a prompt → LLM → answer with citations.
- **Estimates:** documents × chunks × embedding size → vector index size; query RPS → embedding + retrieval + LLM latency budget.
- **Latency budget example:** embed 20 ms + vector search 30 ms + re-rank 50 ms + LLM first token 400 ms → ~0.5 s to first token.
- **Deep dives:** chunking strategy; index type (HNSW vs IVF) and recall vs speed; keeping the index fresh on updates; prompt size (more chunks → better recall but slower, costlier prefill); caching frequent questions; evaluation (retrieval recall, answer faithfulness).
- **Access control:** filter retrieved chunks by the user's permissions — a classic miss.

### B6. Design an LLM inference API

- The full worked answer is [04-mock-interview-llm-inference.md](04-mock-interview-llm-inference.md). Checklist: tokens-based estimates; GPU count; continuous batching; KV-cache memory; TTFT and streaming targets; token-based rate limits; admission control; routing and prefix affinity; usage metering via events; model versioning; cost per token.

### B7. Design a job queue / background processing system

- **Components:** producers, a durable queue (Kafka, SQS, RabbitMQ), worker pools, a results store.
- **Guarantees:** at-least-once + idempotent jobs; visibility timeouts; retries with backoff; dead-letter queue.
- **Scaling:** partition by key for ordering; autoscale workers on queue depth.
- **LLM twist:** batch inference jobs (embedding a million documents) use the same GPUs at night, at lower priority than interactive traffic.

### B8. Design a distributed cache

- **Partitioning:** consistent hashing across nodes; virtual nodes for balance.
- **Replication:** for availability; read from replicas.
- **Eviction:** LRU, TTLs.
- **Problems:** hot keys (replicate hot keys, local caching), cache stampede (many misses at once → request coalescing, staggered TTLs), consistency with the database (cache-aside + invalidation on writes).

### B9. Design a metering and billing system for an API

- **Events:** one usage event per request with a server-generated event id; publish to a durable log.
- **Exactly-once billing:** at-least-once delivery + deduplication on event id.
- **Aggregation:** recompute per tenant × day from raw events; idempotent; handle late events.
- **Billing:** versioned price book, invoices, reconciliation against infrastructure cost; quotas read live counters + aggregates.
- Reference: [06-metering-and-billing.md](../06-metering-and-billing.md).

---

## Part C — Concept questions (short answers)

| Question | Answer |
|---|---|
| What is the difference between latency and throughput? | Latency: how long one request takes. Throughput: how many requests (or tokens) are completed per second. Batching raises throughput and may raise latency. |
| Why use p99 instead of the average? | Averages hide the slow requests users actually notice; the tail is what breaks user experience. |
| Vertical vs horizontal scaling? | Bigger machine vs more machines. Vertical is simpler but capped; horizontal scales further but needs stateless services and a load balancer. |
| Why should services be stateless? | Any instance can serve any request, so you can add/remove instances freely and survive failures. |
| Replication vs sharding? | Replication copies the same data (availability, read scale); sharding splits data (write and storage scale). |
| What does CAP say? | During a network partition you must choose consistency (refuse possibly-stale answers) or availability (answer anyway). |
| When is eventual consistency fine? | When brief staleness is harmless: like counts, feeds, usage dashboards — not balances or inventory at checkout. |
| Cache-aside — how does it work? | Read cache; on miss read the database and fill the cache; on write update the database and invalidate the cache entry. |
| What is idempotency and why does it matter? | Repeating an operation has the same effect as doing it once; it makes retries safe. |
| Why a message queue? | Decouple producers and consumers, absorb bursts, retry failed work, keep the user-facing path fast. |
| What is backpressure / load shedding? | Slowing producers or rejecting work when a system is full, so accepted work stays fast. |
| What is a circuit breaker? | Stop calling a failing dependency for a while and fail fast, instead of piling up waiting requests. |
| What is consistent hashing for? | Spreading keys across servers so adding/removing a server moves only ~1/N of the keys. |
| Why is LLM decode memory-bound? | Each output token requires reading all the weights; there is little arithmetic per byte read. |
| What is the KV cache and why does it limit concurrency? | Stored attention state per conversation; it grows with context length and number of conversations and must fit in GPU memory next to the weights. |
| What does continuous batching do? | Adds and removes requests from the running batch at every token step, so the GPU serves many conversations at once without waiting for the longest one. |
| TTFT vs TPOT? | Time to first token (queueing + prefill) vs time per output token (streaming speed). |
| How do you rate-limit an LLM API fairly? | Limit tokens per minute as well as requests, reserving `prompt + max_tokens` and refunding unused tokens. |
| Why are output tokens priced higher than input tokens? | Input is processed in parallel (prefill); output is generated one token at a time (decode), using far more GPU time per token. |

---

## Part D — Flashcards

Cover the right column and test yourself. Ten minutes a day.

| Term | Definition |
|---|---|
| Functional requirement | what the system does |
| Non-functional requirement | how well: latency, throughput, availability, consistency, cost |
| Latency | time for one request |
| Throughput | work completed per unit of time |
| p95 / p99 | 95% / 99% of requests are faster than this |
| Availability | fraction of time the system works ("nines") |
| 99.9% downtime per year | ~8.8 hours |
| 99.99% downtime per year | ~53 minutes |
| Durability | saved data is not lost |
| SPOF | single point of failure |
| SLI / SLO / SLA | indicator / objective / contract |
| Error budget | the failure an SLO allows |
| Vertical scaling | bigger machine |
| Horizontal scaling | more machines |
| Stateless service | keeps nothing between requests |
| Load balancer | spreads requests across servers, removes unhealthy ones |
| L4 vs L7 load balancer | forwards connections vs understands HTTP |
| Reverse proxy | front server for TLS, caching, limits, routing |
| API gateway | reverse proxy for APIs: auth, rate limits, routing |
| CDN | caches static content near users |
| Cache hit rate | fraction served from cache |
| LRU | evict the least recently used entry |
| TTL | time after which an entry expires |
| Cache-aside | app reads cache, fills it on a miss |
| Replication | copies of the same data on several machines |
| Replication lag | how far replicas are behind the primary |
| Failover | promoting a replica when the primary fails |
| Sharding | splitting data across machines by a key |
| Shard key | field that decides a row's shard |
| Hot spot | one shard/key receiving disproportionate load |
| Consistent hashing | ring placement; adding a node moves ~1/N keys |
| ACID | atomicity, consistency, isolation, durability |
| Strong vs eventual consistency | always latest vs converges later |
| CAP | during a partition, choose consistency or availability |
| Quorum | majority needed to agree (e.g. 3 of 5) |
| Message queue | durable buffer between producers and consumers |
| At-least-once | never lost, may repeat → needs idempotency |
| Idempotent | doing it twice = doing it once |
| Dead-letter queue | where repeatedly failing messages go |
| Rate limiter | caps requests per client per time |
| Token bucket | capacity C, refill r/s, each request takes a token |
| 429 | Too Many Requests |
| 503 | Service Unavailable (overloaded / down) |
| Backpressure | slow producers when consumers are full |
| Load shedding | reject some work to keep the rest fast |
| Circuit breaker | stop calling a failing dependency temporarily |
| Exponential backoff + jitter | growing, randomised retry delays |
| Little's law | in-flight = arrival rate × time in system |
| Utilisation ρ | arrival rate × service time; keep ≤ ~0.7 |
| Token (LLM) | ~¾ of an English word |
| Prefill | parallel processing of the prompt; sets TTFT |
| Decode | one token at a time; memory-bandwidth-bound |
| TTFT | time to first token |
| TPOT | time per output token (streaming speed) |
| KV cache | per-conversation attention memory on the GPU |
| Prefix cache | reuse KV for a shared prompt beginning |
| Continuous batching | batch re-formed at every token step |
| Paged attention | KV cache allocated in blocks on demand |
| Chunked prefill | split long prompts to avoid stalling others |
| Quantisation | fewer bits per weight: less memory, faster |
| Tensor parallelism | one model split across several GPUs |
| SSE | Server-Sent Events: one-way streaming over HTTP |
| RPM / TPM | requests / tokens per minute |

---

## Part E — A self-check before an interview

- [ ] I can do the eight steps in order without notes.
- [ ] I can turn "N users" into peak RPS, storage and machines in under five minutes.
- [ ] I know the latency numbers and the "nines" table.
- [ ] For every building block in [01](01-building-blocks.md), I can say what problem it solves and what it costs.
- [ ] I can draw the LLM inference design from [04](04-mock-interview-llm-inference.md) from memory and walk one request through it.
- [ ] I can explain why batching raises throughput, and what limits batch size.
- [ ] I have practised at least five questions from Part B out loud, timed.

Back to the [track overview](README.md).
