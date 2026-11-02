# System design: turning shilp into an inference service

**A separate set of design documents for the seven pieces a model server needs before other people's products can depend on it — each explained from first principles, worked through with numbers, and mapped onto this codebase step by step.**

The rest of `docs/` explains how shilp works today. This set is about what it must *become*, and how. Each topic document follows the same shape:

1. **The problem** — what goes wrong without it, with numbers.
2. **Concepts** — the ideas you need, explained from the ground up.
3. **Design options** — the realistic choices and their trade-offs.
4. **The design for shilp** — the recommended choice, the files it touches, configuration, and code sketches written against the real modules.
5. **Step-by-step implementation** — small, testable steps in order.
6. **Testing and verification** — how to prove each step works.
7. **Failure modes** — what breaks, and what the design does about it.
8. **Status** — what is in place today and what needs doing.

No new measurements were taken for this set. Numbers marked **measured** come from earlier runs recorded in [05-deploy.md](../05-deploy.md) and [07-operate.md](../07-operate.md) (an M3 Pro with Gemma-3-1B Q4_K_M); numbers marked **computed** are arithmetic from stated inputs; everything else is an **example** chosen to illustrate.

---

## Reading order

| # | Document | Blocks | Effort (estimate) | Status today |
|---|---|---|---|---|
| — | [00 — Current state and capabilities](00-current-state.md) | — | — | where the project is now |
| 1 | [01 — Authentication](01-authentication.md) | **any exposure** | 1 day (single key) · 2–3 days (per-tenant keys) | ⬜ none |
| 2 | [02 — Rate limiting and quotas](02-rate-limiting-and-quotas.md) | multi-user | 2–3 days | ⬜ none |
| 3 | [03 — Request queue and admission control](03-admission-control.md) | predictable latency under load | 2–4 days | 🟡 implicit, unbounded wait on a lock |
| 4 | [04 — Concurrency and batching](04-batching-and-concurrency.md) | throughput beyond one stream | 1–2 weeks | ⬜ one sequence at a time |
| 5 | [05 — Replicas and load balancing](05-replicas-and-load-balancing.md) | availability and horizontal scale | 3–5 days + infrastructure | 🟡 stateless API, no readiness or drain |
| 6 | [06 — Metering and billing](06-metering-and-billing.md) | charging, quotas, cost per team | 3–4 days (metering) · +1 week (billing) | 🟡 per-request rows, no key or aggregation |
| 7 | [07 — Multi-tenancy and abuse controls](07-multi-tenancy-and-abuse.md) | sharing between customers | 1–2 weeks | 🟡 no prompt storage; nothing else |
| — | [08 — Implementation roadmap](08-implementation-roadmap.md) | — | ~8–11 engineer-weeks for all of it | the plan, with dependencies and acceptance criteria |

Effort assumes one engineer who knows this codebase, including tests and documentation; it excludes provisioning infrastructure such as Redis, Postgres or a GPU cluster.

**Legend** used throughout: ✅ implemented and tested · 🟡 partly there, with the gap stated · ⬜ not implemented.

---

## The one-paragraph version

shilp today is a correct, observable, **single-user** model server ([00](00-current-state.md)). To serve a team it needs **authentication** (no caller can be limited, metered or billed until it is known), **request limits** and a **bounded queue** (so overload becomes a fast `429` instead of unbounded waiting). To serve many users it needs **batching**, because without it throughput is flat no matter how requests are queued — that is an engine change, not a scheduling one. To serve many customers it needs **per-key rate limits**, **metering** that can be aggregated and invoiced, and **tenancy** rules for data and abuse. **Replicas** behind a health-aware load balancer multiply whatever one process can do, once its state (rate counters, usage) lives outside it.

## How the pieces depend on each other

```
                    ┌──────────────────────┐
                    │ 1. Authentication    │  who is calling?
                    └──────────┬───────────┘
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ 2. Rate limits   │ │ 6. Metering      │ │ 7. Tenancy &     │
│    & quotas      │ │    & billing     │ │    abuse         │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         ▼                    │                    │
┌──────────────────┐          │                    │
│ 3. Admission     │◄─────────┴────────────────────┘  fair queueing needs tenants
│    control       │
└────────┬─────────┘
         │ the queue feeds…
         ▼
┌──────────────────┐        ┌──────────────────┐
│ 4. Batching      │───────►│ 5. Replicas & LB │  shared counters and usage
│    (engine)      │        │                  │  must move out of the process
└──────────────────┘        └──────────────────┘
```

Authentication is first because everything else needs an identity. Batching is independent of identity and can be built in parallel, but it changes how admission control should be sized (one slot per model becomes *N*). Replicas come last because they force the in-process state — rate counters, usage records — into shared stores.

## Relationship to the other docs

| If you want… | Read |
|---|---|
| Concepts in brief, as a glossary-level overview | [06-service-fundamentals.md](../06-service-fundamentals.md) |
| Capacity arithmetic and measurements | [05-deploy.md](../05-deploy.md) |
| SLOs, load testing, runbooks, rollout | [07-operate.md](../07-operate.md) |
| The full readiness matrix, beyond these seven topics | [08-service-roadmap.md](../08-service-roadmap.md) |
| **The design and build plan for the seven topics** | **this folder** |

The roadmap item ids used here (P1.1, P2.2, …) are the same ids as in [08-service-roadmap.md](../08-service-roadmap.md), so the two stay in step.
