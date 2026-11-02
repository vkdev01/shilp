# 01 — Authentication

**Status:** ⬜ not implemented · **Blocks:** any exposure beyond `127.0.0.1` · **Roadmap:** P1.1 (single key), P1.2 (admin separation), P2.1 (per-tenant keys) · **Effort:** ~1 day for P1.1+P1.2, 2–3 days for P2.1

---

## 1. The problem

Nothing in `src/shilp/server/app.py` checks who is calling. Every route — including `/api/pull`, which downloads any model from the Hub, and `/api/delete`, which removes one — answers anyone who can reach the port.

Today that is safe only because `shilp serve` binds to `127.0.0.1` by default. The moment it listens on a network address, three things become true:

| Exposure | Consequence |
|---|---|
| Anyone can generate | at ~0.81 req/s of capacity (measured), a single script looping requests occupies the whole machine |
| Anyone can pull | `/api/pull` with a 40 GB repository fills the disk |
| Anyone can delete | `/api/delete` removes the model your users depend on |

And every other control in this folder — rate limits, quotas, fair queueing, metering, billing — needs to know **who** made a request. Without authentication, a thousand users and one user are indistinguishable. That is why authentication is not one feature among seven but the precondition for all of them.

---

## 2. Concepts

**Authentication vs authorisation.** *Authentication* answers "who is this?" (a key maps to a principal). *Authorisation* answers "may they do this?" (a principal's scopes allow `inference` but not `admin`). Both are needed: an inference key must not be able to delete models.

**Principal.** The identity a request acts as: a key id, the tenant that owns it, and its scopes. Every later system keys off it — rate limiters count per principal, meters charge per tenant.

**Bearer token.** A secret sent in `Authorization: Bearer <token>`. Whoever *bears* it is authenticated — so it must travel only over TLS and never appear in URLs or logs. OpenAI's API uses exactly this, which is why every OpenAI SDK can send a key to shilp with no changes (`OpenAI(base_url=..., api_key=...)`).

**API key anatomy.** Production keys have two parts:

```
shilp_sk_7Q2M4KXA_Jq3rVb0k-8YxgT1pW6c9LmZ2sHd4fNa5eUoRiC7tKyB
└──┬───┘ └──┬───┘ └──────────────────┬───────────────────────┘
 prefix   key id         secret: 32 random bytes, base64url (43 chars)
```

- The **prefix** makes a leaked key recognisable — to people reading logs and to secret scanners such as GitHub's.
- The **key id** is public and indexed; it lets the server find the key's record without scanning.
- The **secret** carries the security: 32 bytes is 256 bits of entropy. Guessing it is not a realistic attack — at a trillion guesses per second, the expected time is around 10⁵⁷ years.

**Storing keys.** Store only a hash of the secret. Because the secret is random and long, a fast hash (SHA-256) is enough — slow password hashes such as bcrypt exist to resist *dictionary* guessing, which does not apply to random keys. Show the full key once, at creation; afterwards nobody, including the server, can recover it.

**Constant-time comparison.** Compare hashes with `hmac.compare_digest`, which takes the same time whether the first byte or the last one differs, so response timing leaks nothing.

**Other mechanisms, and when they apply.**

| Mechanism | Use when |
|---|---|
| API keys (bearer) | machine-to-machine API access — the default for inference APIs |
| OAuth 2.0 / OIDC | a *person* logs in to your application; the app then calls shilp with its own key |
| JWT from a gateway | an upstream gateway already authenticated the caller and forwards a signed identity |
| mTLS | service-to-service traffic inside a private network |

---

## 3. Design options

| Option | How | Gives you | Costs |
|---|---|---|---|
| **A. Proxy only** | Caddy/nginx checks one shared bearer key, shilp stays on loopback ([05-deploy.md §12](../05-deploy.md)) | protection today, no code | one identity for everyone: no per-user limits or metering; admin endpoints still reachable unless the proxy filters paths |
| **B. Single key in the app** | `SHILP_API_KEY` checked by a FastAPI dependency | protection without a proxy; a refusal to bind publicly without a key; admin separation | still one identity |
| **C. Per-tenant keys in the app** | a key store: tenants, hashed keys, scopes; CLI to issue and revoke | identity per caller — the base for rate limits, metering, tenancy | a small database and key lifecycle to operate |
| **D. External identity** | an API gateway (Kong, Envoy with ext_authz, a cloud gateway) authenticates and forwards a trusted header | central policy for many services | shilp must trust a header, so it must be unreachable except through the gateway |

**Recommendation:** B now (P1.1), C next (P2.1), and keep A as defence in depth. D fits organisations that already run a gateway — the design below accepts a forwarded identity as an option for that case.

---

## 4. The design for shilp

### 4.1 Request flow

```
request ──► authenticate (dependency) ──► authorise (scope) ──► route ──► generate
               │                              │
               └─ 401 + WWW-Authenticate      └─ 403
```

The principal is attached to `request.state.principal` and, from there, to the telemetry record — the first step towards metering ([06](06-metering-and-billing.md)).

### 4.2 Scopes and routes

| Scope | Routes |
|---|---|
| `inference` | `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/api/chat`, `/api/generate`, `/api/embed` |
| `models:read` | `GET /v1/models`, `/v1/models/{id}`, `/api/tags`, `POST /api/show`, `GET /api/ps` |
| `usage:read` | `GET /api/usage` (in Phase 2: only the caller's own usage) |
| `admin` | `POST /api/pull`, `/api/delete` — **disabled over HTTP unless `SHILP_ENABLE_ADMIN=1`** |
| none (open) | `GET /health`; `GET /metrics` open on loopback, otherwise `admin` |

A single-key deployment (P1.1) gives the one key `inference`, `models:read` and `usage:read`; admin needs its own `SHILP_ADMIN_KEY`.

### 4.3 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SHILP_API_KEY` | unset | the single inference key (P1.1) |
| `SHILP_ADMIN_KEY` | unset | key for admin routes |
| `SHILP_ENABLE_ADMIN` | `0` | expose `/api/pull` and `/api/delete` over HTTP at all |
| `SHILP_AUTH` | `none` if no key is set, else `key` | `none`, `key` (P1.1), `keystore` (P2.1), `header` (trusted gateway) |
| `SHILP_TRUSTED_IDENTITY_HEADER` | unset | header carrying the principal when `SHILP_AUTH=header` |

Rule: **`shilp serve` refuses to start on a non-loopback address when `SHILP_AUTH=none`**, with a message naming the variable to set.

### 4.4 Code sketch (P1.1 + P1.2)

New module `src/shilp/server/auth.py`:

```python
"""Who is calling, and may they? Dependencies used by the routers."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from fastapi import HTTPException, Request

@dataclass(frozen=True)
class Principal:
    key_id: str
    tenant: str = "default"
    scopes: frozenset[str] = field(default_factory=frozenset)

ANONYMOUS = Principal("anonymous", scopes=frozenset({"inference", "models:read", "usage:read"}))

def _digest(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()

def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token else None

def authenticate(request: Request) -> Principal:
    settings = request.app.state.shilp.settings
    if settings.auth == "none":
        return ANONYMOUS
    token = _bearer(request)
    if token is None:
        raise HTTPException(401, "missing API key", headers={"WWW-Authenticate": "Bearer"})
    for expected, principal in settings.key_table():   # [(sha256, Principal), ...]
        if hmac.compare_digest(_digest(token), expected):
            request.state.principal = principal
            return principal
    raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})

def require(scope: str):
    def check(request: Request) -> Principal:
        principal = authenticate(request)
        if scope not in principal.scopes:
            raise HTTPException(403, f"this key lacks the {scope!r} scope")
        return principal
    return check
```

Wiring in `create_app` (`src/shilp/server/app.py`) — FastAPI applies a dependency to every route of a router:

```python
from fastapi import Depends
from .auth import require

app.include_router(openai_router, dependencies=[Depends(require("inference"))])
app.include_router(ollama_router, dependencies=[Depends(require("inference"))])
if settings.enable_admin:
    app.include_router(admin_router, dependencies=[Depends(require("admin"))])
```

That needs two small moves: `/api/pull` and `/api/delete` go from `routes_ollama.py` into a new `routes_admin.py` (same paths, so Ollama clients are unaffected), and read-only routes such as `/v1/models` either move to a `models:read` router or use a route-level dependency. `Settings` (`src/shilp/config.py`, a frozen dataclass) gains `auth`, `api_key`, `admin_key` and `enable_admin`, read in `from_env()`; `key_table()` returns the hashed keys so plaintext never sits in a comparison loop.

### 4.5 Per-tenant keys (P2.1)

A key store in `$SHILP_HOME/state/keys.db`:

```sql
CREATE TABLE tenants (id TEXT PRIMARY KEY, name TEXT, created REAL, disabled INTEGER DEFAULT 0);
CREATE TABLE keys (
  id         TEXT PRIMARY KEY,          -- the public key id, e.g. 7Q2M4KXA
  tenant_id  TEXT NOT NULL REFERENCES tenants(id),
  hash       BLOB NOT NULL,             -- sha256(secret)
  scopes     TEXT NOT NULL,             -- "inference,models:read"
  created    REAL NOT NULL,
  last_used  REAL,
  expires    REAL,
  revoked    REAL
);
```

Lookup is by key id (primary key), then one `compare_digest` — no table scan. Results are cached in-process for 30 seconds, which bounds **revocation latency** at 30 s; the cost of a lookup is tens of microseconds against a ~1.2 s request, so the cache exists for throughput under bursts, not latency.

CLI:

```bash
shilp keys create --tenant acme --scopes inference,models:read   # prints the key once
shilp keys list --tenant acme                                      # ids, scopes, last used
shilp keys revoke 7Q2M4KXA
```

**Rotation without downtime**: issue the new key, deploy it to clients, watch the old key's `last_used` stop moving, revoke it.

### 4.6 The browser chat page

`/chat` calls the API from the browser, so with authentication on it needs a key field. Keep the key in `sessionStorage` (cleared when the tab closes), not `localStorage`, and never put it in a URL. On loopback with `SHILP_AUTH=none` the page works as today.

---

## 5. Step-by-step implementation

| Step | Change | Files |
|---|---|---|
| 1 | Add `auth`, `api_key`, `admin_key`, `enable_admin` to `Settings` and `.env.example` | `config.py`, `.env.example` |
| 2 | Create `Principal`, `authenticate`, `require(scope)` | `server/auth.py` (new) |
| 3 | Move `/api/pull`, `/api/delete` to an admin router; include it only when enabled | `server/routes_ollama.py`, `server/routes_admin.py` (new), `server/app.py` |
| 4 | Apply `require(...)` to each router; keep `/health` open | `server/app.py` |
| 5 | 401/403 bodies in each dialect's shape (`{"error": {...}}` on `/v1`, flat on `/api`) | `server/app.py` |
| 6 | Refuse a non-loopback bind without auth | `cli.py` (`serve`) |
| 7 | Put `principal.key_id` on `RequestRecord` and a `key_id` column in telemetry | `server/telemetry.py`, `server/generate.py` |
| 8 | Key field in the chat page | `server/templates/chat.html`, `server/static/js/chat.js` |
| 9 | Docs: API contract in `03-serve.md`, deployment in `05-deploy.md` | docs |
| 10 (P2.1) | Key store, `shilp keys` CLI, scope checks per key | `keys.py` (new), `cli.py`, `server/auth.py` |

---

## 6. Testing and verification

- Every inference route returns **401** without a key and **200** with it (parametrised over all routes).
- A wrong key and a missing key both return 401 with `WWW-Authenticate: Bearer`.
- An inference key on `/api/pull` returns **403**; with `SHILP_ENABLE_ADMIN=0` the route is **404**.
- `/health` answers without a key.
- `shilp serve --host 0.0.0.0` with no key exits non-zero with a clear message.
- The official `openai` Python client works with `api_key=<key>` and raises `AuthenticationError` without it.
- The telemetry row of an authenticated request carries the key id.
- (P2.1) A revoked key stops working within the cache TTL; `last_used` updates.

---

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Key leaked in logs | uvicorn's access log records method, path and status, not headers; never accept keys in query strings; log only key ids |
| Key committed to a repository | the `shilp_sk_` prefix lets secret scanners catch it; revoke and reissue |
| Proxy bypass (client reaches shilp directly) | bind shilp to loopback or a private interface; with `SHILP_AUTH=header`, accept the header only from the gateway's address |
| Plain HTTP on a network | terminate TLS at the proxy; bearer tokens over HTTP are readable by anyone on the path |
| Lost admin key | it is only an environment variable in P1; reset it and restart. In P2, an admin can be re-created with the CLI on the host |
| Browser preflight requests (`OPTIONS`) rejected | when CORS is added, let `OPTIONS` through the auth dependency |

---

## 8. Status

| Item | Status |
|---|---|
| Bind to loopback by default | ✅ |
| Caller authentication | ⬜ P1.1 |
| Admin endpoints separated and off by default | ⬜ P1.2 |
| Refuse public bind without auth | ⬜ P1.1 |
| Key id on telemetry | ⬜ P1.1 (prepares [06](06-metering-and-billing.md)) |
| Per-tenant keys, scopes, rotation, revocation | ⬜ P2.1 |
| Interim: proxy with a shared key and a path allowlist | documented in [05-deploy.md §12](../05-deploy.md) |
