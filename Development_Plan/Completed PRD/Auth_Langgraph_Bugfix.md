Auth + LangGraph Bugfix: Keycloak SSO, Tenant Context, and Return-User Behavior

Purpose
- Fix incorrect tenant detection in LangGraph runs (tenant_id=None) even when FastAPI login succeeds.
- Make authentication consistent across FastAPI and LangGraph by validating Keycloak JWTs and extracting tenant context from JWT claims instead of headers.
- Ensure the orchestrator’s return-user detection works by passing tenant to LangGraph via auth + config and by setting thread metadata.

Why the bug happens (from logs)
- LangGraph dev server shows “auth type=noop”, so request headers (e.g., X-Tenant-ID) aren’t attached to background runs. Nodes can’t see the tenant from headers.
- Threads are auto-created without metadata before the first run, so `metadata.tenant_id` is empty.
- Nodes log `return_user_probe ... tenant_id=None` and behave like a new user (no cached discovery), even though FastAPI logs have the correct tenant (1105).

High‑level fix
1) Validate the user’s Keycloak JWT in the LangGraph server. Extract tenant from JWT claims (either a custom `tenant_id` or a role like `tenant-1105`).
2) If the token has no `tenant_id`, fall back to the FastAPI‑resolved tenant (see “FastAPI fallback” below) by injecting it into thread metadata and `configurable.tenant_id`.
3) Attach tenant to thread metadata and make it available to runs via LangGraph’s “configurable” context.
4) In the UI and FastAPI proxy paths, forward the user’s Bearer token to LangGraph. Don’t rely on `X-Tenant-ID` headers for run context.
5) Gate the first run until the UI knows the tenant (prevents races).

Keycloak setup (Admin Console)
1) Add a protocol mapper that injects tenant context into the token.
   - Realm → Clients → Your Client → Mappers → Create
   - Mapper Type: User Attribute (or Realm Role if you encode tenant via role)
   - Token Claim Name: `tenant_id`
   - Add to Access Token: enabled (ID Token optional if you use it on the client)
   - Save

   Alternative: encode tenant as a realm role like `tenant-1105` and map it into the token. Your app then parses the tenant id from that role string.

LangGraph auth plugin (server‑side)
Create `src/auth.py` and register it in `langgraph.json`. This validates Keycloak JWTs and exposes tenant/user in the auth context.

```python
# src/auth.py
import os
import jwt
import httpx
from functools import lru_cache

try:
    # LangGraph API server (preferred)
    from langgraph_api.auth import Auth  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for older SDKs – if unavailable, this module will not be loaded.
    Auth = None  # type: ignore

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "").rstrip("/")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "")
JWKS_URL = f"{KEYCLOAK_URL}/protocol/openid-connect/certs" if KEYCLOAK_URL else ""

if Auth is not None:
    auth = Auth()

    @lru_cache(maxsize=1)
    def _get_jwks_cached() -> dict:
        if not JWKS_URL:
            raise RuntimeError("KEYCLOAK_URL not configured")
        # Sync call is fine here; LangGraph server loads auth once.
        resp = httpx.get(JWKS_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _decode_token(token: str) -> dict:
        jwks = _get_jwks_cached()
        # Choose the first RSA key (standard single-key setup). If multiple keys, pick by 'kid'.
        key = next((k for k in jwks.get("keys", []) if k.get("kty") == "RSA"), None)
        if not key:
            raise jwt.InvalidTokenError("No RSA key in JWKS")
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
        return jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=CLIENT_ID or None,
            options={"verify_aud": bool(CLIENT_ID)},
        )

    @auth.authenticate
    async def authenticate(authorization: str | None):
        if not authorization:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Missing Authorization")
        try:
            scheme, token = authorization.split(" ", 1)
        except ValueError:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Malformed Authorization header")
        if scheme.lower() != "bearer" or not token:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid auth scheme")
        try:
            claims = _decode_token(token)
        except jwt.ExpiredSignatureError:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise Auth.exceptions.HTTPException(status_code=401, detail=f"Invalid token: {e}")
        # Extract tenant from custom claim or realm role
        tenant_id = claims.get("tenant_id")
        if not tenant_id:
            roles = (claims.get("realm_access") or {}).get("roles", [])
            tenant_role = next((r for r in roles if isinstance(r, str) and r.startswith("tenant-")), None)
            if tenant_role:
                tenant_id = tenant_role.replace("tenant-", "", 1)
        # NOTE: We do not 401 immediately if tenant is missing here; the
        # FastAPI gateway can inject tenant into thread metadata and/or
        # run config (configurable.tenant_id). Nodes read those later.
        return {
            "identity": claims.get("sub"),
            "email": claims.get("email"),
            "username": claims.get("preferred_username"),
            "tenant_id": str(tenant_id) if tenant_id is not None else None,
            "is_authenticated": True,
            # Preserve raw claims if you need them in advanced policies
            "claims": claims,
        }

    @auth.on
    async def attach_tenant_metadata(ctx: "Auth.types.AuthContext", value: dict):  # type: ignore[name-defined]
        """Scope threads/objects to tenant and owner via metadata and filters."""
        tid = (ctx.user or {}).get("tenant_id")
        owner = (ctx.user or {}).get("identity")
        # If tenant is not present in the token, allow the gateway to supply it
        # via metadata/configurable. This keeps prod strict while allowing a
        # fallback path when Keycloak claims don’t carry tenant.
        if not tid:
            tid = (value.get("metadata") or {}).get("tenant_id")
        if not tid:
            # As a last resort, tolerate missing tenant in dev; production
            # deployments should enforce a tenant here.
            dev_ok = (os.getenv("DEV_AUTH_BYPASS") or "").lower() in {"1","true","yes","on"}
            if not dev_ok:
                raise Auth.exceptions.HTTPException(status_code=401, detail="No tenant in auth context")
        metadata = value.setdefault("metadata", {})
        metadata.setdefault("tenant_id", tid)
        if owner:
            metadata.setdefault("owner", owner)
        # Return filters for list/search APIs
        return {"tenant_id": tid}
else:
    # If server cannot import Auth (very old versions), expose a stub so imports succeed.
    auth = None  # type: ignore
```

Register the auth plugin in `lead_generation-main/langgraph.json`:

```jsonc
{
  "dependencies": ["my_agent", "app", "src"],
  "graphs": {
    "orchestrator": "my_agent.agent:build_orchestrator_graph"
  },
  "auth": { "path": "src/auth.py:auth" },
  "env": ".env"
}
```

Environment
```
# Keycloak
KEYCLOAK_URL=https://keycloak.company.com/realms/your-realm
KEYCLOAK_CLIENT_ID=my-langgraph-app

# UI/Proxy (recommended for local dev)
NEXT_PUBLIC_USE_API_PROXY=true
NEXT_PUBLIC_USE_AUTH_HEADER=true
FASTAPI_API_URL=http://127.0.0.1:2024
LANGGRAPH_API_URL=http://127.0.0.1:8001
DEV_AUTH_BYPASS=true   # dev-only; allows fallback to gateway-supplied tenant

FastAPI fallback (when Keycloak token lacks tenant)
When Keycloak does not include a `tenant_id` claim (and you don’t parse it from roles), let FastAPI resolve tenant (e.g., from DB mapping by email/session), then inject it into LangGraph requests. Two places to do this:

1) Graph proxy injector (preferred in dev/local)
   - In `app/graph_proxy.py`, ensure the proxy merges a tenant into the run payload under `config.configurable`. If the client already sets it, the proxy should not overwrite it.

   Example change (pseudo‑diff):
   ```python
   # inside _forward(...), before sending request
   if body and json and ("/threads/" in path and "/runs" in path):
       payload = json.loads(raw or "{}")
       tenant_hdr = request.headers.get("x-tenant-id") or request.headers.get("X-Tenant-ID")
       # Prefer tenant resolved by FastAPI (e.g., request.state.tenant_id)
       tenant_fallback = getattr(getattr(request, "state", object()), "tenant_id", None)
       tenant = tenant_hdr or tenant_fallback
       if tenant:
           cfg = payload.get("config") or {}
           conf = cfg.get("configurable") or {}
           conf.setdefault("tenant_id", str(tenant))
           # also mirror into metadata/context for older nodes
           conf.setdefault("metadata", {"tenant_id": str(tenant)})
           conf.setdefault("context", {"tenant_id": str(tenant)})
           cfg["configurable"] = conf
           payload["config"] = cfg
           body = json.dumps(payload).encode("utf-8")
   ```

2) Server‑side chat endpoints (gateway mode)
   - If you start runs from FastAPI controller functions, include:
   ```python
   await client.runs.stream(
       thread_id=thread_id,
       assistant_id="orchestrator",
       input={"messages": [{"role": "user", "content": text}]},
       config={"configurable": {"tenant_id": str(tenant_id)}},
   )
   ```

Either path ensures `var_child_runnable_config` carries `tenant_id` that nodes can read, even when the Keycloak token does not include it.
```

FastAPI → LangGraph: forward the user’s token
- Your FastAPI already validates auth and resolves tenant. When proxying/creating runs on behalf of the user, forward their Keycloak token to LangGraph:
  - `Authorization: Bearer <user_token>`
  - Do not rely on `X-Tenant-ID` for run context (headers aren’t injected into background runs).

UI integration (Agent Chat)
- The UI already sets `defaultHeaders` with `Authorization` when `NEXT_PUBLIC_USE_AUTH_HEADER=true` and an idToken is present. Enable that flag.
- The stream client should send tenant in the run config. In this repo we’ve already added:
  - `config: { configurable: { tenant_id, metadata: { tenant_id }, context: { tenant_id } } }` in `agent-chat-ui/src/providers/Stream.tsx`.
- Pre-create threads with `metadata: { tenant_id }` and avoid auto-attaching to an old thread without metadata.
- Block the first run until tenant is known (prevents races). If you still race, explicitly `PATCH /threads/{id}` to set `metadata.tenant_id` before invoking a run.

Node access to tenant (already implemented)
- `my_agent/utils/nodes.py` resolves tenant from multiple sources:
  - `var_child_runnable_config` (configurable context),
  - LangGraph request metadata `_lg_get_current_metadata()` (when available),
  - `state.context` / `entry_context`.
- With the Keycloak auth + UI config above, `return_user_probe` and other nodes will log `tenant_context ... tenant_id=<id>` and behave as returning users when cached data exists.

Installation
```
pip install pyjwt[crypto] httpx
```

Expected logs after fix
- LangGraph starts without “auth type=noop”. You should see its custom auth load (auth import succeeds).
- Before a run, you should see a `POST /threads` (create) or `PATCH /threads/{id}` (metadata attached by the client).
- During `return_user_probe`, logs include `tenant_context ... tenant_id=1105` and no longer show `tenant_id=None`.

Troubleshooting
- 401 Invalid token: verify `KEYCLOAK_URL` and `KEYCLOAK_CLIENT_ID`; ensure the token’s `aud` includes your client id or disable audience verification by leaving `KEYCLOAK_CLIENT_ID` empty (not recommended for prod).
- Tenant still None: confirm the UI forwarded `Authorization` (set `NEXT_PUBLIC_USE_AUTH_HEADER=true`); if the Keycloak token truly lacks `tenant_id`, make sure the FastAPI fallback is injecting `config.configurable.tenant_id` or thread `metadata.tenant_id` via the proxy or gateway endpoints.
- Return-user still cold-starts: confirm tenant 1105 actually has cached profiles/Top‑10 rows; otherwise the probe will correctly report `cached=0` even with `tenant_id=1105`.

Why this resolves the bug
- Tenant is carried by the user’s JWT (source of truth) and validated by LangGraph itself.
- Threads/runs are scoped to the tenant via auth and config—no dependence on transient headers.
- The orchestrator’s existing context readers pick up tenant reliably, enabling return-user reuse.
